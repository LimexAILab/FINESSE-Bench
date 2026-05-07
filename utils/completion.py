"""Completion API client with config management and environment variable support."""

import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml
from dotenv import load_dotenv

# Load .env from project root or utils directory
env_files = [
    Path(__file__).parent.parent / ".env",
    Path(__file__).parent / ".env",
]

for env_file in env_files:
    if env_file.exists():
        load_dotenv(env_file)
        break

API_MAX_RETRY = 3
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = None

registered_api_completion: Dict[str, Callable] = {}
registered_engine_completion: Dict[str, Callable] = {}


def register_api(api_type: str) -> Callable:
    """Decorator to register API completion function."""
    def decorator(func: Callable) -> Callable:
        registered_api_completion[api_type] = func
        return func
    return decorator


def register_engine(engine_type: str) -> Callable:
    """Decorator to register engine completion function."""
    def decorator(func: Callable) -> Callable:
        registered_engine_completion[engine_type] = func
        return func
    return decorator


def load_questions(question_file: str) -> List[Dict]:
    """Load questions from JSONL file."""
    questions = []
    with open(question_file, "r", encoding="utf-8") as f:
        for line in f:
            if line:
                questions.append(json.loads(line))
    return questions


def get_endpoint(endpoint_list: Optional[List[Dict]]) -> Optional[Dict]:
    """Randomly select one endpoint from list."""
    if not endpoint_list:
        return None
    import random
    return random.choice(endpoint_list)


def substitute_env_vars(config: Any) -> Any:
    """Recursively substitute ${VAR_NAME} with environment variables.
    
    Supports fallback: ${VAR_NAME:-default_value}
    """
    if isinstance(config, dict):
        return {k: substitute_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [substitute_env_vars(item) for item in config]
    elif isinstance(config, str):
        def replace_func(match: re.Match) -> str:
            var_with_fallback = match.group(1)
            
            if ':-' in var_with_fallback:
                var_name, default_value = var_with_fallback.split(':-', 1)
                return os.getenv(var_name.strip(), default_value.strip())
            else:
                var_name = var_with_fallback.strip()
                value = os.getenv(var_name)
                if value is None:
                    raise ValueError(f"Environment variable not found: {var_name}")
                return value
        
        return re.sub(r'\$\{([^}]+)\}', replace_func, config)
    else:
        return config


def make_config(config_file: str) -> Dict:
    """Load YAML config and substitute environment variables."""
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    
    return substitute_env_vars(config)


def extract_output(completion):
    # 1. Если это словарь (как последний пример)
    if isinstance(completion, dict):
        content = completion.get("answer", "")
        reasoning = completion.get("reasoning", None)
        # Если reasoning явно None, пробуем вытащить reasoning по тегу <think>
        if not reasoning and content:
            think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL | re.IGNORECASE)
            if think_match:
                reasoning = think_match.group(1).strip()
            # Ответ — содержимое <answer>...</answer>, либо всё после <answer>
            answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL | re.IGNORECASE)
            if answer_match:
                answer = answer_match.group(1).strip()
            else:
                # если нет <answer>, берём всё после </think>
                after_think = content.split("</think>")[-1].strip() if "</think>" in content else content.strip()
                answer = after_think
        else:
            answer = content.strip()
        finish_reason = completion.get("finish_reason")
        native_finish_reason = completion.get("native_finish_reason")
        return {
            "answer": answer,
            "reasoning": reasoning,
            "finish_reason": finish_reason,
            "native_finish_reason": native_finish_reason,
        }

    # 2. "Классический" вариант: completion.choices[0].message
    message = completion.choices[0].message
    # content и reasoning как атрибуты или как dict
    if isinstance(message, dict):
        content = message.get("content", "")
        reasoning = message.get("reasoning")
        tool_calls = message.get("tool_calls", [])
    else:
        content = getattr(message, "content", "")
        reasoning = getattr(message, "reasoning", None)
        tool_calls = getattr(message, "tool_calls", [])

    # Если reasoning отсутствует, пробуем вытащить из контента (локальный формат)
    if (not reasoning) and content:
        # 2а. Секция ## Thinking ... ## Final Response
        match = re.search(r"## Thinking\s*(.*?)\s*## Final Response", content, re.DOTALL)
        if match:
            reasoning = match.group(1).strip()
        # 2б. Теги <think>...</think>
        if not reasoning:
            think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL | re.IGNORECASE)
            if think_match:
                reasoning = think_match.group(1).strip()
        # Ответ после ## Final Response или внутри <answer>
        answer = None
        answer_match = re.search(r"## Final Response\s*(.+)", content, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
        if not answer:
            answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL | re.IGNORECASE)
            if answer_match:
                answer = answer_match.group(1).strip()
        if not answer:
            after_think = content.split("</think>")[-1].strip() if "</think>" in content else content.strip()
            answer = after_think
    else:
        answer = content.strip() if content else ""
        if reasoning and reasoning in answer:
            answer = answer.replace(reasoning, "", 1).strip()

    return {
        "answer": answer,
        "reasoning": reasoning,
        "tool_calls" : tool_calls,
        "finish_reason": getattr(completion.choices[0], "finish_reason", None),
        "native_finish_reason": getattr(completion.choices[0], "finish_reason", None),
    }

@register_api("openai_struct_outputs_for_judge")
def judge_response_parse_openai(model, messages, temperature, max_tokens, api_dict=None, **kwargs):
    
    class JudgeVerdict(BaseModel):
        explanation: str = Field(
            ...,
            description="A clear explanation of why the candidate answer is correct or incorrect, based on comparison with the ground truth and the evaluation rules."
        )
        final_verdict: Literal["0", "1"] = Field(
            ...,
            description='The final judgment of consistency: "1" if the candidate answer is consistent with the ground truth, or "0" if it is not.'
        )
    
    import openai
    if api_dict:
        client = openai.OpenAI(
            base_url=api_dict["api_base"],
            api_key=api_dict["api_key"],
        )
    else:
        client = openai.OpenAI()

    if api_dict and "model_name" in api_dict:
        model = api_dict["model_name"]
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            response = client.responses.parse(
                model=model,
                input=messages,
                text_format=JudgeVerdict,
                temperature=temperature,
                max_output_tokens=max_tokens,
                )
            parsed = response.output_parsed
            output = {
                "answer": getattr(parsed, "final_verdict", ""),
                "reasoning": getattr(parsed, "explanation", ""),
                "finish_reason": getattr(response.output[0], "status", ""),
                "native_finish_reason": getattr(response.output[0], "status", ""),
            }
        except openai.RateLimitError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
        except openai.BadRequestError as e:
            print(messages)
            print(type(e), e)
        except KeyError:
            print(type(e), e)
            break
    
    return output


@register_api("openai")
def chat_completion_openai(model, messages, temperature, max_tokens, api_dict=None, tools=None, **kwargs):
    import openai
    if api_dict:
        client = openai.OpenAI(
            base_url=api_dict["api_base"],
            api_key=api_dict["api_key"],
        )
    else:
        client = openai.OpenAI()
        
    if api_dict and "model_name" in api_dict:
        model = api_dict["model_name"]
    
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            # Build completion kwargs
            completion_kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            
            # Add tools if provided
            if tools:
                completion_kwargs["tools"] = tools
            
            completion = client.chat.completions.create(**completion_kwargs)
            # with open("/workspace/hayrapetyan/completions.jsonl", "a") as f:
            #     import json
            #     f.write(json.dumps(completion.to_dict()) + "\n")
            output = extract_output(completion)
            if tools:
                # Add completion object for tool call extraction
                output["completion"] = completion
            break
        except openai.RateLimitError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
        except openai.BadRequestError as e:
            print(messages)
            print(type(e), e)
        except KeyError:
            print(type(e), e)
            break
    
    return output


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["uid"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


def batch_submit_sglang(
    executor, 
    tokenizer, 
    temperature, 
    max_tokens, 
    all_context,
    max_context_length=None,
    end_think_token=None,
):
    print(f"DEBUG: sglang_completion_qwq: max_context_length: {max_context_length}")
    
    sampling_params = {
        "temperature": temperature,
        "skip_special_tokens": False,
        "max_new_tokens": max_tokens - 1,
        "no_stop_trim": True,
    }
        
    batch_prompt_token_ids = []
    batch_uids =[]
    uid_to_prompt = {}
    uid_to_response = {}
    
    for context in all_context:
        prompt_token_ids = tokenizer.apply_chat_template(
            context['turns'],
            add_generation_prompt=True,
            tokenize=True,
        )
        
        if max_context_length and (len(prompt_token_ids) + max_tokens) > max_context_length:
            print(f"DEBUG: sglang_completion_qwq: context length ({len(prompt_token_ids) + max_tokens}) > max_context_length ({max_context_length}), skip this context")
            continue
        
        batch_prompt_token_ids.append(prompt_token_ids)
        batch_uids.append(context['uid'])
        
        uid_to_prompt[context['uid']] = context['turns']
        
    err_msg = f"ERROR: len(batch_prompt_token_ids): {len(batch_prompt_token_ids)} != len(batch_uids): {len(batch_uids)}"
    assert len(batch_prompt_token_ids) == len(batch_uids), err_msg
    
    _ = executor.submit(
        prompt_token_ids=batch_prompt_token_ids,
        sampling_params=[sampling_params] * len(batch_uids),
        keys=batch_uids,
    )
    
    for request in tqdm(executor.as_completed(), total=len(batch_uids)):
        uid = request.key()
        result = request.result()
        raw_response = tokenizer.decode(
            result['output_ids'],
            skip_special_tokens=True,
        )
        
        if end_think_token:
            thought, _, ans = raw_response.partition(end_think_token)
            if ans == "":
                uid_to_response[uid] = {"thought": thought, "answer": raw_response}
            else:
                uid_to_response[uid] = {"thought": thought, "answer": ans}
        else:
            uid_to_response[uid] = {"answer": raw_response}
    
    # assert len(uid_to_response) == len(all_context), f"ERROR: len output ({len(uid_to_response)}) != len input ({len(all_context)})"
    return uid_to_response


def _infer_cuda_tp_world_size():
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cuda_devices is None:
        tp_world_size = 8
    else:
        tp_world_size = len(cuda_devices.split(","))
    return tp_world_size


def download_model(model: str, max_workers: int = 64):
    import subprocess
    
    env = os.environ.copy()
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    
    cmd = [
        "huggingface-cli", 
        "download", 
        f"--max-workers={max_workers}", 
        model
    ]
    
    try:
        subprocess.run(cmd, env=env, check=True)
        print(f"Successfully downloaded model '{model}' with {max_workers} max workers.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while downloading the model: {e}")


@register_engine("sglang")
def sglang_completion(
    model, 
    batch_context,
    answer_file,
    temperature,
    max_tokens=32768,
    end_think_token=None,
    **kwargs,
):
    from transformers import AutoTokenizer
    from utils.sglang_server import SGLangServerExecutor
    import tiktoken
    import re

    tokenizer = AutoTokenizer.from_pretrained(model)
    
    uids = [context['uid'] for context in batch_context]
    prompts = [context['prompt'] for context in batch_context]
    processed_context = [
        {
            "uid": uids[i], 
            "turns": [{
                "content": prompts[i],
                "role": "user",
            }]
        } 
        for i in tqdm(range(len(uids)))
    ]
    download_model(model=model)
    
    server_args = {
        "model_path": model,
        "dtype": "auto",
        "tp_size": _infer_cuda_tp_world_size(),
        "mem_fraction_static": 0.7,
        "max_prefill_tokens": max_tokens,
        "max_workers": 256,
        "server_port": 30000,
    }
    
    executor = SGLangServerExecutor(
        **server_args,
    )
    
    print(f"DEBUG: sglang_completion_qwq: model: {model}")
    
    uid_to_response = batch_submit_sglang(
        executor=executor, 
        tokenizer=tokenizer,
        temperature=temperature,
        max_tokens=max_tokens,
        all_context=processed_context,
        end_think_token=end_think_token,
    )
    
    executor.join()
    print("DEBUG: sglang_completion_qwq: done, sleep 10 seconds...")
    time.sleep(10)
        
    num_null = sum(
        [uid_to_response[uid]['answer'] is None for uid in uids if uid in uid_to_response]
    )
    print(f"Number of null responses: {num_null}")
    
    df = pd.DataFrame()
    df['uid'] = [context['uid'] for context in processed_context if context['uid'] in uid_to_response]
    df['ans_id'] = [shortuuid.uuid() for _ in range(len(df))]
    df['model'] = model
    df['messages'] = [
        context['turns'] + [
            {"content": uid_to_response[context['uid']], "role": "assistant"}
        ]
        for context in processed_context if context['uid'] in uid_to_response
    ]
    df['tstamp'] = [time.time() for _ in range(len(df))]
    
    ans["metadata"] = {
        "timestamp": time.time()
    }
    df["metadata"] = metadata 
    
    df.to_json(answer_file, lines=True, orient="records", force_ascii=False)
    
    pass