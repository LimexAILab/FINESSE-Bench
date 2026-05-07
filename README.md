# FINESSE-Bench

A benchmark suite for evaluating large language models on financial reasoning, financial literacy, document and table understanding, technical analysis, and investment decision-making tasks.

FINESSE-Bench provides a unified evaluation pipeline for running models on multiple finance-oriented benchmarks, scoring their answers with an LLM judge, computing benchmark-level metrics with bootstrap confidence intervals, and aggregating results into group-level rankings.

The framework supports both:
- **API-based models** through OpenAI-compatible endpoints
- **Local inference servers** such as **vLLM** and **sglang**

---

## Overview

FINESSE-Bench is designed to evaluate LLMs across a broad range of finance-related capabilities, including:

- financial literacy
- numerical and multi-step reasoning
- document and table understanding
- derivatives and market microstructure knowledge
- technical analysis and trading logic
- exam-style professional finance tasks
- Russian-language olympiad and finance problems

Unlike single-dataset evaluations, FINESSE-Bench organizes multiple benchmarks under a common execution pipeline and reporting format. This makes it possible to compare models consistently across heterogeneous financial tasks and to produce both **per-benchmark** and **group-level** rankings.

---

## Evaluation Pipeline

FINESSE-Bench uses a **four-stage evaluation workflow**:

1. **Generation**  
   The target model generates answers for all questions in the selected benchmarks.

2. **Judgment**  
   A judge model evaluates generated answers against benchmark references using a configurable prompt and regex-based score extraction.

3. **Metrics**  
   Per-benchmark metrics are computed from judgment files, including **accuracy** and **bootstrap confidence intervals**.

4. **Aggregation**  
   Results are aggregated across predefined benchmark groups using **stratified bootstrap**, producing group-level rankings and statistical tiers.

---

## Supported Benchmarks

The repository currently includes the following benchmarks:

### Public finance QA benchmarks
- **FinQA**
- **ConvFinQA**
- **TAT-QA**

### Exam-style finance benchmarks
- **CFA-like Level 1**
- **CFA-like Level 2**
- **CFA-like Level 3**
- **CMT-like Level 2**
- **CFTE-like Level 1**

### Trading and applied market benchmarks
- **Trading_TA**
- **Trading_derivatives**

### Russian-language benchmark
- **VLigaBench-ru**

These benchmarks cover tasks such as:
- financial statement understanding
- table/text reasoning
- arithmetic and multi-step calculation
- derivatives and options
- technical analysis
- portfolio and investment decision-making
- case-based and exam-style financial reasoning

---

## Repository Structure

```text
FINESSE-Bench/
├── DATASET_LICENSE
├── LICENSE
├── NOTICE
├── README.md
├── aggregated_metrics/
│   └── *.json
├── configs/
│   ├── api_config.yaml.example
│   ├── bench_groups.yaml
│   ├── gen_answer_config.yaml
│   ├── judge_config.yaml
│   └── bench_configs/
│       ├── Trading_TA.yaml
│       ├── Trading_derivatives.yaml
│       ├── VLigaBench-ru.yaml
│       ├── cfa_like_level_1.yaml
│       ├── cfa_like_level_2.yaml
│       ├── cfa_like_level_3.yaml
│       ├── cfte_like_level_1.yaml
│       ├── cmt_like_level_2.yaml
│       ├── convfinqa.yaml
│       ├── finqa.yaml
│       └── tatqa.yaml
├── data/
│   ├── <bench_name>/
│   │   ├── question.jsonl
│   │   └── metrics/
├── scripts/
│   ├── run_generation.py
│   ├── run_judgment.py
│   ├── compute_metrics.py
│   └── aggregate_metrics.py
└── utils/
    ├── completion.py
    ├── judge_utils.py
    └── sglang_server.py
```

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/LimexAILab/FINESSE-Bench.git
cd FINESSE-Bench

python3.11 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

---

## Configuration

FINESSE-Bench uses several config files under `configs/`.

### 1. Model endpoint configuration

Create:

```bash
configs/api_config.yaml
```

based on:

```bash
configs/api_config.yaml.example
```

This file defines API endpoints and generation settings for all available models.

Example:

```yaml
my-model:
  endpoints:
    - api_base: http://localhost:8000/v1
      api_key: ${LOCAL_API_KEY}
  api_type: openai
  model: TheFinAI/Fin-o1-8B
  temperature: 0.6
  max_tokens: 2500
  parallel: 8
```

Supported usage includes:
- hosted OpenAI-compatible APIs
- local **vLLM**
- local **sglang** backends

### 2. Benchmark/model selection

Edit:

```bash
configs/gen_answer_config.yaml
```

This config specifies:
- `bench_list`: which benchmarks to run
- `model_list`: which models to evaluate

Example:

```yaml
bench_list:
  - finqa
  - convfinqa
  - tatqa

model_list:
  - my-model
  - another-model
```

### 3. Judge configuration

Edit:

```bash
configs/judge_config.yaml
```

This file specifies:
- `judge_model`
- judge prompt template
- temperature
- max_tokens
- optional regex patterns for score extraction

The judge model itself must also be defined in `configs/api_config.yaml`.

### 4. Benchmark grouping for ranking

Edit:

```bash
configs/bench_groups.yaml
```

This file defines benchmark groups for aggregated rankings, for example:
- exam-like benchmarks
- public finance benchmarks
- trading / TA benchmarks

If this file is missing, the aggregation script falls back to built-in default groups.

---

## Quick Start

### 1. Generate model answers

Run generation for all benchmarks and models listed in `configs/gen_answer_config.yaml`:

```bash
python -m scripts.run_generation
```

Or explicitly specify config files:

```bash
python -m scripts.run_generation \
  --config configs/gen_answer_config.yaml \
  --api-config configs/api_config.yaml
```

This script:
- loads benchmark question files from `configs/bench_configs/*.yaml`
- loads model endpoints from `configs/api_config.yaml`
- generates answers for every `(benchmark, model)` pair
- appends outputs to answer files
- skips already processed questions using `uid`

### 2. Run LLM judgment

Evaluate generated answers with the configured judge model:

```bash
python -m scripts.run_judgment
```

Or explicitly:

```bash
python -m scripts.run_judgment \
  --config configs/gen_answer_config.yaml \
  --judge-config configs/judge_config.yaml \
  --api-config configs/api_config.yaml
```

This script:
- loads questions and generated answers
- builds judge prompts using the configured template
- calls the judge model
- extracts binary scores (`0` or `1`)
- saves judgments to:

```text
data/<bench_name>/model_judgment/<judge_model>/<model_name>.jsonl
```

### 3. Compute benchmark-level metrics

Compute per-benchmark accuracy and bootstrap confidence intervals:

```bash
python -m scripts.compute_metrics
```

Or with parameters:

```bash
python -m scripts.compute_metrics \
  --config configs/gen_answer_config.yaml \
  --judge-config configs/judge_config.yaml \
  --num-bootstrap 100000 \
  --ci-level 0.95
```

You can also restrict computation to a single benchmark or model:

```bash
python -m scripts.compute_metrics --bench finqa
python -m scripts.compute_metrics --model my-model
```

Outputs are saved to:

```text
data/<bench_name>/metrics/<judge_model>/<model_name>.json
```

### 4. Aggregate rankings across benchmark groups

Aggregate benchmark-level results into group rankings:

```bash
python -m scripts.aggregate_metrics
```

Or explicitly:

```bash
python -m scripts.aggregate_metrics \
  --config configs/gen_answer_config.yaml \
  --judge-config configs/judge_config.yaml \
  --groups-config configs/bench_groups.yaml \
  --output-dir aggregated_metrics \
  --num-bootstrap 100000
```

This script:
- loads binary scores from judgment files
- aggregates results across benchmark groups using **stratified bootstrap**
- ranks models by mean performance
- computes confidence intervals
- identifies overlapping confidence intervals
- assigns **statistical tiers**

Outputs are saved to:

```text
aggregated_metrics/<group_name>_<judge_model>.json
```

---

## Running Local Models

### vLLM example

Launch a local OpenAI-compatible server:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model SUFE-AIFLM-Lab/Fin-R1 \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 16384
```

Then configure `configs/api_config.yaml` accordingly:

```yaml
fin-r1-local:
  endpoints:
    - api_base: http://localhost:8000/v1
      api_key: ${LOCAL_API_KEY}
  api_type: openai
  model: SUFE-AIFLM-Lab/Fin-R1
  temperature: 0.6
  max_tokens: 2500
  parallel: 8
```

### sglang example

For large local batch inference, see:

```text
utils/sglang_server.py
utils/completion.py
```

An sglang-backed model can be configured similarly in `configs/api_config.yaml` using the corresponding `api_type`.

---

## Metrics

### Per-benchmark metric

FINESSE-Bench currently computes:
- **Accuracy**: fraction of judged answers with score `1`

### Confidence intervals

For each `(benchmark, model)` pair, the framework computes bootstrap confidence intervals over binary judgment scores.

Default behavior:
- percentile bootstrap
- configurable number of iterations
- configurable confidence level

---

## Aggregated Ranking Method

Group-level rankings are computed with **stratified bootstrap aggregation**:

1. Questions are resampled **within each benchmark**
2. Benchmark-level accuracies are recomputed
3. A weighted average is taken across benchmarks
4. Confidence intervals are estimated from the bootstrap distribution

This preserves benchmark structure and avoids flattening all questions across datasets into a single pool.

### Statistical tiers

After aggregation, models are assigned to tiers based on **overlapping confidence intervals**:
- models in the same tier are treated as statistically indistinguishable under this heuristic
- the ranking output also includes which models have overlapping confidence intervals

---

## Result Structure

### Questions

Each benchmark stores its questions in:

```text
data/<bench_name>/question.jsonl
```

### Judgments

Judge outputs are stored in:

```text
data/<bench_name>/model_judgment/<judge_model>/<model_name>.jsonl
```

Each judgment entry includes:
- `uid`
- category / subcategory
- judge name
- evaluated model
- one or more games with:
  - extracted `score`
  - raw judge output
  - prompt

### Metrics

Per-benchmark metric files are stored in:

```text
data/<bench_name>/metrics/<judge_model>/<model_name>.json
```

### Aggregated results

Group-level ranking files are stored in:

```text
aggregated_metrics/<group_name>_<judge_model>.json
```

---

## Benchmarks Overview

### FinQA
A benchmark for financial reasoning over text and reports, requiring arithmetic, logic, and interpretation.

**Skills:**
- financial literacy
- textual reasoning
- arithmetic reasoning
- multi-step problem solving

### ConvFinQA
A conversational extension of FinQA focused on multi-turn financial numerical reasoning.

**Skills:**
- dialogue context tracking
- financial reasoning
- table/document understanding
- multi-step calculations

### TAT-QA
A benchmark requiring joint reasoning over tables and text in business and financial settings.

**Skills:**
- table-text integration
- numerical reasoning
- information extraction
- explanation generation

### CFA-like Levels 1–3
A family of exam-style benchmarks inspired by professional finance certification tasks.

**Skills:**
- theory and fundamentals
- valuation and analysis
- portfolio reasoning
- decision-making
- case-based interpretation

### CMT-like Level 2
A benchmark focused on technical analysis and market interpretation.

**Skills:**
- indicator-based reasoning
- chart/TA interpretation
- risk management
- trading decision logic

### CFTE-like Level 1
A finance and trading benchmark covering practical market knowledge and applied reasoning.

### VLigaBench-ru
A Russian-language benchmark of olympiad-style tasks in economics, finance, and investment.

**Skills:**
- economics
- finance and valuation
- investment analysis
- mathematical reasoning
- case-based argumentation

### Trading_TA
A benchmark on professional technical analysis and trading patterns.

**Skills:**
- pattern recognition
- seasonality
- momentum
- rule-based trading logic

### Trading_derivatives
A benchmark of applied problems on derivatives, options, and market microstructure.

**Skills:**
- options pricing
- synthetic positions
- arbitrage
- duration and returns
- derivatives reasoning

---

## Reproducibility Notes

- Generation and judgment are driven by YAML configuration files under `configs/`
- Metrics use bootstrap resampling with configurable seeds
- Aggregation also uses seeded stratified bootstrap
- Existing answers and judgments are reused automatically when possible

---

## License

- **Code**: Apache License 2.0 (see `LICENSE`)
- Includes portions from `lmarena/arena-hard-auto` under Apache-2.0; see `NOTICE`
- **Datasets** (questions, prompts): CC BY-NC 4.0 (see `DATASET_LICENSE`)
- For third-party components and their licenses, see `NOTICE`

---

## Citation

If you use FINESSE-Bench in your research, please cite this repository and any original benchmark sources where applicable.
