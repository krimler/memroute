
# Store Routing for Memory-Augmented Agents

This repository contains the experimental framework used in the paper:

**“Did You Check the Right Pocket? Store Routing for Memory-Augmented Agents”**

The project evaluates routing policies that select which memory stores to retrieve **before** retrieval and generation. The framework supports both:

- **Synthetic routing evaluation** (store-selection correctness)
- **End-to-end LLM QA evaluation** (accuracy vs token cost)

---

## Repository Structure

```
.
├── run_experiments.py               # Main entry point
├── benchmark_comprehensive.py       # Synthetic memory generator
├── ablation_experiments.py          # Feature ablation experiments
├── update_cost_experiments.py       # Store-cost experiments
├── metrics_framework.py             # Routing metrics (coverage, EM, waste)
├── er21.py                          # Real LLM QA evaluation
├── prompt1 / prompt2                # Prompt templates
├── requirements_e2e.txt             # Dependencies
```

---

## Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements_e2e.txt
```

---

## Running Experiments

### 1. Synthetic Routing Evaluation

Evaluates routing coverage, exact match, and waste using synthetic store labels.

```bash
python ablation_experiments.py
python benchmark_comprehensive.py
```

Outputs:
- Feature ablation tables
- Coverage / EM / waste metrics

---

### 2. End-to-End LLM QA Evaluation

Runs routing policies on real LLM question answering tasks.

```bash
python er21.py
```

Outputs:
- Accuracy per policy
- Token usage statistics
- Short-context vs long-context results

---

### 3. Full Experiment Pipeline

To run all sythetic experiments sequentially:

```bash
python run_experiments.py --all
```

---

## Routing Policies Evaluated

- Uniform retrieval
- Oracle routing
- Fixed subset policies (e.g., STM+Sum+LTM)
- Hybrid heuristic routing
- Ablated feature routing variants

---

## Metrics

The framework evaluates routing using:

- **Coverage**: required stores retrieved
- **Exact Match (EM)**: exact store subset selected
- **Waste**: unnecessary stores retrieved
- **Token Cost**: context tokens inserted into prompts
- **QA Accuracy**: substring answer match

---

## Reproducibility Notes

- Synthetic routing labels are derived from query taxonomies
- Store contents are generated deterministically from seed values
- Temperature is set to 0 for LLM evaluation
- Bootstrap resampling is used for statistical significance

---

## Citation

```
@inproceedings{store-routing-2026,
  title={Did You Check the Right Pocket? Store Routing for Memory-Augmented Agents},
  author={Anonymous},
  year={2026}
}
```

---

## Contact

For questions or issues, please open a repository issue or contact the authors.

