# Human-in-the-Loop Learning with Transformers

A HITL (Human-in-the-Loop) learning system that uses transformer-based models to detect and reduce hallucinations in AI-generated responses, evaluated on the TruthfulQA dataset.

---

## Team Members

| Name | Student ID |
|------|-----------|
| Shresta Munikuntla | SXM230048 |
| Sameeraa Kandalgaonkar | SXK220350 |
| Hansita Marwah | HXM230046 |
| Drashti Shah | DSS230002 |

---

## Project Overview

Large language models (LLMs) are powerful but prone to hallucinations — generating confident yet factually incorrect responses. This project addresses that problem by building a HITL pipeline where:

1. A transformer model generates answers to questions
2. Low-confidence or incorrect answers are flagged for human review
3. Human feedback is used to retrain the model
4. The cycle repeats, improving truthfulness over time

A **Streamlit web app** is included to allow human reviewers to interact with the system in real time.

---

## Dataset

**TruthfulQA** — [HuggingFace Link](https://huggingface.co/datasets/domenicrosati/TruthfulQA)

- 817 instances across 38 categories
- 7 features: Question, Best Answer, Correct Answers, Incorrect Answers, Category, Type, Source
- Designed to expose hallucinations in language models

---

## Project Structure

```
├── stage1_data_loading.py        # Load and preprocess TruthfulQA dataset
├── stage2_model_setup.py         # Load pre-trained transformer model
├── stage3_answer_generation.py   # Generate answers using the model
├── stage4_classification.py      # Classify answers as truthful or not
├── stage5_pipeline.py            # Connect all stages into one pipeline
├── stage6_hitl_feedback.py       # Human feedback and model retraining loop
├── stage7_evaluation.py          # Accuracy, F1 score, before/after comparison
├── app.py                        # Streamlit web interface
├── experiment_log.csv            # Logs of feedback and retraining results
├── requirements.txt              # Dependencies
└── README.md
```

---

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd human-in-the-loop

# Install dependencies
pip install -r requirements.txt
```

---

## Requirements

```
torch
transformers
pandas
scikit-learn
streamlit
datasets
```

---

## How to Run

### Run the Full Pipeline
```bash
python stage5_pipeline.py
```

### Launch the Streamlit App
```bash
streamlit run app.py
```

### Run Evaluation
```bash
python stage7_evaluation.py
```

---

## How It Works

```
Stage 1 → Load TruthfulQA dataset
Stage 2 → Load pre-trained transformer model
Stage 3 → Generate answers to questions
Stage 4 → Classify answers as truthful or hallucinated
Stage 5 → Run full pipeline end-to-end
Stage 6 → Human reviews flagged answers → model retrains
Stage 7 → Measure accuracy before vs after feedback loop
```

---

## Results

Results are logged in `experiment_log.csv` and include:
- Accuracy before and after the feedback loop
- F1 score comparison
- Number of human corrections made per iteration

---

## Technologies Used

- **Python** — Core language
- **PyTorch** — Model training and fine-tuning
- **Transformers (HuggingFace)** — Pre-trained LLM backbone
- **Streamlit** — Interactive human feedback interface
- **Pandas / Scikit-learn** — Data preprocessing and evaluation

---

## References

- Vaswani et al., "Attention Is All You Need," 2017
- Lin et al., "TruthfulQA: Measuring How Models Mimic Human Falsehoods," 2021
- Ouyang et al., "Training language models to follow instructions with human feedback (InstructGPT)," 2022
- TruthfulQA Dataset: https://huggingface.co/datasets/domenicrosati/TruthfulQA
