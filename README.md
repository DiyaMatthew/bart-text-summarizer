
# 🤖 BART Abstractive Text Summarizer

> Fine-tuned BART transformer for automatic abstractive text summarization, trained on 50,000 CNN/Daily Mail article-summary pairs.

[![HuggingFace Model](https://img.shields.io/badge/🤗%20HuggingFace-diya2022%2Fbart--text--summarizer-blue)](https://huggingface.co/diya2022/bart-text-summarizer)
[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-HuggingFace%20Spaces-green)](https://huggingface.co/spaces/diya2022/bart-text-summarizer)

---

## 📋 Project Overview

This project is a two-part AI Engineering study:

**Part 1 — Model Comparison Research**
Benchmarked five transformer models (T5, BART, PEGASUS, BERT, RoBERTa) on the CNN/Daily Mail dataset using ROUGE evaluation. BART achieved the best performance and was selected for production deployment.

**Part 2 — Production AI Engineering (this repository)**
Built a complete, deployable summarization system using the best-performing model from Part 1, following production engineering practices.

---

## 📊 Model Performance

Evaluated on 5,000 held-out CNN/Daily Mail test samples:

| Metric     | Score  |
|------------|--------|
| ROUGE-1    | 0.4198 |
| ROUGE-2    | 0.1941 |
| ROUGE-L    | 0.2925 |
| ROUGE-Lsum | 0.3911 |

> **Context:** The original `facebook/bart-large-cnn` scores ~0.44 ROUGE-1. This project achieves **0.42 using bart-base** — a much smaller and more efficient model.

---

## 🏗️ Project Structure

```
bart-summarizer/
├── src/
│   ├── config.py       ← centralised configuration
│   ├── train.py        ← training pipeline
│   ├── evaluate.py     ← ROUGE evaluation
│   └── inference.py    ← reusable inference class
├── app.py              ← Streamlit web app
├── requirements.txt    ← dependencies
└── README.md
```

---

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/YOUR_USERNAME/bart-summarizer
cd bart-summarizer
pip install -r requirements.txt
```

### Run Inference
```python
from src.inference import Summarizer

summarizer = Summarizer()

result = summarizer.summarize("""
    Scientists have discovered a new species of deep-sea fish in the Pacific Ocean
    at a depth of over 8,000 metres. The translucent creature, which has no eyes,
    was found during an expedition by the Schmidt Ocean Institute.
""")

print(result["summary"])
# Output: "Scientists discovered a new eyeless deep-sea fish species 
#           at 8,000 metres depth in the Pacific Ocean."
```

### Run the Web App
```bash
python app.py

https://huggingface.co/spaces/diya2022/bart-text-summarizer 
```

### Run Evaluation
```bash
python src/evaluate.py
```

### Run Training
```bash
export HF_TOKEN=your_huggingface_token
python src/train.py
```

---

## 🔧 Technical Details

| Component | Detail |
|---|---|
| Base model | facebook/bart-base |
| Fine-tuning dataset | CNN/Daily Mail 3.0.0 |
| Training samples | 50,000 |
| Validation samples | 5,000 |
| Test samples | 5,000 |
| Training epochs | 3 |
| Learning rate | 5e-5 |
| Batch size | 4 |
| Max input length | 1,024 tokens |
| Max output length | 128 tokens |
| Hardware | NVIDIA T4 GPU (Google Colab) |
| Training time | ~2 hours |
| Mixed precision | fp16 |

---

## 🌍 Supported Use Cases

The model was tested across 8 real-world text types:

- 📰 News articles
- 📧 Emails
- ⭐ Customer reviews
- 📋 Meeting reports
- 📄 Research papers
- ⚖️ Legal documents
- 🏥 Medical reports
- 💬 Conversations

---

## 📦 Model on HuggingFace Hub

The fine-tuned model is publicly available:

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="diya2022/bart-text-summarizer")
summary = summarizer("Your text here...", max_length=130, min_length=30)
print(summary[0]["summary_text"])
```

---

## 👩‍💻 Author

**Diya Mathew**
MSc Data Science & Analytics, University of Hertfordshire

[![HuggingFace](https://img.shields.io/badge/🤗-diya2022-yellow)](https://huggingface.co/diya2022)

```markdown
**GitHub:** [bart-text-summarizer](https://github.com/DiyaMatthew/bart-text-summarizer)
```
