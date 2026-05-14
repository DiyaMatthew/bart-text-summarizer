# =============================================================
#  api.py — REST API for BART Text Summarizer
#
#  Usage:
#      uvicorn api:app --reload
#
#  Endpoints:
#      GET  /              → health check
#      GET  /model-info    → model details and ROUGE scores
#      POST /summarize     → generate a summary
#      POST /batch-summarize → summarize multiple texts at once
# =============================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Optional
import torch
import time

# ── App setup ─────────────────────────────────────────────────
app = FastAPI(
    title="BART Text Summarization API",
    description=(
        "REST API for abstractive text summarization using a fine-tuned BART model. "
        "Trained on 50,000 CNN/Daily Mail article-summary pairs. "
        "ROUGE-1: 0.4198 | ROUGE-2: 0.1941 | ROUGE-L: 0.2925"
    ),
    version="1.0.0",
)

# Allow all origins (fine for a portfolio project)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model once at startup ────────────────────────────────
MODEL_ID = "diya2022/bart-text-summarizer"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model: {MODEL_ID} on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).to(DEVICE)
model.eval()
print("✅ Model ready.")


# ── Request / Response schemas ────────────────────────────────

class SummarizeRequest(BaseModel):
    """Request body for /summarize endpoint."""
    text: str = Field(
        ...,
        description="Input text to summarise (min 20 words)",
        example=(
            "Scientists have discovered a new species of deep-sea fish in the "
            "Pacific Ocean at a depth of over 8,000 metres. The translucent "
            "creature, which has no eyes, was found during an expedition by the "
            "Schmidt Ocean Institute. Researchers believe the fish has adapted "
            "to the extreme pressure and darkness of the hadal zone."
        )
    )
    max_length: Optional[int] = Field(130, ge=50, le=300, description="Max summary tokens")
    min_length: Optional[int] = Field(30, ge=10, le=100, description="Min summary tokens")
    num_beams: Optional[int] = Field(4, ge=1, le=8, description="Beam search width")
    length_penalty: Optional[float] = Field(1.0, ge=0.5, le=2.0, description="Length penalty")


class SummarizeResponse(BaseModel):
    """Response body for /summarize endpoint."""
    summary: str
    input_words: int
    summary_words: int
    compression_ratio: str
    inference_time_seconds: float
    model: str


class BatchSummarizeRequest(BaseModel):
    """Request body for /batch-summarize endpoint."""
    texts: List[str] = Field(
        ...,
        description="List of texts to summarise (max 10)",
        min_items=1,
        max_items=10,
    )
    max_length: Optional[int] = Field(130, ge=50, le=300)
    min_length: Optional[int] = Field(30, ge=10, le=100)


class BatchSummarizeResponse(BaseModel):
    """Response body for /batch-summarize endpoint."""
    summaries: List[str]
    count: int
    inference_time_seconds: float
    model: str


# ── Helper ────────────────────────────────────────────────────

def generate_summary(
    text: str,
    max_length: int = 130,
    min_length: int = 30,
    num_beams: int = 4,
    length_penalty: float = 1.0,
) -> str:
    """Core summarization function used by all endpoints."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
        padding=True,
    ).to(DEVICE)

    with torch.no_grad():
        ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

    return tokenizer.decode(ids[0], skip_special_tokens=True)


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def health_check():
    """Health check — confirms the API is running."""
    return {
        "status": "online",
        "model": MODEL_ID,
        "device": DEVICE,
        "message": "BART Text Summarization API is running.",
    }


@app.get("/model-info", tags=["Info"])
def model_info():
    """Returns model details and evaluation results."""
    return {
        "model_id": MODEL_ID,
        "base_model": "facebook/bart-base",
        "task": "Abstractive Text Summarization",
        "dataset": "CNN/Daily Mail 3.0.0",
        "training_samples": 50_000,
        "evaluation_results": {
            "rouge1": 0.4198,
            "rouge2": 0.1941,
            "rougeL": 0.2925,
            "rougeLsum": 0.3911,
        },
        "supported_use_cases": [
            "News Articles",
            "Emails",
            "Customer Reviews",
            "Meeting Reports",
            "Research Papers",
            "Legal Documents",
            "Medical Reports",
            "Conversations",
        ],
        "author": "Diya Mathew",
        "institution": "University of Hertfordshire",
    }


@app.post("/summarize", response_model=SummarizeResponse, tags=["Summarization"])
def summarize(request: SummarizeRequest):
    """
    Generate an abstractive summary for a single input text.

    - **text**: Input text (minimum 20 words)
    - **max_length**: Maximum tokens in summary (default 130)
    - **min_length**: Minimum tokens in summary (default 30)
    - **num_beams**: Beam search width — higher = better quality (default 4)
    - **length_penalty**: Values > 1 encourage longer summaries (default 1.0)
    """
    # Validate input length
    word_count = len(request.text.split())
    if word_count < 20:
        raise HTTPException(
            status_code=422,
            detail=f"Input text too short ({word_count} words). Minimum 20 words required."
        )

    # Generate summary and time it
    start = time.time()
    summary = generate_summary(
        text=request.text,
        max_length=request.max_length,
        min_length=request.min_length,
        num_beams=request.num_beams,
        length_penalty=request.length_penalty,
    )
    elapsed = round(time.time() - start, 3)

    summary_words = len(summary.split())
    compression = round((1 - summary_words / word_count) * 100, 1)

    return SummarizeResponse(
        summary=summary,
        input_words=word_count,
        summary_words=summary_words,
        compression_ratio=f"{compression}%",
        inference_time_seconds=elapsed,
        model=MODEL_ID,
    )


@app.post("/batch-summarize", response_model=BatchSummarizeResponse, tags=["Summarization"])
def batch_summarize(request: BatchSummarizeRequest):
    """
    Generate summaries for multiple texts in one request (max 10).

    - **texts**: List of input texts
    - **max_length**: Maximum tokens per summary (default 130)
    - **min_length**: Minimum tokens per summary (default 30)
    """
    if len(request.texts) > 10:
        raise HTTPException(
            status_code=422,
            detail="Maximum 10 texts per batch request."
        )

    start = time.time()
    summaries = []

    for text in request.texts:
        if len(text.split()) < 20:
            summaries.append("⚠️ Text too short to summarise (min 20 words).")
        else:
            summaries.append(
                generate_summary(
                    text=text,
                    max_length=request.max_length,
                    min_length=request.min_length,
                )
            )

    elapsed = round(time.time() - start, 3)

    return BatchSummarizeResponse(
        summaries=summaries,
        count=len(summaries),
        inference_time_seconds=elapsed,
        model=MODEL_ID,
    )
