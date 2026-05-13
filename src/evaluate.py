# =============================================================
#  evaluate.py — Evaluate fine-tuned BART on CNN/Daily Mail
#
#  Usage:
#      python src/evaluate.py
#
#  Loads the fine-tuned model from HuggingFace Hub and
#  evaluates it on the CNN/Daily Mail test set.
#  Prints and saves ROUGE scores.
# =============================================================

import json
import nltk
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import evaluate as evaluate_lib

from config import CONFIG


def load_model_and_tokenizer():
    """Load fine-tuned model from HuggingFace Hub."""
    print(f"Loading model: {CONFIG['finetuned_model']}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["finetuned_model"])
    model = AutoModelForSeq2SeqLM.from_pretrained(CONFIG["finetuned_model"])
    model.eval()
    print("✅ Model loaded.")
    return tokenizer, model


def load_test_data(tokenizer):
    """Load and tokenize CNN/Daily Mail test set."""
    print("Loading test dataset...")
    dataset = load_dataset(CONFIG["dataset_name"], CONFIG["dataset_version"])
    test_data = dataset["test"].shuffle(seed=CONFIG["seed"]).select(
        range(CONFIG["test_samples"])
    )

    def preprocess(batch):
        model_inputs = tokenizer(
            batch["article"],
            max_length=CONFIG["max_input_length"],
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["highlights"],
            max_length=CONFIG["max_target_length"],
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_test = test_data.map(
        preprocess, batched=True,
        remove_columns=test_data.column_names,
        desc="Tokenizing test set"
    )
    print(f"✅ Test data ready — {len(tokenized_test):,} examples")
    return tokenized_test


def build_compute_metrics(tokenizer):
    """Return ROUGE compute_metrics function."""
    rouge = evaluate_lib.load("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.clip(predictions, 0, tokenizer.vocab_size - 1)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        labels = np.clip(labels, 0, tokenizer.vocab_size - 1)

        decoded_preds  = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds  = ["\n".join(nltk.sent_tokenize(p.strip())) for p in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(l.strip())) for l in decoded_labels]

        result = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )
        return {k: round(v, 4) for k, v in result.items()}

    return compute_metrics


def run_evaluation():
    """Main evaluation function."""
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    tokenizer, model = load_model_and_tokenizer()
    tokenized_test = load_test_data(tokenizer)

    # ── Trainer for evaluation only ───────────────────────────
    args = Seq2SeqTrainingArguments(
        output_dir="/tmp/eval_output",
        per_device_eval_batch_size=CONFIG["eval_batch_size"],
        predict_with_generate=True,
        generation_max_length=CONFIG["max_target_length"],
        fp16=CONFIG["fp16"] and torch.cuda.is_available(),
        seed=CONFIG["seed"],
        report_to="none",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding=True
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer),
    )

    # ── Run evaluation ────────────────────────────────────────
    print("\n🔍 Running evaluation on test set...")
    results = trainer.predict(tokenized_test, metric_key_prefix="test")
    metrics = results.metrics

    # ── Print results ─────────────────────────────────────────
    print("\n" + "="*50)
    print("✅ TEST SET RESULTS")
    print("="*50)
    print(f"   ROUGE-1   : {metrics['test_rouge1']:.4f}")
    print(f"   ROUGE-2   : {metrics['test_rouge2']:.4f}")
    print(f"   ROUGE-L   : {metrics['test_rougeL']:.4f}")
    print(f"   ROUGE-Lsum: {metrics['test_rougeLsum']:.4f}")
    print("="*50)

    # ── Save results ──────────────────────────────────────────
    output_path = "test_results.json"
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✅ Results saved to: {output_path}")

    return metrics


if __name__ == "__main__":
    run_evaluation()
