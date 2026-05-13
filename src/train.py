# =============================================================
#  train.py — Fine-tuning BART on CNN/Daily Mail
#
#  Usage (in Colab or terminal):
#      python src/train.py
#
#  What this script does:
#  1. Loads CNN/Daily Mail dataset
#  2. Tokenizes articles and summaries
#  3. Fine-tunes facebook/bart-base
#  4. Pushes the final model to HuggingFace Hub
# =============================================================

import os
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
import evaluate
from huggingface_hub import login

# ── Import config ─────────────────────────────────────────────
from config import CONFIG


def setup():
    """Download NLTK data and log in to HuggingFace."""
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "HuggingFace token not found. "
            "Set it with: export HF_TOKEN=your_token_here"
        )
    login(token=hf_token)
    print("✅ Setup complete.")


def load_data():
    """Load and sample CNN/Daily Mail dataset."""
    print("Loading CNN/Daily Mail dataset...")
    dataset = load_dataset(CONFIG["dataset_name"], CONFIG["dataset_version"])

    train = dataset["train"].shuffle(seed=CONFIG["seed"]).select(range(CONFIG["train_samples"]))
    val   = dataset["validation"].shuffle(seed=CONFIG["seed"]).select(range(CONFIG["val_samples"]))

    print(f"✅ Dataset loaded — Train: {len(train):,} | Val: {len(val):,}")
    return train, val


def tokenize_data(train, val, tokenizer):
    """Tokenize articles and highlights."""
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

    print("Tokenizing datasets...")
    tokenized_train = train.map(
        preprocess, batched=True,
        remove_columns=train.column_names,
        desc="Tokenizing train"
    )
    tokenized_val = val.map(
        preprocess, batched=True,
        remove_columns=val.column_names,
        desc="Tokenizing validation"
    )
    print("✅ Tokenization complete.")
    return tokenized_train, tokenized_val


def build_compute_metrics(tokenizer):
    """Return a compute_metrics function for the Trainer."""
    rouge = evaluate.load("rouge")

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


def train():
    """Main training function."""
    setup()

    # ── Load tokenizer and model ──────────────────────────────
    print(f"Loading model: {CONFIG['model_checkpoint']}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_checkpoint"])
    model = AutoModelForSeq2SeqLM.from_pretrained(CONFIG["model_checkpoint"])

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model loaded — {total_params:,} parameters")

    # ── Load and tokenize data ────────────────────────────────
    train_data, val_data = load_data()
    tokenized_train, tokenized_val = tokenize_data(train_data, val_data, tokenizer)

    # ── Training arguments ────────────────────────────────────
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=CONFIG["output_dir"],
        num_train_epochs=CONFIG["num_train_epochs"],
        learning_rate=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
        warmup_steps=CONFIG["warmup_steps"],
        per_device_train_batch_size=CONFIG["train_batch_size"],
        per_device_eval_batch_size=CONFIG["eval_batch_size"],
        save_strategy="steps",
        save_steps=CONFIG["save_steps"],
        save_total_limit=CONFIG["save_total_limit"],
        eval_strategy="no",
        fp16=CONFIG["fp16"] and torch.cuda.is_available(),
        predict_with_generate=True,
        generation_max_length=CONFIG["max_target_length"],
        push_to_hub=True,
        hub_model_id=CONFIG["hf_repo_name"],
        hub_strategy="checkpoint",
        logging_steps=CONFIG["logging_steps"],
        report_to="none",
        seed=CONFIG["seed"],
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding=True, pad_to_multiple_of=8
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer),
    )

    # ── Train ─────────────────────────────────────────────────
    print("🚀 Starting training...")
    result = trainer.train()

    print("\n✅ Training complete!")
    print(f"   Time  : {result.metrics['train_runtime']/3600:.2f} hours")
    print(f"   Loss  : {result.metrics['train_loss']:.4f}")
    print(f"   Speed : {result.metrics['train_samples_per_second']:.1f} samples/sec")

    # ── Push final model ──────────────────────────────────────
    print("\n📤 Pushing final model to HuggingFace Hub...")
    trainer.push_to_hub()
    print(f"✅ Model live at: https://huggingface.co/{CONFIG['hf_repo_name']}")


if __name__ == "__main__":
    train()
