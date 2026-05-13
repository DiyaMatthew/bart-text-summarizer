# =============================================================
#  config.py — Central configuration for BART Text Summarizer
#  All settings live here. Change values here only.
# =============================================================

CONFIG = {
    # ── Model ────────────────────────────────────────────────
    "model_checkpoint":  "facebook/bart-base",
    "finetuned_model":   "diya2022/bart-text-summarizer",

    # ── Dataset ──────────────────────────────────────────────
    "dataset_name":      "cnn_dailymail",
    "dataset_version":   "3.0.0",
    "train_samples":     50_000,
    "val_samples":        5_000,
    "test_samples":       5_000,

    # ── Tokenization ─────────────────────────────────────────
    "max_input_length":  1024,
    "max_target_length":  128,

    # ── Training ─────────────────────────────────────────────
    "learning_rate":      5e-5,
    "num_train_epochs":   3,
    "train_batch_size":   4,
    "eval_batch_size":    4,
    "weight_decay":       0.01,
    "warmup_steps":       500,
    "logging_steps":      100,
    "save_steps":         1000,
    "save_total_limit":   1,
    "seed":               42,
    "fp16":               True,

    # ── HuggingFace Hub ──────────────────────────────────────
    "hf_repo_name":      "diya2022/bart-text-summarizer",

    # ── Output paths ─────────────────────────────────────────
    "output_dir":        "/content/checkpoints",
    "final_model_dir":   "/content/final_model",

    # ── Inference defaults ───────────────────────────────────
    "inference_max_length":  130,
    "inference_min_length":   30,
    "num_beams":               4,
    "length_penalty":        1.0,
    "no_repeat_ngram_size":    3,
}
