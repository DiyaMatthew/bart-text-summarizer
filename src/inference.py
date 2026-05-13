# =============================================================
#  inference.py — Run summarization on any input text
#
#  Usage:
#      from src.inference import Summarizer
#
#      summarizer = Summarizer()
#      summary = summarizer.summarize("Your text here...")
#      print(summary)
# =============================================================

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from config import CONFIG


class Summarizer:
    """
    A clean inference class for the fine-tuned BART summarizer.

    Example:
        summarizer = Summarizer()
        result = summarizer.summarize("Paste any long text here...")
        print(result["summary"])
        print(result["compression_ratio"])
    """

    def __init__(self, model_id: str = None):
        """
        Load the model from HuggingFace Hub.

        Args:
            model_id: HuggingFace model ID. Defaults to fine-tuned model in config.
        """
        self.model_id = model_id or CONFIG["finetuned_model"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading model: {self.model_id}")
        print(f"Device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()

        print("✅ Summarizer ready.")

    def summarize(
        self,
        text: str,
        max_length: int = None,
        min_length: int = None,
        num_beams: int = None,
        length_penalty: float = None,
    ) -> dict:
        """
        Generate a summary for the input text.

        Args:
            text         : Input text to summarise
            max_length   : Max tokens in summary (default from config)
            min_length   : Min tokens in summary (default from config)
            num_beams    : Beam search width (default from config)
            length_penalty: Length penalty (default from config)

        Returns:
            dict with keys:
                - summary          : generated summary string
                - input_words      : word count of input
                - summary_words    : word count of summary
                - compression_ratio: % reduction in length
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty.")

        # Use config defaults if not specified
        max_length    = max_length    or CONFIG["inference_max_length"]
        min_length    = min_length    or CONFIG["inference_min_length"]
        num_beams     = num_beams     or CONFIG["num_beams"]
        length_penalty = length_penalty or CONFIG["length_penalty"]

        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=CONFIG["max_input_length"],
            truncation=True,
            padding=True,
        ).to(self.device)

        # Generate summary
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                no_repeat_ngram_size=CONFIG["no_repeat_ngram_size"],
                early_stopping=True,
            )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Compute stats
        input_words   = len(text.split())
        summary_words = len(summary.split())
        compression   = round((1 - summary_words / input_words) * 100, 1) if input_words > 0 else 0

        return {
            "summary":           summary,
            "input_words":       input_words,
            "summary_words":     summary_words,
            "compression_ratio": f"{compression}%",
        }


# ── Quick test when run directly ──────────────────────────────
if __name__ == "__main__":
    summarizer = Summarizer()

    test_text = """
    Scientists have discovered a new species of deep-sea fish in the Pacific Ocean
    at a depth of over 8,000 metres. The translucent creature, which has no eyes,
    was found during an expedition by the Schmidt Ocean Institute. Researchers believe
    the fish has adapted to the extreme pressure and darkness of the hadal zone.
    The discovery adds to a growing list of species found in the world's deepest trenches,
    highlighting how little we know about deep-sea biodiversity. The team published
    their findings in the journal Nature on Monday.
    """

    result = summarizer.summarize(test_text)

    print("\n" + "="*60)
    print("INPUT TEXT:")
    print(test_text.strip())
    print("\nGENERATED SUMMARY:")
    print(result["summary"])
    print(f"\nStats: {result['input_words']} words → {result['summary_words']} words "
          f"({result['compression_ratio']} compression)")
    print("="*60)
