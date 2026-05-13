import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# ── Load model ────────────────────────────────────────────────
MODEL_ID = "diya2022/bart-text-summarizer"

print(f"Loading model: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
model.eval()
print("✅ Model loaded.")


def summarize(text, max_length, min_length, num_beams, length_penalty):
    """Generate a summary for the input text."""
    if not text or not text.strip():
        return "⚠️ Please enter some text to summarise."

    word_count = len(text.split())
    if word_count < 20:
        return "⚠️ Text is too short. Please provide at least 20 words."

    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
        padding=True,
    )

    with torch.no_grad():
        ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=int(max_length),
            min_length=int(min_length),
            num_beams=int(num_beams),
            length_penalty=float(length_penalty),
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

    summary = tokenizer.decode(ids[0], skip_special_tokens=True)
    summary_words = len(summary.split())
    compression = round((1 - summary_words / word_count) * 100, 1)

    output = f"{summary}\n\n"
    output += f"📊 Stats: {word_count} words → {summary_words} words ({compression}% compression)"
    return output


# ── Examples ──────────────────────────────────────────────────
examples = [
    [
        "Scientists have discovered a new species of deep-sea fish in the Pacific Ocean "
        "at a depth of over 8,000 metres. The translucent creature, which has no eyes, "
        "was found during an expedition by the Schmidt Ocean Institute. Researchers believe "
        "the fish has adapted to the extreme pressure and darkness of the hadal zone. "
        "The discovery adds to a growing list of species found in the world's deepest "
        "trenches, highlighting how little we know about deep-sea biodiversity. The team "
        "published their findings in the journal Nature on Monday.",
        130, 30, 4, 1.0
    ],
    [
        "Dear Team, I am writing to inform you that our quarterly review meeting has been "
        "rescheduled from Thursday 9th to Monday 13th at 10am in the main conference room. "
        "Please update your calendars accordingly. The agenda will include Q3 performance "
        "review, budget planning for Q4, and discussion of the new product roadmap. "
        "Please come prepared with your department updates. If you cannot attend, "
        "please send your report to me by Friday. Best regards, James",
        80, 20, 4, 1.0
    ],
    [
        "The new restaurant downtown has quickly become a favourite among locals. "
        "The menu features a wide range of dishes inspired by Mediterranean cuisine, "
        "with fresh ingredients sourced from local farmers. The ambiance is warm and "
        "inviting, with soft lighting and comfortable seating. The staff are attentive "
        "and knowledgeable about the menu. Prices are reasonable for the quality offered. "
        "The dessert menu is particularly impressive, with homemade gelato and tiramisu "
        "that have already gained a loyal following. Reservations are recommended as the "
        "restaurant fills up quickly, especially on weekends.",
        80, 20, 4, 1.0
    ],
]

# ── Build Gradio UI ───────────────────────────────────────────
with gr.Blocks(title="BART Text Summarizer") as demo:

    gr.Markdown("""
    # 🤖 Automatic Text Summarization using BART Transformer
    Fine-tuned **BART** model trained on **50,000 CNN/Daily Mail** article-summary pairs.

    | Metric | Score |
    |--------|-------|
    | ROUGE-1 | 0.4198 |
    | ROUGE-2 | 0.1941 |
    | ROUGE-L | 0.2925 |

    *Diya Mathew — MSc Data Science & Analytics, University of Hertfordshire*
    """)

    with gr.Row():
        with gr.Column(scale=3):
            input_text = gr.Textbox(
                label="📄 Input Text",
                placeholder="Paste your article, email, meeting report, review, or any text here...",
                lines=12,
            )

        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Settings")
            max_length = gr.Slider(50, 300, value=130, step=10,
                                   label="Max Summary Length")
            min_length = gr.Slider(10, 100, value=30, step=5,
                                   label="Min Summary Length")
            num_beams  = gr.Slider(1, 8, value=4, step=1,
                                   label="Beam Search Width",
                                   info="Higher = better quality, slower")
            length_penalty = gr.Slider(0.5, 2.0, value=1.0, step=0.1,
                                       label="Length Penalty",
                                       info=">1 encourages longer summaries")

    summarize_btn = gr.Button("🚀 Generate Summary", variant="primary", size="lg")

    output_text = gr.Textbox(
        label="📋 Generated Summary",
        lines=6,
    )

    summarize_btn.click(
        fn=summarize,
        inputs=[input_text, max_length, min_length, num_beams, length_penalty],
        outputs=output_text,
    )

    gr.Markdown("### 💡 Try an Example")
    gr.Examples(
        examples=examples,
        inputs=[input_text, max_length, min_length, num_beams, length_penalty],
        outputs=output_text,
        fn=summarize,
        cache_examples=False,
        label="Click any example to load it",
    )

    gr.Markdown("""
    ---
    ### 🌍 Supported Use Cases
    ✅ News Articles &nbsp; ✅ Emails &nbsp; ✅ Customer Reviews &nbsp; ✅ Meeting Reports
    &nbsp; ✅ Research Papers &nbsp; ✅ Legal Documents &nbsp; ✅ Medical Reports

    **Model:** [diya2022/bart-text-summarizer](https://huggingface.co/diya2022/bart-text-summarizer)
    &nbsp;|&nbsp;
    **GitHub:** [bart-text-summarizer](https://github.com/diya2022/bart-text-summarizer)
    """)


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
