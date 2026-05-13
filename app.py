import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="BART Text Summarizer",
    page_icon="📝",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.title("📝 BART Text Summarizer")
st.sidebar.markdown("---")
st.sidebar.write("### About")
st.sidebar.write(
    "This app uses a fine-tuned **BART** (Bidirectional and Auto-Regressive "
    "Transformer) model to generate abstractive summaries from input text."
)
st.sidebar.markdown("---")
st.sidebar.write("### Model Info")
st.sidebar.write("- **Base:** facebook/bart-base")
st.sidebar.write("- **Fine-tuned on:** CNN/Daily Mail")
st.sidebar.write("- **Training samples:** 50,000")
st.sidebar.write("- **ROUGE-1:** 0.4198")
st.sidebar.write("- **ROUGE-2:** 0.1941")
st.sidebar.write("- **ROUGE-L:** 0.2925")
st.sidebar.markdown("---")
st.sidebar.write("### Supported Use Cases")
st.sidebar.write(
    "✅ News Articles\n\n✅ Emails\n\n✅ Customer Reviews\n\n"
    "✅ Meeting Reports\n\n✅ Research Papers\n\n✅ Legal Documents\n\n"
    "✅ Medical Reports\n\n✅ Stories & Conversations"
)
st.sidebar.markdown("---")
st.sidebar.write("*Diya Mathew | MSc Data Science & Analytics*")
st.sidebar.write("*University of Hertfordshire*")

# ── Model loading ─────────────────────────────────────────────
MODEL_ID = "diya2022/bart-text-summarizer"

@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    model.eval()
    return tokenizer, model

# ── Main UI ───────────────────────────────────────────────────
st.title("🤖 Automatic Text Summarization using BART Transformer")
st.markdown(
    "Generate concise, human-like abstractive summaries using a fine-tuned "
    "**BART** model trained on 50,000 CNN/Daily Mail article-summary pairs."
)
st.markdown("---")

with st.spinner("⏳ Loading model — first run takes ~30 seconds..."):
    tokenizer, model = load_model()
st.success("✅ Model loaded and ready!")

st.markdown("---")

# ── Input and settings ────────────────────────────────────────
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("📄 Input Text")
    input_text = st.text_area(
        label="Paste your text here (article, email, report, review, etc.)",
        height=300,
        placeholder="Paste your article, email, meeting report, or any text here...",
    )
    word_count = len(input_text.split()) if input_text.strip() else 0
    st.caption(f"Word count: {word_count}")

with col2:
    st.subheader("⚙️ Settings")
    max_length = st.slider("Max Summary Length", 50, 300, 130, 10)
    min_length = st.slider("Min Summary Length", 10, 100, 30, 5)
    num_beams  = st.slider("Beam Search Width", 1, 8, 4, 1,
                           help="Higher = better quality but slower")
    length_penalty = st.slider("Length Penalty", 0.5, 2.0, 1.0, 0.1,
                                help="Values > 1 encourage longer summaries")

# ── Summarize ─────────────────────────────────────────────────
st.markdown("---")

def generate_summary(text, max_len, min_len, beams, len_pen):
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
            max_length=max_len,
            min_length=min_len,
            num_beams=beams,
            length_penalty=len_pen,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
    return tokenizer.decode(ids[0], skip_special_tokens=True)


if st.button("🚀 Generate Summary", use_container_width=True):
    if not input_text.strip():
        st.warning("⚠️ Please enter some text first.")
    elif word_count < 20:
        st.warning("⚠️ Text is too short. Please provide more context.")
    else:
        with st.spinner("⏳ Generating summary..."):
            summary = generate_summary(
                input_text, max_length, min_length, num_beams, length_penalty
            )

        st.markdown("---")
        st.subheader("📋 Generated Summary")
        st.success(summary)

        summary_words  = len(summary.split())
        compression    = round((1 - summary_words / word_count) * 100, 1)
        st.caption(
            f"Summary: {summary_words} words | "
            f"Original: {word_count} words | "
            f"Compression: {compression}%"
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button(
                "⬇️ Download Summary",
                data=summary,
                file_name="summary.txt",
                mime="text/plain",
            )

# ── Examples ──────────────────────────────────────────────────
st.markdown("---")
with st.expander("💡 Example — News Article"):
    st.text_area("Copy and paste into the input box above:", height=120, value=(
        "Scientists have discovered a new species of deep-sea fish in the Pacific Ocean "
        "at a depth of over 8,000 metres. The translucent creature, which has no eyes, "
        "was found during an expedition by the Schmidt Ocean Institute. Researchers believe "
        "the fish has adapted to the extreme pressure and darkness of the hadal zone. "
        "The discovery adds to a growing list of species found in the world's deepest trenches, "
        "highlighting how little we know about deep-sea biodiversity."
    ))

with st.expander("💡 Example — Email"):
    st.text_area("Copy and paste into the input box above:", height=120, value=(
        "Dear Team, I am writing to follow up on our Q3 project proposal. Our team has "
        "completed the initial research phase and we are ready to move into development. "
        "We need your approval on the budget allocation of £45,000 by Friday in order to "
        "proceed on schedule. We also need to schedule a kick-off meeting for the week of "
        "the 15th. We have a hard deadline of the 30th for the first deliverable. "
        "Please let us know if you have any questions. Best regards, Sarah"
    ))
