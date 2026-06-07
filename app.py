"""
AI-Powered Question & Answering App
Compatible with: streamlit>=1.32, transformers>=4.40, torch>=2.0
"""

import streamlit as st
from transformers import pipeline
import time

# ── Page Configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Q&A",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:ital,wght@0,400;0,500;1,400&display=swap');

/* Global font override */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif !important;
}

/* App background */
.stApp {
    background: #0d0f14;
    color: #e8e4dc;
}

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* Hero title */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.1;
    background: linear-gradient(135deg, #f0e8d5 0%, #c9a96e 50%, #e8c98a 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}

.hero-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #6b7280;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

/* Answer card */
.answer-card {
    background: linear-gradient(135deg, #1a1f2e 0%, #131720 100%);
    border: 1px solid #c9a96e44;
    border-left: 4px solid #c9a96e;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-top: 1rem;
    font-family: 'Syne', sans-serif;
    font-size: 1.25rem;
    font-weight: 600;
    color: #f0e8d5;
    box-shadow: 0 0 40px #c9a96e18;
}

.answer-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    color: #c9a96e;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

/* Confidence bar container */
.conf-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-top: 1.2rem;
}

.conf-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #6b7280;
    letter-spacing: 0.1em;
    white-space: nowrap;
}

.conf-value {
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    font-weight: 500;
    color: #c9a96e;
}

/* Highlighted answer in context */
.highlight-context {
    background: #1a1f2e;
    border: 1px solid #2a3040;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    line-height: 1.8;
    color: #9ca3af;
    white-space: pre-wrap;
    word-break: break-word;
}

.highlight-span {
    background: #c9a96e33;
    color: #f0d090;
    border-radius: 3px;
    padding: 1px 4px;
    font-weight: 500;
    border-bottom: 2px solid #c9a96e88;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: #0a0c10 !important;
    border-right: 1px solid #1e2330;
}

section[data-testid="stSidebar"] * {
    color: #9ca3af !important;
}

.sidebar-tag {
    display: inline-block;
    background: #1a1f2e;
    border: 1px solid #2a3040;
    border-radius: 6px;
    padding: 3px 10px;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #c9a96e !important;
    margin-bottom: 0.5rem;
}

/* Text areas */
.stTextArea textarea {
    background: #131720 !important;
    border: 1px solid #2a3040 !important;
    border-radius: 10px !important;
    color: #e8e4dc !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
    line-height: 1.7 !important;
    caret-color: #c9a96e;
}

.stTextArea textarea:focus {
    border-color: #c9a96e !important;
    box-shadow: 0 0 0 1px #c9a96e44 !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #c9a96e, #a8823e) !important;
    color: #0d0f14 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.05em !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.5rem !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px #c9a96e44 !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background: #131720 !important;
    border: 1px solid #2a3040 !important;
    color: #e8e4dc !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
}

/* Status / spinner */
.stSpinner > div {
    border-top-color: #c9a96e !important;
}

/* Divider */
hr {
    border-color: #1e2330 !important;
}

/* Metric */
[data-testid="stMetric"] {
    background: #131720;
    border: 1px solid #2a3040;
    border-radius: 10px;
    padding: 0.8rem 1rem;
}

[data-testid="stMetricLabel"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    color: #6b7280 !important;
    text-transform: uppercase;
}

[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.6rem !important;
    font-weight: 800 !important;
    color: #c9a96e !important;
}
</style>
""", unsafe_allow_html=True)


# ── Model Registry ─────────────────────────────────────────────────────────────
MODELS = {
    "timpal0l/mdeberta-v3-base-squad2 (Multilingual)": "timpal0l/mdeberta-v3-base-squad2",
    "deepset/roberta-base-squad2 (Fast & General)":    "deepset/roberta-base-squad2",
    "distilbert-base-uncased-distilled-squad (Lite)":  "distilbert-base-uncased-distilled-squad",
    "deepset/deberta-v3-base-squad2 (High Accuracy)":  "deepset/deberta-v3-base-squad2",
}


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 Model")
    selected_label = st.selectbox(
        label="Choose model",
        options=list(MODELS.keys()),
        label_visibility="collapsed",
    )
    model_id = MODELS[selected_label]
    st.markdown(f'<div class="sidebar-tag">{model_id}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📖 How to use")
    st.markdown("""
1. Paste a **context** passage on the right  
2. Write your **question** about it  
3. Hit **Ask the AI**  
4. See the highlighted answer & confidence score
""")

    st.markdown("---")
    st.markdown("### ⚙️ Advanced")
    top_k = st.slider("Top-K answers", min_value=1, max_value=5, value=1)
    show_highlight = st.toggle("Show answer in context", value=True)


# ── Model Loader (cached per model_id) ────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    return pipeline(
        task="question-answering",
        model=model_name,
        tokenizer=model_name,
    )


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">Ask the AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Extractive Q&A · Hugging Face Transformers</div>', unsafe_allow_html=True)

# Model loading state
model_placeholder = st.empty()
with model_placeholder.container():
    with st.spinner(f"Loading `{model_id}` — first load may take a moment…"):
        qa_model = load_model(model_id)
model_placeholder.empty()

# ── Main Layout ────────────────────────────────────────────────────────────────
col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.markdown("#### 📄 Context")
    context = st.text_area(
        label="context_area",
        label_visibility="collapsed",
        height=220,
        placeholder="Paste any paragraph, article, or document text here…",
        key="context",
    )

    st.markdown("#### ❓ Question")
    question = st.text_area(
        label="question_area",
        label_visibility="collapsed",
        height=100,
        placeholder="Ask something based on the context above…",
        key="question",
    )

    ask_btn = st.button("Ask the AI →", type="primary", use_container_width=True)

# ── Output Column ──────────────────────────────────────────────────────────────
with col_output:
    st.markdown("#### 💡 Answer")

    if ask_btn:
        # Validation
        if not context.strip():
            st.warning("⚠️ Please add some context text first.")
        elif not question.strip():
            st.warning("⚠️ Please enter a question.")
        else:
            with st.spinner("Thinking…"):
                start_time = time.perf_counter()
                results = qa_model(
                    question=question,
                    context=context,
                    top_k=top_k,
                )
                elapsed = time.perf_counter() - start_time

            # Normalise: pipeline returns dict when top_k=1, list otherwise
            if isinstance(results, dict):
                results = [results]

            # ── Primary answer card ────────────────────────────────────────
            best = results[0]
            answer_text = best["answer"]
            confidence  = best["score"]

            st.markdown(
                f'<div class="answer-card">'
                f'  <div class="answer-label">Answer</div>'
                f'  {answer_text}'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Metrics row
            m1, m2, m3 = st.columns(3)
            m1.metric("Confidence", f"{confidence:.1%}")
            m2.metric("Inference time", f"{elapsed:.2f}s")
            m3.metric("Answers returned", len(results))

            # ── Additional top-k answers ───────────────────────────────────
            if top_k > 1 and len(results) > 1:
                with st.expander(f"All {len(results)} candidate answers"):
                    for i, r in enumerate(results, 1):
                        st.markdown(
                            f"**#{i}** `{r['score']:.1%}` — {r['answer']}"
                        )

            # ── Highlighted context ────────────────────────────────────────
            if show_highlight:
                start, end = best["start"], best["end"]
                before  = context[:start]
                span    = context[start:end]
                after   = context[end:]

                import html
                highlighted = (
                        html.escape(before)
                        + f'<span class="highlight-span">{html.escape(span)}</span>'
                        + html.escape(after)
                )

                st.markdown("**Answer in context:**")
                st.markdown(
                    f'<div class="highlight-context">{highlighted}</div>',
                    unsafe_allow_html=True,
                )

    else:
        # Placeholder state
        st.markdown(
            '<div style="color:#2a3040; font-family:\'DM Mono\',monospace; '
            'font-size:0.85rem; margin-top:3rem; text-align:center;">'
            '— answer appears here —</div>',
            unsafe_allow_html=True,
        )