"""Grounded scientific explanations through OpenRouter."""

import streamlit as st

from src.copilot import ask_openrouter, build_evidence_bundle
from src.data.repository import load_batch_catalog

st.set_page_config(page_title="Scientific Copilot", page_icon="💬", layout="wide")
st.title("Scientific Copilot")
st.caption("Evidence-grounded explanation using a free-tier OpenRouter model")

batch_ids = load_batch_catalog()["batch_id"].tolist()
batch_id = st.selectbox("Batch", batch_ids, index=batch_ids.index("B014"))
question = st.text_area("Question", f"Why did {batch_id} differ from its historical reference?")
evidence = build_evidence_bundle(batch_id)

with st.expander("Evidence supplied to the model", expanded=True):
    st.json(evidence)

if st.button("Generate explanation", type="primary"):
    try:
        with st.spinner("Requesting a grounded explanation from OpenRouter..."):
            result = ask_openrouter(question, evidence)
        st.markdown(result["answer"])
        st.caption(
            f"Requested model: {result['requested_model']} · Served model: {result['served_model']}"
        )
    except RuntimeError as error:
        st.error(str(error))
        st.info("The complete evidence remains available above for manual interpretation.")

st.warning(
    "This educational explanation uses synthetic data. It supports investigation and is not "
    "a validated process decision or causal conclusion."
)
