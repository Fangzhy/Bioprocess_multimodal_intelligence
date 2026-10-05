"""Semantic search over synthetic scientist notes."""

import pandas as pd
import streamlit as st
from chromadb.errors import NotFoundError

from src.embeddings.store import CHROMA_PATH, TEXT_COLLECTION, get_client, search_notes

st.set_page_config(page_title="Text Intelligence", page_icon="📝", layout="wide")
st.title("Text Intelligence")
st.caption("Sentence Transformer semantic search · Synthetic scientist notes")

try:
    count = get_client().get_collection(TEXT_COLLECTION).count()
except NotFoundError:
    st.error("Text embeddings are missing. Run `python -m src.embeddings.store`.")
    st.stop()

st.metric("Embedded notes", count)
query = st.text_input("Search by meaning", "oxygen control problem")
limit = st.slider("Number of results", 3, 12, 5)
if st.button("Search notes", type="primary"):
    with st.spinner("Encoding query and searching ChromaDB..."):
        results = pd.DataFrame(search_notes(query, limit))
    st.dataframe(
        results[["batch_id", "time_hr", "text_type", "content", "similarity"]],
        hide_index=True,
        use_container_width=True,
    )

with st.expander("How this works"):
    st.write(
        f"Notes are stored in `{TEXT_COLLECTION}` under `{CHROMA_PATH}` using cosine "
        "distance. Similarity is displayed as 1 - cosine distance."
    )
