"""CLIP similarity search for synthetic microscopy images."""

from pathlib import Path

import pandas as pd
import streamlit as st
from chromadb.errors import NotFoundError

from src.data.repository import load_batch_catalog, load_images
from src.embeddings.store import IMAGE_COLLECTION, get_client, search_similar_images

PROJECT_ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title="Microscopy Intelligence", page_icon="🖼️", layout="wide")
st.title("Microscopy Intelligence")
st.caption("CLIP image retrieval · Programmatic synthetic microscopy illustrations")

try:
    embedded_count = get_client().get_collection(IMAGE_COLLECTION).count()
except NotFoundError:
    st.error("Image embeddings are missing. Run `python -m src.embeddings.store`.")
    st.stop()

images = load_images()
catalog = load_batch_catalog()
labels = images.assign(label=images["batch_id"] + " · " + images["time_hr"].astype(str) + " h")
default = labels.index[
    labels["batch_id"].eq("B031") & labels["time_hr"].eq(96)
]
selected_index = int(default[0]) if len(default) else 0
selected_label = st.selectbox("Query image", labels["label"], index=selected_index)
selected = labels.loc[labels["label"].eq(selected_label)].iloc[0]
st.image(str(PROJECT_ROOT / selected["file_path"]), width=260, caption=selected_label)

if st.button("Find similar images", type="primary"):
    results = pd.DataFrame(search_similar_images(selected["image_id"], limit=5))
    results = results.merge(
        catalog[["batch_id", "final_titer_g_l", "final_viability_pct"]],
        on="batch_id",
        how="left",
    )
    columns = st.columns(len(results))
    for column, (_, row) in zip(columns, results.iterrows(), strict=True):
        with column:
            st.image(str(PROJECT_ROOT / row["file_path"]), use_container_width=True)
            st.write(f"**{row['batch_id']} · {row['time_hr']} h**")
            st.write(f"Similarity: {row['similarity']:.3f}")
            st.write(f"Titer: {row['final_titer_g_l']:.2f} g/L")
            st.write(f"Viability: {row['final_viability_pct']:.1f}%")

st.info(f"{embedded_count} illustrative images are indexed. Similarity is exploratory evidence.")
