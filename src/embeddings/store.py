"""Build and query Chroma collections for notes and microscopy images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
import open_clip
import torch
from chromadb.errors import NotFoundError
from PIL import Image
from sentence_transformers import SentenceTransformer

from src.data.repository import load_images, load_text_records

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_PATH = PROJECT_ROOT / "database" / "chroma"
EMBEDDING_PATH = PROJECT_ROOT / "data" / "processed" / "embeddings"
TEXT_COLLECTION = "bioprocess_notes"
IMAGE_COLLECTION = "microscopy_images"
TEXT_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
CLIP_MODEL_ID = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"


def get_client(path: Path = CHROMA_PATH) -> chromadb.PersistentClient:
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


def load_text_encoder(*, local_files_only: bool = False) -> SentenceTransformer:
    return SentenceTransformer(TEXT_MODEL_ID, local_files_only=local_files_only)


def load_image_encoder():
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_ID, pretrained=CLIP_PRETRAINED, device="cpu"
    )
    model.eval()
    return model, preprocess


def encode_texts(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    return np.asarray(
        model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
        dtype=np.float32,
    )


def encode_images(paths: list[Path], model, preprocess, batch_size: int = 32) -> np.ndarray:
    encoded: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            tensors = []
            for path in paths[start : start + batch_size]:
                with Image.open(path) as image:
                    tensors.append(preprocess(image.convert("RGB")))
            batch = torch.stack(tensors)
            values = model.encode_image(batch)
            values = values / values.norm(dim=-1, keepdim=True)
            encoded.append(values.cpu().numpy().astype(np.float32))
    return np.vstack(encoded)


def _replace_collection(client, name: str):
    try:
        client.delete_collection(name)
    except NotFoundError:
        pass
    return client.create_collection(name, metadata={"hnsw:space": "cosine"})


def build_embedding_store(
    chroma_path: Path = CHROMA_PATH,
    embedding_path: Path = EMBEDDING_PATH,
) -> dict[str, Any]:
    notes = load_text_records()
    images = load_images()
    text_model = load_text_encoder()
    text_vectors = encode_texts(notes["content"].tolist(), text_model)

    image_model, image_preprocess = load_image_encoder()
    image_paths = [PROJECT_ROOT / path for path in images["file_path"]]
    image_vectors = encode_images(image_paths, image_model, image_preprocess)

    client = get_client(chroma_path)
    text_collection = _replace_collection(client, TEXT_COLLECTION)
    text_collection.add(
        ids=notes["text_id"].tolist(),
        embeddings=text_vectors.tolist(),
        documents=notes["content"].tolist(),
        metadatas=notes[["batch_id", "time_hr", "text_type"]].to_dict("records"),
    )
    image_collection = _replace_collection(client, IMAGE_COLLECTION)
    image_collection.add(
        ids=images["image_id"].tolist(),
        embeddings=image_vectors.tolist(),
        metadatas=images[
            ["batch_id", "time_hr", "image_type", "file_path"]
        ].to_dict("records"),
    )

    embedding_path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        embedding_path / "text_embeddings.npz",
        ids=notes["text_id"].to_numpy(dtype=str),
        batch_ids=notes["batch_id"].to_numpy(dtype=str),
        vectors=text_vectors,
    )
    np.savez_compressed(
        embedding_path / "image_embeddings.npz",
        ids=images["image_id"].to_numpy(dtype=str),
        batch_ids=images["batch_id"].to_numpy(dtype=str),
        vectors=image_vectors,
    )
    manifest = {
        "text": {
            "model": TEXT_MODEL_ID,
            "dimension": int(text_vectors.shape[1]),
            "count": len(text_vectors),
            "normalized": True,
        },
        "image": {
            "model": CLIP_MODEL_ID,
            "pretrained": CLIP_PRETRAINED,
            "dimension": int(image_vectors.shape[1]),
            "count": len(image_vectors),
            "normalized": True,
        },
        "distance_metric": "cosine",
    }
    (embedding_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def search_notes(query: str, limit: int = 5, batch_id: str | None = None) -> list[dict]:
    vector = encode_texts([query], load_text_encoder(local_files_only=True))[0]
    collection = get_client().get_collection(TEXT_COLLECTION)
    where = {"batch_id": batch_id} if batch_id else None
    result = collection.query(
        query_embeddings=[vector.tolist()],
        n_results=limit,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "text_id": result["ids"][0][index],
            "content": result["documents"][0][index],
            **result["metadatas"][0][index],
            "similarity": 1 - result["distances"][0][index],
        }
        for index in range(len(result["ids"][0]))
    ]


def search_similar_images(image_id: str, limit: int = 5) -> list[dict]:
    collection = get_client().get_collection(IMAGE_COLLECTION)
    stored = collection.get(ids=[image_id], include=["embeddings"])
    if not stored["ids"]:
        raise ValueError(f"Unknown image: {image_id}")
    result = collection.query(
        query_embeddings=[stored["embeddings"][0]],
        n_results=limit + 1,
        include=["metadatas", "distances"],
    )
    rows = []
    for index, result_id in enumerate(result["ids"][0]):
        if result_id == image_id:
            continue
        rows.append(
            {
                "image_id": result_id,
                **result["metadatas"][0][index],
                "similarity": 1 - result["distances"][0][index],
            }
        )
    return rows[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chroma-path", type=Path, default=CHROMA_PATH)
    args = parser.parse_args()
    manifest = build_embedding_store(args.chroma_path)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
