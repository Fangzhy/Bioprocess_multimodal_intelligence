"""Tests for text/image retrieval and temporal event alignment."""

import numpy as np
import pytest

from src.data.events import align_event
from src.embeddings.store import (
    IMAGE_COLLECTION,
    TEXT_COLLECTION,
    get_client,
    search_similar_images,
)


def test_embedding_collections_have_expected_dimensions_and_counts() -> None:
    client = get_client()
    text = client.get_collection(TEXT_COLLECTION)
    images = client.get_collection(IMAGE_COLLECTION)
    assert text.count() == 200
    assert images.count() == 150
    assert len(text.peek(limit=1)["embeddings"][0]) == 384
    assert len(images.peek(limit=1)["embeddings"][0]) == 512


def test_image_search_excludes_query_and_returns_cosine_similarity() -> None:
    results = search_similar_images("I0092", limit=4)
    assert len(results) == 4
    assert all(result["image_id"] != "I0092" for result in results)
    assert all(np.isfinite(result["similarity"]) for result in results)


def test_event_alignment_uses_inclusive_window() -> None:
    event = align_event("B014", 72, 96)
    assert event.sensors["time_hr"].tolist() == list(range(72, 97, 4))
    assert event.notes["time_hr"].between(72, 96).all()
    assert event.images["time_hr"].tolist() == [96]
    assert event.outcome["batch_id"].tolist() == ["B014"]
    assert not event.sensor_summary.empty


def test_event_alignment_rejects_invalid_window() -> None:
    with pytest.raises(ValueError, match="Event window"):
        align_event("B014", 100, 72)
