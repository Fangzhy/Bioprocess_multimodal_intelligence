"""Tests for text/image retrieval and temporal event alignment."""

from unittest.mock import Mock, call

import numpy as np
import pytest

from src.data.events import align_event
from src.embeddings import store
from src.embeddings.store import (
    IMAGE_COLLECTION,
    TEXT_COLLECTION,
    get_client,
    search_similar_images,
)


@pytest.fixture
def isolated_text_encoder_cache():
    store.get_cached_text_encoder.cache_clear()
    yield
    store.get_cached_text_encoder.cache_clear()


def test_note_search_downloads_missing_encoder_and_reuses_it(
    monkeypatch, isolated_text_encoder_cache
) -> None:
    with np.load(store.EMBEDDING_PATH / "text_embeddings.npz") as embeddings:
        query_vector = embeddings["vectors"][0].copy()
        expected_text_id = str(embeddings["ids"][0])
        batch_id = str(embeddings["batch_ids"][0])
    encoder = Mock()
    encoder.encode.return_value = query_vector[np.newaxis, :]
    loader = Mock(side_effect=[OSError("Checkpoint is not cached"), encoder])
    monkeypatch.setattr(store, "load_text_encoder", loader)

    for query in ("oxygen control problem", "culture observations"):
        results = store.search_notes(query, limit=2, batch_id=batch_id)
        assert len(results) == 2
        assert results[0]["text_id"] == expected_text_id
        assert all(result["batch_id"] == batch_id for result in results)

    assert loader.call_args_list == [
        call(local_files_only=True),
        call(local_files_only=False),
    ]
    assert encoder.encode.call_count == 2


def test_text_encoder_uses_local_checkpoint_without_downloading(
    monkeypatch, isolated_text_encoder_cache
) -> None:
    encoder = Mock()
    loader = Mock(return_value=encoder)
    monkeypatch.setattr(store, "load_text_encoder", loader)

    assert store.get_cached_text_encoder() is encoder
    assert store.get_cached_text_encoder() is encoder
    loader.assert_called_once_with(local_files_only=True)


def test_failed_encoder_download_can_be_retried(
    monkeypatch, isolated_text_encoder_cache
) -> None:
    encoder = Mock()
    loader = Mock(
        side_effect=[
            OSError("Checkpoint is not cached"),
            OSError("Download is unavailable"),
            OSError("Checkpoint is still not cached"),
            encoder,
        ]
    )
    monkeypatch.setattr(store, "load_text_encoder", loader)

    with pytest.raises(OSError, match="Download is unavailable"):
        store.get_cached_text_encoder()
    assert store.get_cached_text_encoder() is encoder
    assert store.get_cached_text_encoder() is encoder
    assert loader.call_count == 4


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
