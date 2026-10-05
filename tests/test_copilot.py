"""Tests for grounded copilot evidence and OpenRouter request handling."""

from unittest.mock import Mock, patch

from src.copilot import ask_openrouter, build_evidence_bundle, build_prompt


def test_evidence_bundle_contains_traceable_values() -> None:
    evidence = build_evidence_bundle("B014")
    assert evidence["batch_id"] == "B014"
    assert evidence["outcome"]["final_titer_g_l"] > 0
    assert evidence["sensor_comparison"]["do_std_pct"]["selected"] > 0
    assert any("DO" in note["content"] for note in evidence["notes"])
    assert len(evidence["similar_images"]) == 3


def test_prompt_requires_grounded_hypotheses() -> None:
    messages = build_prompt("Why?", {"batch_id": "B014"})
    assert "Use only the supplied evidence" in messages[0]["content"]
    assert "hypotheses" in messages[0]["content"]
    assert "B014" in messages[1]["content"]


@patch("src.copilot.requests.post")
def test_openrouter_response_is_parsed(mock_post: Mock, monkeypatch) -> None:
    monkeypatch.setenv("OPEN_ROUTER_API", "test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "openrouter/free")
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "model": "free-test-model",
        "choices": [{"message": {"content": "Evidence-grounded answer"}}],
        "usage": {"prompt_tokens": 10},
    }
    mock_post.return_value = response

    result = ask_openrouter("Why?", {"batch_id": "B014"})
    assert result["answer"] == "Evidence-grounded answer"
    assert result["served_model"] == "free-test-model"
    authorization = mock_post.call_args.kwargs["headers"]["Authorization"]
    assert authorization == "Bearer test-key"
