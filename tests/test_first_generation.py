"""Tests for the first text-generation example; inference is not executed in tests."""

import pytest

from ai_engineering_lab.text_generation.first_generation import (
    choose_device,
    extract_generated_text,
)


@pytest.mark.parametrize(
    ("cuda_available", "expected"),
    [(True, "cuda"), (False, "cpu")],
)
def test_choose_device(cuda_available: bool, expected: str) -> None:
    assert choose_device(cuda_available) == expected


def test_extract_generated_text_from_plain_text() -> None:
    assert extract_generated_text([{"generated_text": "Hello there!"}]) == "Hello there!"


def test_extract_generated_text_from_chat_messages() -> None:
    result = [
        {
            "generated_text": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello there!"},
            ]
        }
    ]
    assert extract_generated_text(result) == "Hello there!"


def test_extract_generated_text_rejects_unknown_shape() -> None:
    with pytest.raises(ValueError, match="Unexpected"):
        extract_generated_text([{"generated_text": {"text": "unexpected"}}])
