import pytest

from utils import ocr


class FakeResponse:
    def __init__(self, status, text=""):
        self.status_code = status
        self._text = text

    def json(self):
        return {"candidates": [{"content": {"parts": [{"text": self._text}]}}]}


def test_pages_are_split_by_markers(monkeypatch):
    text = "=== Page 1 ===\nINVOICE MKT-38\n\n=== Page 2 ===\nThank you\n"
    monkeypatch.setattr(ocr.requests, "post", lambda *a, **k: FakeResponse(200, text))
    assert ocr.ocr_pdf(b"%PDF-1.4") == [(1, "INVOICE MKT-38"), (2, "Thank you")]


def test_rate_limit_gives_friendly_error(monkeypatch):
    monkeypatch.setattr(ocr.requests, "post", lambda *a, **k: FakeResponse(429))
    with pytest.raises(ValueError, match="busy"):
        ocr.ocr_pdf(b"%PDF-1.4")


def test_api_key_goes_in_header_not_url(monkeypatch):
    seen = {}

    def fake_post(url, **kwargs):
        seen.update(url=url, headers=kwargs["headers"])
        return FakeResponse(200, "=== Page 1 ===\nx")

    monkeypatch.setattr(ocr.requests, "post", fake_post)
    ocr.ocr_pdf(b"%PDF-1.4")
    assert "key=" not in seen["url"] and "x-goog-api-key" in seen["headers"]


def test_too_large_is_rejected():
    with pytest.raises(ValueError, match="too large"):
        ocr.ocr_pdf(b"x" * (ocr.MAX_OCR_BYTES + 1))
