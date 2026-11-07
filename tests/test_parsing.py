import chain
from utils.citations import sources_label
from utils.llm import is_rate_limit, retry_after_seconds


def run_stream(chunks):
    return "".join(chain._strip_reasoning_stream(iter(chunks)))


def test_reasoning_removed_even_when_tags_are_split():
    assert run_stream(["<thi", "nk>plan</th", "ink>\n\nHello ", "world"]) == "Hello world"
    assert run_stream(["Plain ", "answer"]) == "Plain answer"
    assert run_stream(["a<", "b"]) == "a<b"


def test_retry_after_seconds():
    assert retry_after_seconds(Exception("Please try again in 7m12.5s.")) == 433
    assert retry_after_seconds(Exception("Please try again in 1h2m3s")) == 3723
    assert retry_after_seconds(Exception("Please try again in 850ms")) == 1
    assert retry_after_seconds(Exception("no hint")) == 60


def test_is_rate_limit():
    class RateLimitError(Exception):
        pass
    assert is_rate_limit(RateLimitError("too many"))
    assert is_rate_limit(Exception("Error code: 429"))
    assert not is_rate_limit(ValueError("bad input"))


class Doc:
    def __init__(self, source, page):
        self.metadata = {"source": source, "page": page}


def test_sources_label():
    assert sources_label([Doc("a.pdf", 0), Doc("a.pdf", 1)]) == "Pages 1, 2"
    assert sources_label([Doc("a.pdf", 0), Doc("b.pdf", 2)]) == "a.pdf p. 1; b.pdf p. 3"
    assert sources_label([]) == ""
