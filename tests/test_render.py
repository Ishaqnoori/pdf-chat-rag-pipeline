from utils.render import _fix_tables, split_followups, stream_preview


def test_split_followups():
    text = "Total is **348**.\n\n```followups\n- Tax amount?\n- Who is the seller?\n- When paid?\n```"
    body, questions = split_followups(text)
    assert body == "Total is **348**."
    assert questions == ["Tax amount?", "Who is the seller?", "When paid?"]


def test_split_followups_without_block():
    assert split_followups("Just an answer") == ("Just an answer", [])


def test_stream_preview_hides_half_written_blocks():
    assert "```" not in stream_preview('Table\n```chart\n{"type": "ba')
    assert "Drawing chart" in stream_preview('Table\n```chart\n{"type": "ba')
    assert stream_preview("Answer\n```follow") == "Answer\n"
    assert stream_preview("Use `code` here") == "Use `code` here"


def test_fix_tables_adds_blank_lines():
    assert _fix_tables("Here:\n|a|b|\n|-|-|\n|1|2|\nAfter") == "Here:\n\n|a|b|\n|-|-|\n|1|2|\n\nAfter"
