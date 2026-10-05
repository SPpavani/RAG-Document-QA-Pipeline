from evaluate import first_hit_rank, hit_rate, mrr, normalize, percentile


def test_normalize_collapses_whitespace_and_case():
    assert normalize("12   MILLION\ndollars") == "12 million dollars"


def test_first_hit_rank_finds_first_matching_chunk():
    chunks = ["nothing here", "invest 12 million dollars", "12 million again"]
    assert first_hit_rank(chunks, ["12 million"]) == 2


def test_first_hit_rank_requires_all_keywords():
    chunks = ["only budget", "budget and Hyderabad"]
    assert first_hit_rank(chunks, ["budget", "hyderabad"]) == 2
    assert first_hit_rank(chunks, ["budget", "mumbai"]) is None


def test_first_hit_rank_handles_line_breaks():
    assert first_hit_rank(["12 million\ndollars"], ["12 million dollars"]) == 1


def test_hit_rate():
    assert hit_rate([1, 3, None, 2], k=2) == 0.5
    assert hit_rate([], k=3) == 0.0


def test_mrr():
    assert mrr([1, 2, None]) == (1 + 0.5) / 3
    assert mrr([]) == 0.0


def test_percentile():
    values = [1, 2, 3, 4, 5]
    assert percentile(values, 50) == 3
    assert percentile(values, 95) == 5
    assert percentile([], 50) == 0.0
