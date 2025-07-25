
from transtrans.deoral import match_repetitions_robust


def test_match_repetitions_robust():
    tokens = ["我","感到","快乐","啊","快乐","我","感到","啊","快乐","啊","快乐"]
    tags = ["r", "v", "n", "interj", "n", "r", "v", "interj", "n", "interj", "n"]

    
    result = match_repetitions_robust(tokens, tags, ngram=1, ignore_tags=['interj'])
    assert set(result) == {2,3,8,9}

    result = match_repetitions_robust(tokens, tags, ngram=2, ignore_tags=['interj'])
    assert set(result) == set()

    result = match_repetitions_robust(tokens, tags, ngram=3, ignore_tags=['interj'])
    assert set(result) == set()

    result = match_repetitions_robust(tokens, tags, ngram=4, ignore_tags=['interj'])
    assert set(result) == {0,1,2,3,4}

    result = match_repetitions_robust(tokens, tags, ngram=5, ignore_tags=['interj'])
    assert set(result) == set()