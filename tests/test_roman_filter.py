from src.data_prep import is_sentence_roman_only, is_sentence_majority_roman

def test_roman_only():
    assert is_sentence_roman_only(["hello","world"]) is True
    assert is_sentence_roman_only(["hello","कोड"]) is False

def test_majority_roman():
    assert is_sentence_majority_roman(["hello","world","!"]) is True
    assert is_sentence_majority_roman(["NYUAD","rocks","कोड"]) is True   # 2 roman vs 1 dev
    assert is_sentence_majority_roman(["यह","कोड"]) is False            # 0 roman vs 2 dev
