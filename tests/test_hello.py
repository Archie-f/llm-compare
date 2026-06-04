def test_addition():
    assert 1 + 1 == 2

def test_string_upper():
    result = 'hello'.upper()
    assert result == 'HELLO'

def test_list_length():
    items = [1, 2, 3]
    assert len(items) == 3