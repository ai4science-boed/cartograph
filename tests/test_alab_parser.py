from alab_retraction_audit import parse_formula


def test_parse_formula_with_parentheses():
    assert parse_formula("Mn7(P2O7)4") == {"Mn": 7.0, "P": 8.0, "O": 28.0}


def test_parse_formula_with_decimal_occupancy():
    parsed = parse_formula("Ba0.5Sr0.5TiO3")
    assert parsed == {"Ba": 0.5, "Sr": 0.5, "Ti": 1.0, "O": 3.0}

