def is_digit(char: str) -> bool:
    assert len(char) == 1, f"{char=}"
    return "0" <= char <= "9"

def is_alpha(char: str) -> bool:
    assert len(char) == 1, f"{char=}"
    return ('a' <= char <= 'z') or ('A' <= char <= 'Z') or (char == '_')

def is_alpha_digit(char: str) -> bool:
    return is_digit(char) or is_alpha(char)
