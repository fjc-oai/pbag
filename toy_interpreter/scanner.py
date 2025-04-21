from dataclasses import dataclass
from enum import Enum
from typing import Any
from utils import is_digit, is_alpha, is_alpha_digit


class TokenType(Enum):
    LEFT_PAREN = "()"
    RIGHT_PAREN = ")"
    LEFT_BRACE = "["
    RIGHT_BRACE = "]"
    COMMA = ","
    DOT = "."
    MINUS = "-"
    PLUS = "+"
    SEMICOLON = ";"
    SLASH = "/"
    STAR = "*"
    # One or two character tokens.
    BANG = "!"
    BANG_EQUAL = "!="
    EQUAL = "="
    EQUAL_EQUAL = "=="
    GREATER = ">"
    GREATER_EQUAL = ">="
    LESS = "<"
    LESS_EQUAL = "<="
    # Literals.
    IDENTIFIER = "id"
    STRING = "str"
    NUMBER = "num"
    # Keywords.
    AND = "and"
    CLASS = "class"
    ELSE = "else"
    FALSE = "false"
    FUN = "fun"
    FOR = "for"
    IF = "if"
    NIL = "nil"
    OR = "or"
    PRINT = "print"
    RETURN = "ret"
    SUPER = "super"
    THIS = "this"
    TRUE = "true"
    VAR = "var"
    WHILE = "while"
    EOF = "eof"
    DEBUG = "debug"
    DISCARD = "discard"  # e.g. comment, space, etc

KEYWORDS: dict[str, TokenType] = {
    "and": TokenType.AND,
    "class": TokenType.CLASS,
    # ...
}


@dataclass
class Token:
    token_type: TokenType
    lexeme: str
    literal: Any
    lineno: int

    def __str__(self) -> str:
        return f"[{self.token_type}] {self.lexeme}"


class Scanner:
    def __init__(self, source: str):
        self._start = 0
        self._cur = 0
        self._lineno = 1
        self._source = source

    def scan(self) -> list[Token]:
        tokens = []
        while not self._is_at_end():
            self._start = self._cur
            token = self._scan_a_token()
            if token.token_type != TokenType.DISCARD:
                tokens.append(token)

        eof_token = Token(TokenType.EOF, "", None, self._lineno)
        tokens.append(eof_token)
        return tokens

    def _gen_token(self, token_type: TokenType, literal: Any = None) -> Token:
        lexeme = self._source[self._start : self._cur]
        return Token(token_type, lexeme, literal, self._lineno)

    def _is_at_end(self) -> bool:
        return self._cur >= len(self._source)

    def _advance(self) -> str:
        assert not self._is_at_end()
        c = self._source[self._cur]
        self._cur += 1
        return c

    def _peek(self) -> str:
        assert not self._is_at_end()
        return self._source[self._cur]

    def _match(self, char: str) -> bool:
        if self._is_at_end():
            return False
        if char != self._source[self._cur]:
            return False
        self._cur += 1
        return True

    def _string(self) -> Token:
        while not self._is_at_end() and self._peek() != '"':
            if self._peek() == "\n":
                self._lineno += 1
            self._advance()
        if self._peek() != '"':
            raise ValueError(f"Unterminated string")
        self._advance()
        value = self._source[self._start + 1 : self._cur]
        return self._gen_token(TokenType.STRING, value)

    def _number(self) -> Token:
        while not self._is_at_end() and is_digit(self._peek()):
            self._advance()
        if not self._is_at_end() and self._peek() == ".":
            self._advance()
            while not self._is_at_end() and is_digit(self._peek()):
                self._advance()
        value = eval(self._source[self._start : self._cur])
        return self._gen_token(TokenType.NUMBER, value)

    def _identifier(self) -> Token:
        while not self._is_at_end() and is_alpha_digit(self._peek()):
            self._advance()
        name = self._source[self._start:self._cur]
        if name in KEYWORDS:
            return self._gen_token(KEYWORDS[name])
        else:
            return self._gen_token(TokenType.IDENTIFIER)
        

    def _scan_a_token(self) -> None:
        c = self._advance()
        match c:
            case "(":
                return self._gen_token(TokenType.LEFT_PAREN)
            case ")":
                return self._gen_token(TokenType.RIGHT_PAREN)
            case "{":
                return self._gen_token(TokenType.LEFT_BRACE)
            case "}":
                return self._gen_token(TokenType.RIGHT_PAREN)
            case ",":
                return self._gen_token(TokenType.COMMA)
            case ".":
                return self._gen_token(TokenType.DOT)
            case "-":
                return self._gen_token(TokenType.MINUS)
            case "+":
                return self._gen_token(TokenType.PLUS)
            case ";":
                return self._gen_token(TokenType.SEMICOLON)
            case "*":
                return self._gen_token(TokenType.STAR)
            case "!":
                if self._match("="):
                    return self._gen_token(TokenType.BANG_EQUAL)
                else:
                    return self._gen_token(TokenType.EQUAL)
            case "=":
                if self._match("="):
                    return self._gen_token(TokenType.EQUAL_EQUAL)
                else:
                    return self._gen_token(TokenType.EQUAL)
            case "<":
                if self._match("="):
                    return self._gen_token(TokenType.LESS_EQUAL)
                else:
                    return self._gen_token(TokenType.LESS)
            case ">":
                if self._match("="):
                    return self._gen_token(TokenType.GREATER_EQUAL)
                else:
                    return self._gen_token(TokenType.GREATER)
            case "/":
                if self._match("/"):
                    while not self._is_at_end() and self._peek() != "\n":
                        self._advance()
                    return self._gen_token(TokenType.DISCARD)
                else:
                    return self._gen_token(TokenType.SLASH)
            case " ":
                return self._gen_token(TokenType.DISCARD)
            case "\r":
                return self._gen_token(TokenType.DISCARD)
            case "\t":
                return self._gen_token(TokenType.DISCARD)
            case "\n":
                self._lineno += 1
                return self._gen_token(TokenType.DISCARD)
            case '"':
                return self._string()
            case _:
                if is_digit(c):
                    return self._number()
                if is_alpha(c):
                    return self._identifier()

                return self._gen_token(TokenType.DISCARD)
                raise ValueError(f"Unexpected char {c} at {self._lineno}:{self._cur}")

        return Token(TokenType.DEBUG, c, None, self._lineno)


def test_scan():
    source = """
    // this is a comment
    (()) {} // groups
    +-*/=>= // ops
    "this is a string" // a string
    123.321 // a number
    and 
    and_is_a_var
    """
    scanner = Scanner(source)
    tokens = scanner.scan()
    print(f"Scanner: {len(tokens)} tokens")
    for i, token in enumerate(tokens):
        print(f"{i}: {token}")


def main():
    test_scan()


if __name__ == "__main__":
    main()
