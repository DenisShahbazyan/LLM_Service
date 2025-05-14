from enum import StrEnum, auto


class TokenDirection(StrEnum):
    """Направление подсчета токенов."""

    ENCODE = auto()
    DECODE = auto()
