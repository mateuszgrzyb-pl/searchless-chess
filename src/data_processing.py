"""Module with utility functions for file processing."""


def allowed_pieces_only(fen: str, allowed_pieces: str) -> bool:
    """Checks if only allowed pieces are present in the given FEN string.

    Args:
        fen: FEN notation string (full format). Only the piece placement section
            (first field before space) is validated.
        allowed_pieces: String containing allowed piece symbols. Case-insensitive.

    Returns:
        True if all letters in the FEN placement section belong to the set of
        allowed symbols; False otherwise.
    """
    placement = fen.partition(' ')[0]
    allowed = set(allowed_pieces.lower()) | {'k'}
    contains = allowed.__contains__
    for ch in placement:
        if ch.isalpha() and not contains(ch.lower()):
            return False
    return True