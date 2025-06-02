def tokenize(input_text):
    """
    Tokenize a string.
    This is a simple tokenizer that splits the string into tokens
    based on whitespace.
    It also removes any non-ASCII characters.
    It also removes any trailing underscores.
    It also removes any trailing numbers.
    It also removes any trailing punctuation.
    """
    cleaned_text = clean_unicode(input_text)
    tokens = cleaned_text.lower().split()
    if input_text.endswith("_"):
        """
        Discard the last two tokens since
        there will likely be a word cut in two.
        """
        tokens = tokens[:-2]
    return tokens


def get_bigrams(num_bigrams, tokens):
    """
    Get bigrams (a sequence of two adjacent tokens) 
    from a list of tokens.
    This is a simple function that takes a list 
    of tokens and returns a list of bigrams.
    """
    num_bigrams = min(num_bigrams, len(tokens) - 1)
    bigrams = [f"{tokens[i]} {tokens[i + 1]}" for i in range(num_bigrams)]
    return bigrams


def clean_unicode(s: str) -> str:
    """
    Clean unicode from a string.
    This is a simple function that takes a string and returns
    a string with any non-ASCII characters removed.
    """
    return s.encode("utf-8", errors="ignore").decode("utf-8")
