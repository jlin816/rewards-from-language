flips = {
    "more": "less",
    "most": "least",
    "cheap": "expensive",
    "cheaper": "more expensive",
    "cheapest": "most expensive",
    "short": "long",
    "shorter": "longer",
    "shortest": "longest",
    "fewer": "many",
    "low": "high",
    "lower": "higher",
    "like": "don't like",
    "prefer": "don't want",
    "want": "don't want",
    "hate": "enjoy",
    "want to avoid": "want to get",
}

reverse_flips = {v: k for k, v in flips.items()}
flips.update(reverse_flips)


def flip(utt):
    """
    Args:
        utt (str)
    Returns:
        a list of flipped utterances
    """
    # replace longer substrings like "don't _" first
    for flip_tok in sorted(flips.keys(), key=len, reverse=True):
        if flip_tok in utt:
            yield utt.replace(flip_tok, flips[flip_tok], 1)
