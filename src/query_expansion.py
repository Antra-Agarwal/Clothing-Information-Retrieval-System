from src.preprocess import normalize_tokens

CLOTHING_SYNONYMS = {
    "shirt": ["top", "blouse", "tee"],
    "pant": ["trouser", "jean", "bottom"],
    "shoe": ["footwear", "sneaker"],
    "dress": ["gown", "frock"],
    "jacket": ["coat", "blazer"],
    "sweatshirt": ["hoodie", "pullover"],
}

def expand_query(query):
    original = normalize_tokens(query)
    expanded, seen = [], set()
    for term in original:
        if term not in seen:
            expanded.append((term, 1.0, "original")); seen.add(term)
        for synonym in CLOTHING_SYNONYMS.get(term, []):
            for normalized in normalize_tokens(synonym):
                if normalized not in seen:
                    expanded.append((normalized, 0.35, "synonym of " + term)); seen.add(normalized)
    return original, expanded
