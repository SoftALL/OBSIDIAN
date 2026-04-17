def clean_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    return " ".join(text.split())