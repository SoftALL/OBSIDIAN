import pandas as pd
from src.preprocess import clean_text
from src.inference import predict_text


def detect_text_column(df: pd.DataFrame):
    candidates = ["cleaned_text", "text", "tweet", "tweet_text", "content"]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def run_batch_inference(df: pd.DataFrame, tokenizer, model):
    text_col = detect_text_column(df)
    if text_col is None:
        raise ValueError(
            "No supported text column found. Use one of: cleaned_text, text, tweet, tweet_text, content."
        )

    result_df = df.copy()
    result_df[text_col] = result_df[text_col].astype(str).fillna("").apply(clean_text)

    labels = []
    confidences = []

    for text in result_df[text_col]:
        pred = predict_text(text, tokenizer, model)
        labels.append(pred["label"])
        confidences.append(pred["confidence"])

    result_df["predicted_label"] = labels
    result_df["confidence"] = confidences
    return result_df, text_col