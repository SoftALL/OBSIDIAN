import pandas as pd
from src.preprocess import clean_text
from src.inference import predict_text


def get_text_column_candidates(df: pd.DataFrame):
    candidates = ["cleaned_text", "text", "tweet", "tweet_text", "content"]
    return [col for col in candidates if col in df.columns]


def run_batch_inference(df: pd.DataFrame, tokenizer, model, text_col: str):
    if text_col not in df.columns:
        raise ValueError(f"Selected text column '{text_col}' does not exist in the uploaded file.")

    result_df = df.copy()
    result_df[text_col] = result_df[text_col].fillna("").astype(str).apply(clean_text)

    labels = []
    confidences = []

    for text in result_df[text_col]:
        pred = predict_text(text, tokenizer, model)
        labels.append(pred["label"])
        confidences.append(pred["confidence"])

    result_df["predicted_label"] = labels
    result_df["confidence"] = confidences
    result_df["confidence_percent"] = result_df["confidence"].apply(lambda x: f"{x:.2%}")

    return result_df