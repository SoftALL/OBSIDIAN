import pandas as pd
import streamlit as st

from src.preprocess import clean_text
from src.inference import predict_text


def get_text_column_candidates(df: pd.DataFrame):
    preferred_names = ["cleaned_text", "text", "tweet", "tweet_text", "content"]

    lower_to_actual = {col.lower(): col for col in df.columns}
    matches = []

    for name in preferred_names:
        if name in lower_to_actual:
            matches.append(lower_to_actual[name])

    return matches


def run_batch_inference(df: pd.DataFrame, tokenizer, model, text_col: str):
    if text_col not in df.columns:
        raise ValueError(f"Selected text column '{text_col}' does not exist in the uploaded file.")

    result_df = df.copy()
    result_df[text_col] = result_df[text_col].fillna("").astype(str).apply(clean_text)

    labels = []
    confidences = []

    total_rows = len(result_df)

    progress_bar = st.progress(0, text="Starting batch prediction...")
    status_text = st.empty()

    for idx, text in enumerate(result_df[text_col], start=1):
        pred = predict_text(text, tokenizer, model)
        labels.append(pred["label"])
        confidences.append(pred["confidence"])

        progress = idx / total_rows
        progress_bar.progress(
            progress,
            text=f"Processing rows: {idx}/{total_rows} ({progress:.1%})"
        )
        status_text.caption(f"Processed {idx} out of {total_rows} rows.")

    result_df["predicted_label"] = labels
    result_df["confidence"] = confidences
    result_df["confidence_percent"] = result_df["confidence"].apply(lambda x: f"{x:.2%}")

    progress_bar.progress(1.0, text="Batch prediction completed.")
    status_text.caption(f"Finished processing all {total_rows} rows.")

    return result_df