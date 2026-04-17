import streamlit as st
import pandas as pd
import plotly.express as px

from src.preprocess import clean_text
from src.inference import load_model_and_tokenizer, predict_text
from src.batch import run_batch_inference, get_text_column_candidates
from src.utils import load_uploaded_file

st.set_page_config(page_title="OBSIDIAN Arabic Tweet Classifier", layout="wide")

st.title("OBSIDIAN Arabic Tweet Classifier")
st.markdown(
    """
    This app uses a fine-tuned **AraBERT** model to classify Arabic tweets or short texts into **5 categories**:

    - **Threat**
    - **Violence**
    - **Distress**
    - **Complaint**
    - **Neutral**
    """
)

st.info(
    "Choose **Single Text** to classify one Arabic sentence, or **Batch Upload** to classify a CSV/XLSX file."
)

with st.expander("How to use this app"):
    st.markdown(
        """
        **Single Text**
        - Paste one Arabic sentence or tweet
        - Click **Predict**
        - Review the predicted label, confidence, and probability chart

        **Batch Upload**
        - Upload a CSV or XLSX file
        - Select the text column to classify
        - Click **Run Batch Prediction**
        - Review the preview, label distribution chart, and download the full results
        """
    )

with st.expander("Example Arabic inputs"):
    st.markdown(
        """
        **Threat**
        - سأقتلك إذا رأيتك مرة أخرى

        **Violence**
        - قاموا بضرب الرجل في الشارع بعنف شديد

        **Distress**
        - أنا خائف جدًا ولا أعرف ماذا أفعل، أحتاج مساعدة

        **Complaint**
        - الخدمة سيئة جدًا والتطبيق يتعطل كل مرة

        **Neutral**
        - الجو اليوم معتدل والناس في الحديقة
        """
    )


@st.cache_resource
def get_model():
    return load_model_and_tokenizer()


try:
    tokenizer, model = get_model()
    model_loaded = True
except Exception as e:
    tokenizer, model = None, None
    model_loaded = False
    st.warning("Model could not be loaded yet. Make sure model.safetensors is placed inside the model folder.")
    st.caption(str(e))

tab1, tab2 = st.tabs(["Single Text", "Batch Upload"])

with tab1:
    st.subheader("Single Text Classification")
    st.caption("Enter one Arabic text or tweet, then click Predict.")

    user_text = st.text_area("Arabic text", height=150, placeholder="اكتب النص العربي هنا...")

    if st.button("Predict", width="stretch"):
        if not user_text.strip():
            st.error("Please enter some text first.")
        elif not model_loaded:
            st.error("Model is not loaded.")
        else:
            cleaned = clean_text(user_text)
            result = predict_text(cleaned, tokenizer, model)

            col1, col2 = st.columns([2, 1])

            with col1:
                st.success(f"Predicted label: **{result['label']}**")

            with col2:
                st.metric("Confidence", f"{result['confidence']:.2%}")

            probs_df = pd.DataFrame({
                "Label": list(result["probabilities"].keys()),
                "Probability": list(result["probabilities"].values())
            })

            st.write("### Prediction Probability Distribution")
            fig = px.bar(
                probs_df,
                x="Label",
                y="Probability",
                title="Probability by Class"
            )
            st.plotly_chart(fig, width="stretch")

            st.write("### Class Probability Table")
            st.dataframe(probs_df, width="stretch")

with tab2:
    st.subheader("Batch File Classification")
    st.caption("Upload a CSV or XLSX file, choose the text column, and run batch prediction.")

    uploaded_file = st.file_uploader("Upload CSV or XLSX file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            df = load_uploaded_file(uploaded_file)

            total_rows = len(df)
            total_cols = len(df.columns)

            st.write(f"Uploaded file contains **{total_rows} rows** and **{total_cols} columns**.")
            st.write("### Uploaded Data Preview")
            st.caption("Showing first 10 rows only.")
            st.dataframe(df.head(10), width="stretch")

            candidate_cols = get_text_column_candidates(df)

            if candidate_cols:
                selected_text_col = st.selectbox(
                    "Select the text column to classify",
                    options=candidate_cols,
                    index=0
                )
            else:
                st.error(
                    "No supported text column was detected automatically.\n\n"
                    "Expected one of these column names (case-insensitive): "
                    "cleaned_text, text, tweet, tweet_text, content."
                )
                st.write("Available columns in your file:")
                st.write(list(df.columns))
                selected_text_col = None

            if selected_text_col is not None:
                st.caption(f"Selected text column: {selected_text_col}")

                preview_df = df[[selected_text_col]].head(5).copy()
                st.write("### Selected Text Column Preview")
                st.caption("Showing first 5 rows only.")
                st.dataframe(preview_df, width="stretch")

                if st.button("Run Batch Prediction", width="stretch"):
                    if not model_loaded:
                        st.error("Model is not loaded.")
                    else:
                        result_df = run_batch_inference(df, tokenizer, model, selected_text_col)

                        st.success(
                            f"Predictions completed successfully using column: {selected_text_col}"
                        )

                        display_cols = [selected_text_col, "predicted_label", "confidence_percent"]
                        st.write("### Classification Results Preview")
                        st.caption("Showing first 20 rows only. Download the CSV for the full output.")
                        st.dataframe(result_df[display_cols].head(20), width="stretch")

                        counts = result_df["predicted_label"].value_counts().reset_index()
                        counts.columns = ["Label", "Count"]

                        st.write("### Predicted Label Distribution")
                        fig = px.pie(
                            counts,
                            names="Label",
                            values="Count",
                            title="Distribution of Predicted Labels"
                        )
                        st.plotly_chart(fig, width="stretch")

                        csv_data = result_df.to_csv(index=False).encode("utf-8-sig")
                        st.download_button(
                            "Download Results as CSV",
                            data=csv_data,
                            file_name="obsidian_predictions.csv",
                            mime="text/csv",
                            width="stretch"
                        )

        except Exception as e:
            st.error(f"Error while processing file: {str(e)}")