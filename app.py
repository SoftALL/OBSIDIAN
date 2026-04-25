import streamlit as st
import pandas as pd
import plotly.express as px

from src.preprocess import clean_text
from src.inference import load_model_and_tokenizer, predict_text
from src.batch import run_batch_inference, get_text_column_candidates
from src.utils import load_uploaded_file
from src.live import (
    load_demo_live_tweets,
    fetch_live_tweets_from_n8n,
    classify_live_tweets,
    get_live_summary,
    dataframe_to_excel_bytes,
)


def apply_custom_branding():
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .obsidian-hero {
            background: linear-gradient(180deg, #162A36 0%, #10202B 100%);
            border: 1px solid #233847;
            border-radius: 18px;
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
        }

        .obsidian-title {
            font-size: 3rem;
            font-weight: 800;
            color: #F5F1E8;
            line-height: 1;
            margin-bottom: 0.45rem;
        }

        .obsidian-subtitle {
            font-size: 1.05rem;
            color: #CFC6B8;
            margin-bottom: 0.8rem;
        }

        .obsidian-description {
            color: #D9E1E7;
            font-size: 1rem;
            margin-bottom: 0.9rem;
        }

        .obsidian-tag {
            display: inline-block;
            padding: 0.32rem 0.72rem;
            margin: 0.2rem 0.4rem 0 0;
            border-radius: 999px;
            background-color: #1D3342;
            border: 1px solid #314A5B;
            color: #F5F1E8;
            font-size: 0.9rem;
            font-weight: 600;
        }

        div[data-testid="stMetric"] {
            background-color: #162A36;
            border: 1px solid #2B4252;
            padding: 1rem;
            border-radius: 14px;
        }

        div.stButton > button {
            background-color: #E6D8C2;
            color: #10202B;
            border: none;
            border-radius: 12px;
            font-weight: 700;
            height: 2.9rem;
        }

        div.stButton > button:hover {
            background-color: #D9CAB1;
            color: #10202B;
        }

        div[data-baseweb="tab-list"] {
            gap: 0.5rem;
        }

        div[data-baseweb="tab"] {
            border-radius: 10px 10px 0 0;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .obsidian-section-note {
            color: #CFC6B8;
            font-size: 0.95rem;
            margin-bottom: 0.3rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def preview_caption(total_rows: int, preview_limit: int) -> str:
    """
    Returns a smart preview caption based on the number of available rows.
    """
    shown_rows = min(total_rows, preview_limit)

    if total_rows > preview_limit:
        return f"Showing first {shown_rows} rows only out of {total_rows} total rows."

    return f"Showing all {shown_rows} rows."


st.set_page_config(page_title="OBSIDIAN Arabic Tweet Classifier", layout="wide")
apply_custom_branding()

with st.container():
    st.markdown('<div class="obsidian-hero">', unsafe_allow_html=True)

    hero_col1, hero_col2 = st.columns([0.45, 5.55], gap="small", vertical_alignment="center")

    with hero_col1:
        st.image("assets/Crystal.png", width=140)

    with hero_col2:
        st.markdown('<div class="obsidian-title">OBSIDIAN</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="obsidian-subtitle">Real-time social media intelligence and threat detection system</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="obsidian-description">Arabic tweet and short-text classification powered by a fine-tuned AraBERT model.</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            """
            <span class="obsidian-tag">Threat</span>
            <span class="obsidian-tag">Violence</span>
            <span class="obsidian-tag">Distress</span>
            <span class="obsidian-tag">Complaint</span>
            <span class="obsidian-tag">Neutral</span>
            """,
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("")

exp_col1, exp_col2 = st.columns(2)

with exp_col1:
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

            **Live Monitor**
            - Use demo mode to simulate real-time Arabic tweet monitoring
            - Or connect an n8n webhook if available
            - Fetch tweets, classify them, review alerts, and download results
            """
        )

with exp_col2:
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
    st.warning(
        "Model could not be loaded. Please check the Hugging Face model repo connection or your internet connection."
    )
    st.caption(str(e))


tab1, tab2, tab3 = st.tabs(["Single Text", "Batch Upload", "Live Monitor"])


with tab1:
    st.subheader("Single Text Classification")
    st.markdown(
        '<div class="obsidian-section-note">Enter one Arabic text or tweet, then click Predict.</div>',
        unsafe_allow_html=True
    )

    user_text = st.text_area("Arabic text", height=150, placeholder="Enter Arabic text here...")

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
    st.markdown(
        '<div class="obsidian-section-note">Upload a CSV or XLSX file, choose the text column, and run batch prediction.</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader("Upload CSV or XLSX file", type=["csv", "xlsx"])
    st.caption("You can test the app using the sample file provided in the repository: data_samples/sample_test.csv")

    if uploaded_file is not None:
        try:
            df = load_uploaded_file(uploaded_file)

            total_rows = len(df)
            total_cols = len(df.columns)

            st.write(f"Uploaded file contains **{total_rows} rows** and **{total_cols} columns**.")

            uploaded_preview_limit = 10
            st.write("### Uploaded Data Preview")
            st.caption(preview_caption(total_rows, uploaded_preview_limit))
            st.dataframe(df.head(uploaded_preview_limit), width="stretch")

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

                selected_text_preview_limit = 5
                preview_df = df[[selected_text_col]].head(selected_text_preview_limit).copy()
                st.write("### Selected Text Column Preview")
                st.caption(preview_caption(total_rows, selected_text_preview_limit))
                st.dataframe(preview_df, width="stretch")

                if st.button("Run Batch Prediction", width="stretch"):
                    if not model_loaded:
                        st.error("Model is not loaded.")
                    else:
                        result_df = run_batch_inference(df, tokenizer, model, selected_text_col)

                        st.success(
                            f"Predictions completed successfully using column: {selected_text_col}"
                        )

                        results_preview_limit = 20
                        display_cols = [selected_text_col, "predicted_label", "confidence_percent"]
                        st.write("### Classification Results Preview")
                        st.caption(
                            f"{preview_caption(len(result_df), results_preview_limit)} "
                            "Download the CSV for the full output."
                        )
                        st.dataframe(result_df[display_cols].head(results_preview_limit), width="stretch")

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


with tab3:
    st.subheader("Live Tweet Monitor")
    st.markdown(
        '<div class="obsidian-section-note">Simulate or fetch near real-time Arabic tweets, classify them, and generate alert-level insights.</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Demo mode is stable for presentations. n8n mode uses Abdullah's live workflow "
        "with postLimit, timeWindowHours, and xQuery. If the live workflow fails, retry once "
        "or reduce the tweet limit."
    )

    source_mode = st.radio(
        "Select live data source",
        options=["Demo Simulation", "n8n Webhook"],
        horizontal=True,
    )

    col_a, col_b = st.columns(2)

    with col_a:
        tweet_limit = st.slider(
            "Number of tweets to process",
            min_value=5,
            max_value=100,
            value=25,
            step=5,
        )

    with col_b:
        min_confidence = st.slider(
            "Minimum confidence to display",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
        )

    webhook_url = ""
    time_window_hours = 1
    query = "place_country:SA lang:ar"

    if source_mode == "n8n Webhook":
        st.write("### n8n Fetch Settings")

        n8n_col1, n8n_col2 = st.columns(2)

        with n8n_col1:
            time_window_hours = st.number_input(
                "Time window in hours",
                min_value=1,
                max_value=24,
                value=1,
                step=1,
            )

        with n8n_col2:
            query = st.text_input(
                "X/Twitter query",
                value="place_country:SA lang:ar",
            )

        secret_webhook_url = st.secrets.get("N8N_WEBHOOK_URL", "")

        if secret_webhook_url:
            st.success("n8n webhook URL loaded from Streamlit secrets.")
            use_secret_webhook = st.checkbox(
                "Use webhook URL from Streamlit secrets",
                value=True,
            )

            if use_secret_webhook:
                webhook_url = secret_webhook_url
            else:
                webhook_url = st.text_input(
                    "n8n webhook URL",
                    type="password",
                    placeholder="Paste another n8n webhook URL here..."
                )
        else:
            st.warning("No n8n webhook URL found in Streamlit secrets.")
            webhook_url = st.text_input(
                "n8n webhook URL",
                type="password",
                placeholder="Paste the n8n webhook URL here..."
            )

        st.caption(
            "For deployment, store the webhook URL in Streamlit secrets as "
            "`N8N_WEBHOOK_URL`. Avoid hardcoding private URLs in GitHub."
        )

    if st.button("Fetch and Classify Live Tweets", width="stretch"):
        if not model_loaded:
            st.error("Model is not loaded.")
        else:
            try:
                with st.spinner("Fetching tweets..."):
                    if source_mode == "Demo Simulation":
                        live_df = load_demo_live_tweets(limit=tweet_limit)
                    else:
                        if not webhook_url.strip():
                            st.error("Please enter the n8n webhook URL first.")
                            st.stop()

                        live_df = fetch_live_tweets_from_n8n(
                            webhook_url=webhook_url.strip(),
                            limit=tweet_limit,
                            time_window_hours=int(time_window_hours),
                            query=query.strip() or "place_country:SA lang:ar",
                            timeout=180,
                            max_retries=2,
                            retry_delay=3,
                        )

                if live_df.empty:
                    st.warning("No tweets were returned from the selected source.")
                    st.stop()

                fetched_preview_limit = 20
                st.write("### Fetched Tweets Preview")
                st.caption(
                    f"{preview_caption(len(live_df), fetched_preview_limit)} "
                    "These are the raw fetched tweets before classification."
                )

                preview_df = live_df.head(fetched_preview_limit).rename(
                    columns={"username": "author_id"}
                )
                st.dataframe(preview_df, width="stretch")

                with st.spinner("Classifying live tweets..."):
                    result_df = classify_live_tweets(
                        live_df,
                        tokenizer,
                        model,
                        text_col="text",
                        batch_size=32,
                    )

                if min_confidence > 0:
                    result_df = result_df[result_df["confidence"] >= min_confidence].copy()

                if result_df.empty:
                    st.warning("No tweets matched the selected confidence threshold.")
                    st.stop()

                summary = get_live_summary(result_df)

                st.success("Live tweet classification completed successfully.")

                metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)

                with metric_col1:
                    st.metric("Total Tweets", summary["total_tweets"])

                with metric_col2:
                    st.metric("High Alerts", summary["high_alerts"])

                with metric_col3:
                    st.metric("Medium Alerts", summary["medium_alerts"])

                with metric_col4:
                    st.metric("Avg Confidence", f"{summary['average_confidence']:.2%}")

                with metric_col5:
                    st.metric("Dominant Label", summary["dominant_label"])

                st.write("### High & Medium Alert Tweets")

                display_cols = [
                    "tweet_id",
                    "timestamp",
                    "username",
                    "text",
                    "predicted_label",
                    "confidence_percent",
                    "alert_level",
                    "source",
                ]

                available_display_cols = [
                    col for col in display_cols if col in result_df.columns
                ]

                high_risk_df = result_df[
                    result_df["alert_level"].isin(["High Alert", "Medium Alert"])
                ].copy()

                if not high_risk_df.empty:
                    st.warning("High or medium alert tweets detected.")

                    high_risk_display_df = high_risk_df[available_display_cols].rename(
                        columns={"username": "author_id"}
                    )

                    st.dataframe(
                        high_risk_display_df,
                        width="stretch",
                    )
                else:
                    st.success("No high or medium alert tweets detected.")

                counts = result_df["predicted_label"].value_counts().reset_index()
                counts.columns = ["Label", "Count"]

                st.write("### Live Label Distribution")
                fig = px.pie(
                    counts,
                    names="Label",
                    values="Count",
                    title="Distribution of Predicted Labels in Live Monitor",
                )
                st.plotly_chart(fig, width="stretch")

                alert_counts = result_df["alert_level"].value_counts().reset_index()
                alert_counts.columns = ["Alert Level", "Count"]

                st.write("### Alert Level Distribution")
                alert_fig = px.bar(
                    alert_counts,
                    x="Alert Level",
                    y="Count",
                    title="Alert Levels Detected",
                )
                st.plotly_chart(alert_fig, width="stretch")

                st.write("### All Classified Live Tweets")
                st.caption("Showing all processed tweets after filtering by confidence threshold.")

                display_result_df = result_df[available_display_cols].rename(
                    columns={"username": "author_id"}
                )

                st.dataframe(
                    display_result_df,
                    width="stretch",
                )

                export_df = result_df.rename(columns={"username": "author_id"})

                csv_data = export_df.to_csv(index=False).encode("utf-8-sig")
                excel_data = dataframe_to_excel_bytes(export_df)

                download_col1, download_col2 = st.columns(2)

                with download_col1:
                    st.download_button(
                        "Download Live Results as CSV",
                        data=csv_data,
                        file_name="obsidian_live_results.csv",
                        mime="text/csv",
                        width="stretch",
                    )

                with download_col2:
                    st.download_button(
                        "Download Live Results as Excel",
                        data=excel_data,
                        file_name="obsidian_live_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        width="stretch",
                    )

            except Exception as e:
                st.error(f"Error while running live monitor: {str(e)}")