import time
import pandas as pd
from io import BytesIO
from typing import Dict, Any

from src.preprocess import clean_text
from src.inference import predict_batch_texts
from src.labels import LABELS


DEFAULT_DEMO_TWEETS = [
    {
        "timestamp": "2026-04-25 12:00:00 KSA",
        "timestamp_utc": "2026-04-25T09:00:00.000Z",
        "username": "user_01",
        "text": "الخدمة سيئة جدًا والتطبيق يتعطل كل مرة",
        "source": "demo",
    },
    {
        "timestamp": "2026-04-25 12:01:00 KSA",
        "timestamp_utc": "2026-04-25T09:01:00.000Z",
        "username": "user_02",
        "text": "أنا خائف جدًا ولا أعرف ماذا أفعل، أحتاج مساعدة الآن",
        "source": "demo",
    },
    {
        "timestamp": "2026-04-25 12:02:00 KSA",
        "timestamp_utc": "2026-04-25T09:02:00.000Z",
        "username": "user_03",
        "text": "سأؤذيك إذا اقتربت مني مرة أخرى",
        "source": "demo",
    },
    {
        "timestamp": "2026-04-25 12:03:00 KSA",
        "timestamp_utc": "2026-04-25T09:03:00.000Z",
        "username": "user_04",
        "text": "حدث شجار عنيف في الشارع والناس يصرخون",
        "source": "demo",
    },
    {
        "timestamp": "2026-04-25 12:04:00 KSA",
        "timestamp_utc": "2026-04-25T09:04:00.000Z",
        "username": "user_05",
        "text": "اليوم الجو جميل والناس مستمتعة بالإجازة",
        "source": "demo",
    },
    {
        "timestamp": "2026-04-25 12:05:00 KSA",
        "timestamp_utc": "2026-04-25T09:05:00.000Z",
        "username": "user_06",
        "text": "تأخر الرد من الدعم الفني وهذا أمر مزعج جدًا",
        "source": "demo",
    },
    {
        "timestamp": "2026-04-25 12:06:00 KSA",
        "timestamp_utc": "2026-04-25T09:06:00.000Z",
        "username": "user_07",
        "text": "أشعر بضيق شديد ولا أستطيع النوم منذ أيام",
        "source": "demo",
    },
    {
        "timestamp": "2026-04-25 12:07:00 KSA",
        "timestamp_utc": "2026-04-25T09:07:00.000Z",
        "username": "user_08",
        "text": "سأكسر كل شيء إذا لم يتم حل المشكلة اليوم",
        "source": "demo",
    },
    {
        "timestamp": "2026-04-25 12:08:00 KSA",
        "timestamp_utc": "2026-04-25T09:08:00.000Z",
        "username": "user_09",
        "text": "الفعالية كانت ممتازة والتنظيم رائع",
        "source": "demo",
    },
    {
        "timestamp": "2026-04-25 12:09:00 KSA",
        "timestamp_utc": "2026-04-25T09:09:00.000Z",
        "username": "user_10",
        "text": "أين المسؤولون عن هذه المشكلة؟ الوضع غير مقبول",
        "source": "demo",
    },
]


def format_timestamp_to_ksa(timestamp: Any) -> str:
    """
    Converts raw UTC/ISO timestamps such as 2026-04-25T14:10:33.000Z
    into a cleaner KSA dashboard format: 2026-04-25 17:10:33 KSA.

    If parsing fails, returns the original timestamp as a string.
    """
    if timestamp is None or str(timestamp).strip() == "":
        return ""

    raw_timestamp = str(timestamp).strip()

    try:
        parsed = pd.to_datetime(raw_timestamp, utc=True, errors="coerce")

        if pd.isna(parsed):
            return raw_timestamp

        ksa_time = parsed.tz_convert("Asia/Riyadh")
        return ksa_time.strftime("%Y-%m-%d %H:%M:%S KSA")

    except Exception:
        return raw_timestamp


def load_demo_live_tweets(limit: int = 10) -> pd.DataFrame:
    """
    Loads demo tweets to simulate real-time monitoring when no live source is available.
    """
    df = pd.DataFrame(DEFAULT_DEMO_TWEETS)
    return df.head(limit).copy()


def normalize_live_tweets(raw_items: Any) -> pd.DataFrame:
    """
    Converts different possible n8n/API response formats into a clean DataFrame.

    Expected output columns:
    tweet_id, timestamp, timestamp_utc, username, text, source
    """
    if raw_items is None:
        return pd.DataFrame(columns=["tweet_id", "timestamp", "timestamp_utc", "username", "text", "source"])

    if isinstance(raw_items, dict):
        if "tweets" in raw_items:
            raw_items = raw_items["tweets"]
        elif "data" in raw_items:
            raw_items = raw_items["data"]
        elif "items" in raw_items:
            raw_items = raw_items["items"]
        else:
            raw_items = [raw_items]

    if not isinstance(raw_items, list):
        raw_items = [raw_items]

    rows = []

    for item in raw_items:
        if not isinstance(item, dict):
            continue

        text = (
            item.get("cleanText")
            or item.get("cleaned_text")
            or item.get("text")
            or item.get("tweet")
            or item.get("content")
            or item.get("full_text")
            or ""
        )

        author_data = item.get("author") or item.get("user") or {}

        if isinstance(author_data, dict):
            nested_username = (
                author_data.get("username")
                or author_data.get("userName")
                or author_data.get("screen_name")
                or author_data.get("handle")
                or author_data.get("name")
                or author_data.get("id")
            )
        else:
            nested_username = author_data

        username = (
            item.get("username")
            or item.get("userName")
            or item.get("screen_name")
            or item.get("handle")
            or item.get("authorUsername")
            or item.get("author_name")
            or item.get("authorName")
            or item.get("authorId")
            or nested_username
            or "unknown"
        )

        raw_timestamp = (
            item.get("timestamp")
            or item.get("created_at")
            or item.get("createdAt")
            or item.get("created_at_iso")
            or item.get("date")
            or item.get("time")
            or item.get("publishedAt")
            or item.get("published_at")
            or ""
        )

        tweet_id = (
            item.get("tweet_id")
            or item.get("tweetId")
            or item.get("id")
            or item.get("url")
            or ""
        )

        source = item.get("source") or "n8n"

        rows.append(
            {
                "tweet_id": tweet_id,
                "timestamp": format_timestamp_to_ksa(raw_timestamp),
                "timestamp_utc": raw_timestamp,
                "username": username,
                "text": text,
                "source": source,
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        return pd.DataFrame(columns=["tweet_id", "timestamp", "timestamp_utc", "username", "text", "source"])

    df["text"] = df["text"].fillna("").astype(str).apply(clean_text)
    df = df[df["text"].str.strip().str.len() > 0].reset_index(drop=True)

    return df


def fetch_live_tweets_from_n8n(
    webhook_url: str,
    limit: int = 50,
    time_window_hours: int = 1,
    query: str = "place_country:SA lang:ar",
    timeout: int = 180,
    max_retries: int = 2,
    retry_delay: int = 3,
) -> pd.DataFrame:
    """
    Fetches live/simulated tweets from the n8n webhook using Abdullah's
    Colab Block 17 parameters.

    Adds retry logic because live webhook calls can be intermittent.
    """
    if not webhook_url:
        raise ValueError("Webhook URL is required.")

    import requests

    params = {
        "postLimit": limit,
        "timeWindowHours": time_window_hours,
        "xQuery": query,
    }

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(
                webhook_url,
                params=params,
                timeout=timeout,
            )
            response.raise_for_status()

            raw_data = response.json()
            df = normalize_live_tweets(raw_data)

            if limit:
                df = df.head(limit).copy()

            return df

        except requests.exceptions.Timeout as e:
            last_error = e

        except requests.exceptions.ConnectionError as e:
            last_error = e

        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"n8n returned an HTTP error: {e}")

        except requests.exceptions.RequestException as e:
            last_error = e

        if attempt < max_retries:
            time.sleep(retry_delay)

    raise RuntimeError(
        "Could not connect to n8n after retrying. "
        "The workflow may be sleeping, busy, rate-limited, or temporarily unavailable. "
        "Try again, reduce the tweet limit, or use Demo Simulation."
    ) from last_error


def add_alert_levels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds alert_level based on predicted label and confidence.
    """
    result_df = df.copy()

    def get_alert_level(row):
        label = row.get("predicted_label")
        confidence = float(row.get("confidence", 0))

        if label in ["Threat", "Violence"] and confidence >= 0.80:
            return "High Alert"

        if label == "Distress" and confidence >= 0.75:
            return "Medium Alert"

        if label == "Complaint" and confidence >= 0.85:
            return "Medium Alert"

        return "Normal"

    result_df["alert_level"] = result_df.apply(get_alert_level, axis=1)

    return result_df


def classify_live_tweets(
    df: pd.DataFrame,
    tokenizer,
    model,
    text_col: str = "text",
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Classifies live/demo tweets using the batch prediction function.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    if text_col not in df.columns:
        raise ValueError(f"Text column '{text_col}' does not exist.")

    result_df = df.copy()
    result_df[text_col] = result_df[text_col].fillna("").astype(str).apply(clean_text)

    predictions = predict_batch_texts(
        result_df[text_col].tolist(),
        tokenizer,
        model,
        batch_size=batch_size,
    )

    result_df["predicted_label"] = [pred["label"] for pred in predictions]
    result_df["confidence"] = [pred["confidence"] for pred in predictions]
    result_df["confidence_percent"] = result_df["confidence"].apply(lambda x: f"{x:.2%}")

    for label in LABELS:
        result_df[f"score_{label}"] = [
            pred["probabilities"].get(label, 0.0) for pred in predictions
        ]

    result_df = add_alert_levels(result_df)

    return result_df


def get_live_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Returns dashboard-friendly summary metrics.
    """
    if df is None or df.empty:
        return {
            "total_tweets": 0,
            "high_alerts": 0,
            "medium_alerts": 0,
            "average_confidence": 0.0,
            "dominant_label": "N/A",
        }

    label_counts = df["predicted_label"].value_counts()

    return {
        "total_tweets": len(df),
        "high_alerts": int((df["alert_level"] == "High Alert").sum()),
        "medium_alerts": int((df["alert_level"] == "Medium Alert").sum()),
        "average_confidence": float(df["confidence"].mean()),
        "dominant_label": label_counts.idxmax() if not label_counts.empty else "N/A",
    }


def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    """
    Converts a DataFrame to Excel bytes for Streamlit download_button.
    """
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="live_results")

    return output.getvalue()