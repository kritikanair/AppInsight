import streamlit as st
import pandas as pd

from appinsight import (
    prepare_reviews, train_sentiment_model, predict_reviews,
    aspect_summary, build_review_summary, word_frequency,
    preprocess_text, extract_aspects, FINANCIAL_APPS
)

st.set_page_config(page_title="AppInsight", page_icon="📱", layout="wide")
st.title("📱 AppInsight")
st.caption("Aspect-based NLP analysis of financial and mobile banking app reviews")

# If the dataset is placed beside app.py, load it automatically. Otherwise allow one upload.
local_dataset = "googleplaystore_user_reviews.csv"
uploaded = None
try:
    import os
    dataset_exists_locally = os.path.exists(local_dataset)
except Exception:
    dataset_exists_locally = False

if dataset_exists_locally:
    dataset_source = local_dataset
else:
    uploaded = st.file_uploader("Upload review dataset (CSV)", type=["csv"])
    if uploaded is None:
        st.stop()
    dataset_source = uploaded

try:
    raw_df = pd.read_csv(dataset_source)
    df = prepare_reviews(raw_df, finance_only=True)
except Exception as exc:
    st.error(f"Could not prepare the review dataset: {exc}")
    st.stop()

st.success(f"Loaded {len(df):,} usable reviews from {df['App'].nunique():,} matching financial apps.")
with st.expander("Financial apps included"):
    st.write(", ".join(sorted(df["App"].dropna().astype(str).unique())))
with st.expander("Preview filtered reviews"):
    st.dataframe(df[["Review", "App", "Sentiment", "Category"]].head(10), use_container_width=True)

if st.button("Train sentiment model", type="primary"):
    with st.spinner("Training TF-IDF + Logistic Regression..."):
        try:
            model, metrics = train_sentiment_model(df)
            st.session_state["appinsight_model"] = model
            st.session_state["appinsight_metrics"] = metrics
            st.session_state["appinsight_predictions"] = predict_reviews(model, df["Review"].tolist())
            st.success("Model trained successfully.")
        except Exception as exc:
            st.error(str(exc))

if "appinsight_model" in st.session_state:
    metrics = st.session_state["appinsight_metrics"]
    cols = st.columns(4)
    for col, key in zip(cols, ["Accuracy", "Precision (macro)", "Recall (macro)", "F1-score (macro)"]):
        col.metric(key, f"{metrics[key]:.3f}")
    df["Predicted Sentiment"] = st.session_state["appinsight_predictions"]
else:
    df["Predicted Sentiment"] = df["Sentiment"].where(
        df["Sentiment"].isin(["Positive", "Negative", "Neutral"]), "Unclassified"
    )

st.divider()
left, right = st.columns(2)
with left:
    st.subheader("Sentiment distribution")
    st.bar_chart(df["Predicted Sentiment"].value_counts())
with right:
    st.subheader("Common words")
    freq = word_frequency(df["Review"].tolist(), 12)
    st.dataframe(freq, hide_index=True, use_container_width=True)

st.subheader("Aspect / issue analysis")
aspects = aspect_summary(df)
st.dataframe(aspects, hide_index=True, use_container_width=True)

st.subheader("Review analyzer")
review_input = st.text_area("Enter a review", "The app is slow and my payment is not going through.")
if st.button("Analyze review"):
    st.write("**Cleaned text:**", preprocess_text(review_input))
    st.write("**Detected aspect(s):**", ", ".join(extract_aspects(review_input)))
    if "appinsight_model" in st.session_state:
        st.write("**Predicted sentiment:**", predict_reviews(st.session_state["appinsight_model"], [review_input])[0])

st.divider()
st.subheader("Overall feedback summary")
summary_count = st.slider(
    "Number of reviews to include", min_value=10, max_value=max(10, min(500, len(df))),
    value=min(100, max(10, len(df))), step=10
)
if st.button("Generate summary", type="primary"):
    subset = df.head(summary_count).copy()
    summary = build_review_summary(subset)
    st.markdown("**Summary**")
    st.write(summary["overview"])
    if summary["aspects"]:
        st.markdown("**Most-mentioned topics**")
        for aspect, count in summary["aspects"]:
            if aspect != "General feedback":
                st.write(f"- **{aspect}:** {count} reviews")
    if summary["examples"]:
        st.markdown("**Representative review examples**")
        for example in summary["examples"]:
            st.write(f"- {example}")

st.divider()
st.download_button(
    "Export analyzed reviews (CSV)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="appinsight_financial_reviews.csv",
    mime="text/csv"
)
