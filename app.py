import streamlit as st
import preprocessor
import helper
import matplotlib.pyplot as plt
import base64
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


# -------------------------------
# Background Setup
# -------------------------------
def set_background(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()

    bg_style = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(bg_style, unsafe_allow_html=True)

set_background("bg.jpeg")


# Sidebar styling
st.markdown("""
<style>
section[data-testid="stSidebar"] {
    background-color: #1C4136;
}
section[data-testid="stSidebar"] * {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)


# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_mentalbert():
    tokenizer = AutoTokenizer.from_pretrained("mentalbert_model")
    model = AutoModelForSequenceClassification.from_pretrained("mentalbert_model")
    return tokenizer, model

tokenizer, model = load_mentalbert()


# -------------------------------
# Prediction Function (CLEAN)
# -------------------------------
def predict_mental_state(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    pred = torch.argmax(logits, dim=1).item()

    return pred


# -------------------------------
# UI
# -------------------------------
st.sidebar.title("WhatsApp Chat Analyzer")
uploaded_file = st.sidebar.file_uploader("Choose a file")

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")

    df = preprocessor.preprocess(data)

    # User selection
    user_list = df["users"].unique().tolist()
    user_list.remove("group_notification")
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

    if st.sidebar.button("Show Analysis"):

        # -------------------------------
        # Stats
        # -------------------------------
        num_msgs, num_words, num_media, num_links = helper.fetch_stats(selected_user, df)

        st.title("Message Statistics")

        col1, col2, col3, col4 = st.columns(4)

        col1.subheader("Total Messages")
        col1.title(num_msgs)

        col2.subheader("Total Words")
        col2.title(num_words)

        col3.subheader("Total Media")
        col3.title(num_media)

        col4.subheader("Links Shared")
        col4.title(num_links)

        # -------------------------------
        # Timeline
        # -------------------------------
        st.title("Timeline")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Monthly Timeline")
            timeline = helper.monthly_timeline(selected_user, df)
            fig, ax = plt.subplots()
            ax.plot(timeline["time"], timeline["msgs"])
            plt.xticks(rotation=90)
            st.pyplot(fig)

        with col2:
            st.subheader("Daily Timeline")
            daily_timeline = helper.daily_timeline(selected_user, df)
            fig, ax = plt.subplots()
            ax.plot(daily_timeline["only_date"], daily_timeline["msgs"])
            plt.xticks(rotation=90)
            st.pyplot(fig)

        # -------------------------------
        # Active Users
        # -------------------------------
        if selected_user == "Overall":
            st.title("Most Active Users")

            x, new_df = helper.fetch_top_users(df)

            col1, col2 = st.columns(2)

            col1.dataframe(new_df)

            fig, ax = plt.subplots()
            ax.bar(x.index, x.values)
            plt.xticks(rotation=90)
            col2.pyplot(fig)

        # -------------------------------
        # Word Analysis
        # -------------------------------
        st.title("Word Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("WordCloud")
            wc = helper.wordcloud(selected_user, df)
            if wc is not None:
                st.image(wc.to_array())
            else:
                st.write("No words available.")

        with col2:
            st.subheader("Most Common Words")
            common_df = helper.most_common_words(selected_user, df)

            if not common_df.empty:
                fig, ax = plt.subplots()
                ax.barh(common_df[0], common_df[1])
                st.pyplot(fig)
            else:
                st.write("No words found.")

        # -------------------------------
        # Emoji Analysis
        # -------------------------------
        st.title("Emoji Analysis")

        emoji_df = helper.emoji_helper(selected_user, df)

        col1, col2 = st.columns(2)

        col1.dataframe(emoji_df)

        if not emoji_df.empty:
            fig, ax = plt.subplots()
            ax.pie(emoji_df["Count"].head(), labels=emoji_df["Emoji"].head(), autopct="%0.2f%%")
            col2.pyplot(fig)
        else:
            col2.write("No emojis found.")

        # -------------------------------
        # Mental Health Analysis
        # -------------------------------
        st.title("Mental Health Analysis (MentalBERT)")

        # =========================
# Mental Health Analysis
# =========================


# ✅ get processed dataframe
        temp_df = helper.get_mental_df(
            selected_user, df, tokenizer, model, predict_mental_state
)

# -------------------------
# Pie Chart
# -------------------------
        st.subheader("Mental State Distribution")

        mental_counts = helper.get_mental_distribution(temp_df)

        if not mental_counts.empty:
            fig, ax = plt.subplots()
            ax.pie(mental_counts, labels=mental_counts.index, autopct="%0.2f%%")
            st.pyplot(fig)
        else:
            st.write("No predictions available.")

# -------------------------
# Table
# -------------------------
        st.subheader("User-wise Mental State")

        user_mental = helper.get_user_mental_table(temp_df, selected_user)

        st.dataframe(user_mental)