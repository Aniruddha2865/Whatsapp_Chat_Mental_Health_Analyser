# WhatsApp Chat Analyzer + Mental Health Detection

An advanced WhatsApp Chat Analysis tool built using **Python, Streamlit, and NLP**, enhanced with a fine-tuned **BERT model (MentalBERT)** to detect mental health patterns from conversations.

---

## 🚀 Features

### 📊 Chat Analytics
- Total Messages, Words, Media, and Links
- Most Active Users
- Monthly & Daily Timeline Analysis
- WordCloud (with Hinglish stopword filtering)
- Most Common Words
- Emoji Analysis with distribution charts

### 🧠 Mental Health Analysis (MentalBERT)
Classifies messages into:
- Anxiety  
- Depression  
- Mental Disorder  
- Normal  
- Suicide Watch  

Includes:
- Overall mental state distribution  
- User-wise mental health breakdown  

---

## 🏗️ Project Structure
```
project/
│
├── app.py                  # Main Streamlit application
├── helper.py               # Chat analysis functions
├── preprocessor.py         # Chat parsing and preprocessing
├── stop_hinglish.txt       # Custom stopword list
├── bg.jpeg                 # Background image
├── mentalbert_model/       # Saved fine-tuned model
└── README.md
```

---

## ⚙️ How It Works

### 1. Chat Preprocessing
- Parses exported WhatsApp chat using regex
- Extracts user, message, and timestamps

### 2. Statistical Analysis
- Computes message stats, timelines, and word frequencies

### 3. Mental Health Prediction
- Uses a fine-tuned BERT model to classify each message
- Outputs mental health labels

---

## 🧪 Tech Stack

- **Frontend:** Streamlit  
- **Backend:** Python  
- **Model:** HuggingFace Transformers (BERT)  

### Libraries Used:
- pandas  
- matplotlib  
- wordcloud  
- emoji  
- torch  
- transformers  
- urlextract  

---

## 📦 Installation

```bash
git clone https://github.com/your-username/whatsapp-chat-analyzer.git
cd whatsapp-chat-analyzer
pip install -r requirements.txt

streamlit run app.py
```
The model results file is not uploaded in the repo due to size limitations.
[GDrive Link](https://drive.google.com/drive/folders/1B0RPXlIf0qHYor_gJWeIvD4bz7OfNy4F?usp=sharing)

Download everything , and rename the folder "mentalbert_results" to "mentalbert_model". Keep everything under one directory.
