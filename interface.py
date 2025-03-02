import streamlit as st
import joblib
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

# Load các mô hình đã lưu
model_dir = "saved_models"

# Load TF-IDF vectorizer & Logistic Regression model
tfidf_vectorizer = joblib.load(f"{model_dir}/TF-IDF_vectorizer.pkl")
logistic_model = joblib.load(f"{model_dir}/TF-IDF_Logistic Regression.pkl")

# Load Word2Vec Skip-gram & MLP Classifier
w2v_sg = Word2Vec.load(f"{model_dir}/word2vec_skipgram.model")
mlp_model = joblib.load(f"{model_dir}/Word2Vec Skip-gram_MLP Classifier.pkl")

# Mapping nhãn cảm xúc từ số → tiếng Việt
sentiment_mapping = {
    0: "😡 Tiêu cực",
    1: "😐 Trung lập",
    2: "😊 Tích cực"
}

# Hàm chuyển câu thành vector Word2Vec Skip-gram
def sentence_vector(sentence, model):
    words = sentence.split()
    vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# Hàm dự đoán sentiment cho nhiều câu
def predict_sentiment(input_texts):
    predictions = []
    
    for text in input_texts:
        # TF-IDF + Logistic Regression
        tfidf_features = tfidf_vectorizer.transform([text])
        tfidf_prediction = logistic_model.predict(tfidf_features)[0]

        # Word2Vec Skip-gram + MLP Classifier
        w2v_features = sentence_vector(text, w2v_sg).reshape(1, -1)
        w2v_prediction = mlp_model.predict(w2v_features)[0]

        predictions.append({
            "Câu": text,
            "TF-IDF + Logistic Regression": sentiment_mapping[tfidf_prediction],
            "Word2Vec Skip-gram + MLP": sentiment_mapping[w2v_prediction]
        })
    
    return pd.DataFrame(predictions)  # Trả về DataFrame để hiển thị đẹp hơn

# ------------------------ Streamlit UI ------------------------
st.title("🔮 Dự đoán cảm xúc với TF-IDF & Word2Vec")

# Input từ người dùng (có thể nhập nhiều câu)
user_input = st.text_area("Nhập các câu cần phân tích (mỗi dòng là một câu):")

if st.button("Dự đoán"):
    if user_input.strip():
        input_sentences = user_input.split("\n")  # Chia nhiều câu theo dòng
        predictions_df = predict_sentiment(input_sentences)

        # Hiển thị kết quả dạng bảng
        st.subheader("📊 Kết quả dự đoán:")
        st.dataframe(predictions_df, use_container_width=True)
    else:
        st.warning("⚠ Vui lòng nhập ít nhất một câu!")
