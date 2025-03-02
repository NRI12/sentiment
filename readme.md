# README - Hệ thống phân tích cảm xúc văn bản tiếng Việt sử dụng máy học đơn giản 

## 1. Giới thiệu
Hệ thống này được phát triển nhằm phân tích cảm xúc trong phản hồi của sinh viên tiếng Việt. Hệ thống sử dụng các phương pháp xử lý ngôn ngữ tự nhiên (NLP) và các mô hình học máy để tự động phân loại phản hồi thành ba nhãn cảm xúc chính:
- 😡 **Tiêu cực (Negative)**
- 😐 **Trung lập (Neutral)**
- 😊 **Tích cực (Positive)**

---

## 2. Dữ liệu sử dụng

Hệ thống sử dụng hai bộ dữ liệu chính:

### 2.1. UIT-VSFC - Vietnamese Students’ Feedback Corpus (https://nlp.uit.edu.vn/datasets)
UIT-VSFC là bộ dữ liệu phản hồi của sinh viên tiếng Việt được thu thập từ các nguồn thực tế. Dữ liệu này gồm hàng nghìn câu phản hồi, được gán nhãn thành ba loại cảm xúc và phân loại theo bốn chủ đề:
- **Lecturer** (Giảng viên)
- **Curriculum** (Chương trình giảng dạy)
- **Facility** (Cơ sở vật chất)
- **Others** (Khác)

UIT-VSFC là tập dữ liệu thực tế có giá trị cao, giúp đảm bảo độ chính xác của mô hình khi ứng dụng vào môi trường thực tế.

### 2.2. The Synthetic Vietnamese Students’ Feedback Corpus (https://www.kaggle.com/datasets/toreleon/synthetic-vietnamese-students-feedback-corpus?resource=download)
Bộ dữ liệu tổng hợp này được tạo ra bằng **ChatGPT API** dựa trên hướng dẫn từ UIT-VSFC. Bộ dữ liệu này gồm hơn **10.000 câu phản hồi**, có nhãn cảm xúc tương tự UIT-VSFC. Việc sử dụng dữ liệu tổng hợp giúp:
- Mở rộng tập dữ liệu để huấn luyện mô hình tốt hơn
- Kiểm soát nội dung phản hồi một cách chính xác
- Giúp mô hình nhận diện được nhiều cách biểu đạt cảm xúc khác nhau

Tuy nhiên, do dữ liệu này được tổng hợp từ AI, nên cần kiểm tra kỹ để đảm bảo không làm sai lệch mô hình.

---

## 3. Phương pháp trích xuất đặc trưng và thuật toán sử dụng

Hệ thống áp dụng **ba phương pháp chính để trích xuất đặc trưng** từ văn bản:
- **TF-IDF (Term Frequency - Inverse Document Frequency)**: Biểu diễn văn bản dưới dạng ma trận trọng số dựa trên tần suất xuất hiện của từ trong tập dữ liệu.
- **Word2Vec CBOW (Continuous Bag of Words)**: Dự đoán từ dựa trên ngữ cảnh xung quanh để tạo vector biểu diễn từ.
- **Word2Vec Skip-gram**: Dự đoán ngữ cảnh xung quanh dựa trên từ hiện tại, giúp mô hình học được các mối quan hệ ngữ nghĩa giữa các từ.

Sau khi trích xuất đặc trưng, các mô hình học máy được huấn luyện bao gồm:
- **Hồi quy Logistic**: Mô hình tuyến tính đơn giản nhưng hiệu quả với TF-IDF.
- **Mạng nơ-ron nhân tạo (MLP Classifier)**: Phù hợp với dữ liệu biểu diễn bằng Word2Vec.
- **Rừng ngẫu nhiên (Random Forest)**: Giúp giảm overfitting và xử lý dữ liệu hiệu quả.
- **Máy vector hỗ trợ (SVM)**: Hoạt động tốt với dữ liệu có độ phức tạp cao.

---

## 4. Kết quả mô hình

Bảng dưới đây tổng hợp kết quả đánh giá các mô hình theo **độ chính xác (Accuracy), F1 Score, Precision, Recall và thời gian huấn luyện (Training Time in seconds)**.

| Feature | Model | Accuracy | F1 Score | Precision | Recall | Training Time (s) |
|---------|---------------------|----------|----------|----------|----------|----------------|
| TF-IDF | Logistic Regression | **83.44%** | 82.44% | 82.43% | **83.44%** | **0.39** |
| TF-IDF | MLP Classifier | 81.78% | 81.72% | 81.67% | 81.78% | **311.66** |
| TF-IDF | Random Forest | 83.31% | 81.85% | 83.04% | 83.31% | 17.18 |
| TF-IDF | SVM | **83.69%** | **82.59%** | **82.68%** | 83.69% | 51.41 |
| Word2Vec CBOW | Logistic Regression | 78.18% | 75.75% | 77.19% | 78.18% | 0.66 |
| Word2Vec CBOW | MLP Classifier | 81.11% | 80.14% | 80.04% | 81.11% | 20.20 |
| Word2Vec CBOW | Random Forest | 78.00% | 76.32% | 77.35% | 78.00% | 15.89 |
| Word2Vec CBOW | SVM | 76.87% | 72.01% | 79.72% | 76.87% | 48.18 |
| Word2Vec Skip-gram | Logistic Regression | 80.02% | 77.94% | 78.77% | 80.02% | 0.45 |
| Word2Vec Skip-gram | MLP Classifier | **83.34%** | 82.53% | 82.50% | **83.34%** | 17.33 |
| Word2Vec Skip-gram | Random Forest | 80.68% | 78.57% | 80.20% | 80.68% | 16.45 |
| Word2Vec Skip-gram | SVM | 80.17% | 77.00% | 80.22% | 80.17% | 39.53 |

---

## 5. Ứng dụng thực tế
Hệ thống có thể được ứng dụng vào nhiều lĩnh vực như:

✅ **Phân tích phản hồi khách hàng**: Tự động đánh giá cảm xúc trong đánh giá sản phẩm, dịch vụ.  
✅ **Quản lý mạng xã hội**: Giám sát bình luận tiêu cực để xử lý kịp thời.  
✅ **Hỗ trợ chatbot thông minh**: Cải thiện tương tác với người dùng dựa trên phân tích cảm xúc.  
✅ **Nghiên cứu và phân tích dư luận**: Theo dõi xu hướng cảm xúc trên các nền tảng trực tuyến.  

---

## 6. Cách sử dụng
Cài python 3.10 (only 3.10 vì thư viện underthesea support 3.10)

Ứng dụng được triển khai bằng **Streamlit**, người dùng có thể nhập văn bản và nhận kết quả phân tích cảm xúc trực quan.

Câu lệnh để chạy ứng dụng:
```sh
pip install -r requirements.txt
streamlit run app.py
```

---

💡 **Liên hệ**: Nếu bạn có câu hỏi hoặc muốn đóng góp, vui lòng liên hệ qua email["ctv55345@gmail.com"] hoặc GitHub repository. 🚀
