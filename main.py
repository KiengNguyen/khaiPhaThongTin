import os
import numpy as np
import pickle
from word2vec_tn2 import model_w2v, document_vector
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from gensim.models import Word2Vec

import pickle

# Tải mô hình Word2Vec đã huấn luyện
try:
    model_w2v = Word2Vec.load("word2vec.model")
except Exception as e:
    print(f"⚠️ Lỗi khi tải mô hình Word2Vec: {e}")
    model_w2v = None
    
# Hàm đọc các văn bản và nhãn từ thư mục vnexpress_data
def read_labeled_documents(folder_path):
    """Đọc toàn bộ các file .txt và trả về danh sách văn bản và nhãn"""
    documents = []
    labels = []
    for label in os.listdir(folder_path):
        label_folder = os.path.join(folder_path, label)
        
        if os.path.isdir(label_folder):
            for filename in os.listdir(label_folder):
                if filename.endswith(".txt"):
                    file_path = os.path.join(label_folder, filename)
                    with open(file_path, "r", encoding="utf-8") as f:
                        doc_content = f.read()
                        documents.append(doc_content)
                        labels.append(label)  # Nhãn là tên thư mục con
    return documents, labels

# Đọc dữ liệu và nhãn thực tế
documents, labels = read_labeled_documents("vnexpress_tap_kiem_thu")
print(f"📂 Đã đọc {len(documents)} văn bản với {len(labels)} nhãn.")

# Chuyển các văn bản thành vector
X = np.array([document_vector(model_w2v, doc) for doc in documents])

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Huấn luyện mô hình KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
with open("knn_model.pkl", "wb") as f:
    pickle.dump(knn, f)

# Huấn luyện mô hình Naïve Bayes
nb = MultinomialNB()
nb.fit(X_train_scaled, y_train)
with open("nb_model.pkl", "wb") as f:
    pickle.dump(nb, f)

# Huấn luyện mô hình Rocchio (NearestCentroid) nếu chưa có tệp mô hình
if not os.path.exists("rocchio_model.pkl"):
    rocchio = NearestCentroid()
    rocchio.fit(X_train_scaled, y_train)
    with open("rocchio_model.pkl", "wb") as f:
        pickle.dump(rocchio, f)
    print("🎯 Đã lưu mô hình Rocchio.")

# Đọc mô hình KNN, Naïve Bayes và Rocchio đã lưu
with open("knn_model.pkl", "rb") as f:
    knn = pickle.load(f)
with open("nb_model.pkl", "rb") as f:
    nb = pickle.load(f)
with open("rocchio_model.pkl", "rb") as f:
    rocchio = pickle.load(f)

# Dự đoán nhãn cho tập kiểm tra
y_pred_knn = knn.predict(X_test_scaled)
y_pred_nb = nb.predict(X_test_scaled)

# Cài đặt thuật toán Rocchio (NearestCentroid)
rocchio = NearestCentroid()
rocchio.fit(X_train_scaled, y_train)  # Dùng tập huấn luyện để huấn luyện Rocchio
y_pred_rocchio = rocchio.predict(X_test_scaled)

# Đánh giá kết quả cho KNN
knn_precision = precision_score(y_test, y_pred_knn, average='weighted', zero_division=1)
knn_recall = recall_score(y_test, y_pred_knn, average='weighted', zero_division=1)
knn_f1 = f1_score(y_test, y_pred_knn, average='weighted', zero_division=1)
knn_accuracy = accuracy_score(y_test, y_pred_knn)

# Đánh giá kết quả cho Naïve Bayes
nb_precision = precision_score(y_test, y_pred_nb, average='weighted', zero_division=1)
nb_recall = recall_score(y_test, y_pred_nb, average='weighted', zero_division=1)
nb_f1 = f1_score(y_test, y_pred_nb, average='weighted', zero_division=1)
nb_accuracy = accuracy_score(y_test, y_pred_nb)

# Đánh giá kết quả cho Rocchio (Nearest Centroid)
rocchio_precision = precision_score(y_test, y_pred_rocchio, average='weighted', zero_division=1)
rocchio_recall = recall_score(y_test, y_pred_rocchio, average='weighted', zero_division=1)
rocchio_f1 = f1_score(y_test, y_pred_rocchio, average='weighted', zero_division=1)
rocchio_accuracy = accuracy_score(y_test, y_pred_rocchio)

# In kết quả đánh giá
print(f"🎯 KNN Accuracy: {knn_accuracy:.4f}, Precision: {knn_precision:.4f}, Recall: {knn_recall:.4f}, F1-Score: {knn_f1:.4f}")
print(f"🎯 Naïve Bayes Accuracy: {nb_accuracy:.4f}, Precision: {nb_precision:.4f}, Recall: {nb_recall:.4f}, F1-Score: {nb_f1:.4f}")
print(f"🎯 Rocchio Accuracy: {rocchio_accuracy:.4f}, Precision: {rocchio_precision:.4f}, Recall: {rocchio_recall:.4f}, F1-Score: {rocchio_f1:.4f}")


# Lưu nhãn phân lớp vào file
with open("labels_predicted_knn.txt", "w", encoding="utf-8") as f:
    for label in y_pred_knn:
        f.write(label + "\n")

with open("labels_predicted_nb.txt", "w", encoding="utf-8") as f:
    for label in y_pred_nb:
        f.write(label + "\n")

with open("labels_predicted_rocchio.txt", "w", encoding="utf-8") as f:
    for label in y_pred_rocchio:
        f.write(label + "\n")

print("\n✅ Nhãn phân lớp đã được lưu vào file.")