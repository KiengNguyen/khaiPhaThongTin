import os
import numpy as np
import pickle
from tfidf import compute_tfidf_vectors  # Import function từ tfidf.py
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler

# Hàm đọc văn bản và nhãn
def read_labeled_documents(folder_path):
    documents = []
    labels = []
    for label in os.listdir(folder_path):
        label_folder = os.path.join(folder_path, label)
        if os.path.isdir(label_folder):
            for filename in os.listdir(label_folder):
                if filename.endswith(".txt"):
                    file_path = os.path.join(label_folder, filename)
                    with open(file_path, "r", encoding="utf-8") as f:
                        documents.append(f.read())
                        labels.append(label)
    return documents, labels

# Đọc dữ liệu huấn luyện và kiểm thử
train_documents, train_labels = read_labeled_documents("data/vnexpress_data")
test_documents, test_labels = read_labeled_documents("data/vnexpress_tap_kiem_thu")

print(f"Đã đọc {len(train_documents)} văn bản huấn luyện và {len(test_documents)} văn bản kiểm thử.")

# Tính toán vector TF-IDF cho văn bản huấn luyện và kiểm thử
X_train_tfidf = compute_tfidf_vectors(train_documents)
X_test_tfidf = compute_tfidf_vectors(test_documents)
y_train = train_labels
y_test = test_labels

# Chuẩn hóa dữ liệu TF-IDF
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_tfidf.toarray())  # Chuyển đổi từ sparse matrix thành array
X_test_scaled = scaler.transform(X_test_tfidf.toarray())  # Chuyển đổi từ sparse matrix thành array

# Huấn luyện và lưu KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
with open("knn_model.pkl", "wb") as f:
    pickle.dump(knn, f)

# Huấn luyện và lưu Naïve Bayes
nb = MultinomialNB()
nb.fit(X_train_scaled, y_train)
with open("nb_model.pkl", "wb") as f:
    pickle.dump(nb, f)

# Huấn luyện và lưu Rocchio
rocchio = NearestCentroid()
rocchio.fit(X_train_scaled, y_train)
with open("rocchio_model.pkl", "wb") as f:
    pickle.dump(rocchio, f)

# Dự đoán
y_pred_knn = knn.predict(X_test_scaled)
y_pred_nb = nb.predict(X_test_scaled)
y_pred_rocchio = rocchio.predict(X_test_scaled)

# Đánh giá KNN
print("KNN:")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_knn):.4f} | "
      f"Precision: {precision_score(y_test, y_pred_knn, average='weighted', zero_division=1):.4f} | "
      f"Recall: {recall_score(y_test, y_pred_knn, average='weighted', zero_division=1):.4f} | "
      f"F1-score: {f1_score(y_test, y_pred_knn, average='weighted', zero_division=1):.4f}")

# Đánh giá Naïve Bayes
print("Naïve Bayes:")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_nb):.4f} | "
      f"Precision: {precision_score(y_test, y_pred_nb, average='weighted', zero_division=1):.4f} | "
      f"Recall: {recall_score(y_test, y_pred_nb, average='weighted', zero_division=1):.4f} | "
      f"F1-score: {f1_score(y_test, y_pred_nb, average='weighted', zero_division=1):.4f}")

# Đánh giá Rocchio
print("Rocchio:")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_rocchio):.4f} | "
      f"Precision: {precision_score(y_test, y_pred_rocchio, average='weighted', zero_division=1):.4f} | "
      f"Recall: {recall_score(y_test, y_pred_rocchio, average='weighted', zero_division=1):.4f} | "
      f"F1-score: {f1_score(y_test, y_pred_rocchio, average='weighted', zero_division=1):.4f}")

# Lưu kết quả dự đoán
with open("labels_predicted_knn.txt", "w", encoding="utf-8") as f:
    for label in y_pred_knn:
        f.write(label + "\n")

with open("labels_predicted_nb.txt", "w", encoding="utf-8") as f:
    for label in y_pred_nb:
        f.write(label + "\n")

with open("labels_predicted_rocchio.txt", "w", encoding="utf-8") as f:
    for label in y_pred_rocchio:
        f.write(label + "\n")

print("\nĐã lưu kết quả phân loại vào file.")
