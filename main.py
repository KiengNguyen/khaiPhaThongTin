import os
import numpy as np
import pickle
from word2vec_tn2 import model_w2v, document_vector
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from gensim.models import Word2Vec

# Tải mô hình Word2Vec đã huấn luyện
try:
    model_w2v = Word2Vec.load("word2vec.model")
except Exception as e:
    print(f"Lỗi khi tải mô hình Word2Vec: {e}")
    model_w2v = None

# Hàm đọc văn bản và nhãn
def read_labeled_documents(folder_path):
    documents = []
    labels = []
    for label in os.listdir(folder_path):  # Đây là thư mục con, ví dụ: 'the_thao', 'chinh_tri'
        label_folder = os.path.join(folder_path, label)
        if os.path.isdir(label_folder):
            for filename in os.listdir(label_folder):
                if filename.endswith(".txt"):
                    file_path = os.path.join(label_folder, filename)
                    with open(file_path, "r", encoding="utf-8") as f:
                        documents.append(f.read())
                        labels.append(label)  # Nhãn lấy từ tên thư mục con
    return documents, labels

# Đọc dữ liệu huấn luyện và kiểm thử
train_documents, train_labels = read_labeled_documents("data/vnexpress_data")
test_documents, test_labels = read_labeled_documents("data/vnexpress_tap_kiem_thu")

print(f"Đã đọc {len(train_documents)} văn bản huấn luyện và {len(test_documents)} văn bản kiểm thử.")

# Chuyển sang vector
X_train = np.array([document_vector(model_w2v, doc) for doc in train_documents])
X_test = np.array([document_vector(model_w2v, doc) for doc in test_documents])
y_train = train_labels
y_test = test_labels

# Chuẩn hóa dữ liệu
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

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
