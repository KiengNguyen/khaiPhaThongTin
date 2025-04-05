import os
import numpy as np
import pickle
from tfidf import compute_tfidf_vectors
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer

def compute_tfidf_vectors(documents, max_features=1000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(documents)
    return X

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

# Đọc dữ liệu
documents, labels = read_labeled_documents("data/vnexpress_tap_kiem_thu")
print(f"📂 Đã đọc {len(documents)} văn bản với {len(labels)} nhãn.")

# ----- TF-IDF vectors -----
X_tfidf = compute_tfidf_vectors(documents)
X = X_tfidf.toarray()

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Chuẩn hóa
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

# Naïve Bayes
nb = MultinomialNB()
nb.fit(X_train_scaled, y_train)
y_pred_nb = nb.predict(X_test_scaled)

# Rocchio (Nearest Centroid)
rocchio = NearestCentroid()
rocchio.fit(X_train_scaled, y_train)
y_pred_rocchio = rocchio.predict(X_test_scaled)

# Hàm đánh giá
def evaluate_model(name, y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)
    accuracy = accuracy_score(y_true, y_pred)
    print(f"🎯 {name} Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    return precision, recall, f1, accuracy

# Đánh giá
evaluate_model("KNN", y_test, y_pred_knn)
evaluate_model("Naïve Bayes", y_test, y_pred_nb)
evaluate_model("Rocchio", y_test, y_pred_rocchio)

# Ghi kết quả nhãn dự đoán ra file
with open("labels_predicted_knn.txt", "w", encoding="utf-8") as f:
    f.writelines(label + "\n" for label in y_pred_knn)

with open("labels_predicted_nb.txt", "w", encoding="utf-8") as f:
    f.writelines(label + "\n" for label in y_pred_nb)

with open("labels_predicted_rocchio.txt", "w", encoding="utf-8") as f:
    f.writelines(label + "\n" for label in y_pred_rocchio)

print("\n✅ Đã lưu nhãn dự đoán vào file.")
