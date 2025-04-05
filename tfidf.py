import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

def read_documents_from_folder(folder_path):
    """
    Đọc các văn bản từ thư mục con trong thư mục 'folder_path' và trả về danh sách các văn bản.
    """
    documents = []
    for label in os.listdir(folder_path):
        label_folder = os.path.join(folder_path, label)
        if os.path.isdir(label_folder):
            for filename in os.listdir(label_folder):
                if filename.endswith(".txt"):
                    file_path = os.path.join(label_folder, filename)
                    with open(file_path, "r", encoding="utf-8") as f:
                        documents.append(f.read())
    return documents

def compute_tfidf_vectors(documents, max_features=1000, model_filename='tfidf_model.pkl'):
    """
    Nhận vào danh sách văn bản, trả về ma trận TF-IDF và lưu mô hình TF-IDF sau khi huấn luyện.
    """
    # Tạo TfidfVectorizer và tính toán TF-IDF
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(documents)
    
    # Lưu mô hình TF-IDF
    with open(model_filename, 'wb') as model_file:
        pickle.dump(vectorizer, model_file)
    
    return X

# Đọc dữ liệu từ thư mục 'vnexpress_data'
folder_path = "data/vnexpress_data"
documents = read_documents_from_folder(folder_path)

# Tính toán và lưu mô hình TF-IDF
X = compute_tfidf_vectors(documents, max_features=1000, model_filename='tfidf_model.pkl')

print("Đã lưu mô hình TF-IDF sau khi huấn luyện.")
