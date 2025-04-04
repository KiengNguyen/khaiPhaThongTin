import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os
import numpy as np
from underthesea import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity

def read_text_file(file_path):
    """Đọc dữ liệu từ file văn bản"""
    if os.path.isfile(file_path):  # Kiểm tra file có tồn tại không
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    else:
        print(f"File {file_path} không tồn tại.")
        return None

def preprocess_text(text):
    """Tiền xử lý văn bản: tách từ và chuyển thành chữ thường"""
    return word_tokenize(text, format="text").lower()

def read_files_from_folder(folder_path):
    """Đọc tất cả các file .txt trong thư mục"""
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and filename.endswith(".txt"):  # Chỉ đọc file .txt
            text = read_text_file(file_path)
            if text:
                documents.append(preprocess_text(text))
    return documents


def train_doc2vec(folder_path, model_path="doc2vec_model.model"):
    """Huấn luyện mô hình Doc2Vec từ nhiều văn bản trong thư mục"""
    texts = read_files_from_folder(folder_path)  # Đọc và xử lý văn bản

    # Hiển thị danh sách các tài liệu đã dùng
    print(f"Danh sách tài liệu đã sử dụng để huấn luyện chủ đề {topic}:")
    for i, filename in enumerate(os.listdir(folder_path)):
        if filename.endswith(".txt"):
            print(f"  {i + 1}. {filename}")

    # Gán nhãn cho mỗi tài liệu
    documents = [TaggedDocument(words=text.split(), tags=[f"Vector_TaiLieu_{i}"]) for i, text in enumerate(texts)]

    # Huấn luyện mô hình
    model = Doc2Vec(documents, vector_size=100, window=5, min_count=1, workers=4, epochs=20)

    # Lưu mô hình
    model.save(model_path)
    save_vectors_to_txt(model)
    # Load mô hình với mmap (tải một phần)
    model = Doc2Vec.load("doc2vec_model.model", mmap='r')

    # Truy xuất vector của tài liệu hoặc từ một phần
    taiLieu_vector = model.dv["Vector_TaiLieu_0"]
    print("Mô hình đã được huấn luyện và lưu thành công!")

def save_vectors_to_txt(model, filename="taiLieu_vectors.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        for i in range(len(model.dv)):
            vector = model.dv[i]  # Lấy vector của tài liệu
            f.write(f"Vector_TaiLieu_{i}: {list(vector)}\n")  # Lưu vector dưới dạng danh sách
def get_document_vector(model_path="doc2vec_model.model", Vector_TaiLieu_index=2):
    """Lấy vector biểu diễn văn bản từ mô hình Doc2Vec"""
    model = Doc2Vec.load(model_path)
    vector = model.dv[f"Vector_TaiLieu_{Vector_TaiLieu_index}"]
    print(f"Vector biểu diễn văn bản {Vector_TaiLieu_index}:", vector)
    return vector

def get_vector_from_file(file_path, model_path="doc2vec_model.model"):
    """Lấy vector từ một file cụ thể bằng mô hình Doc2Vec"""
    text = read_text_file(file_path)
    if text:
        processed_text = preprocess_text(text)
        model = Doc2Vec.load(model_path)
        vector = model.infer_vector(processed_text.split())  # Suy diễn vector từ văn bản mới
        print(f"Vector biểu diễn file {file_path}:", vector)
        return vector
    return None
def compare_vectors(vector1, vector2):
    """So sánh hai vector bằng Cosine Similarity và Euclidean Distance"""
    vector1 = np.array(vector1).reshape(1, -1)
    vector2 = np.array(vector2).reshape(1, -1)

    cosine_sim = cosine_similarity(vector1, vector2)[0][0]
    euclidean_dist = np.linalg.norm(vector1 - vector2)
    print("------Kết quả so sánh-------")
    print(f"Độ tương đồng Cosine: {cosine_sim:.4f}")
    print(f"Khoảng cách Euclidean: {euclidean_dist:.4f}")

    return cosine_sim, euclidean_dist
if __name__ == "__main__":
    # Nhập tên topic và đường dẫn của thư mục
    topic = input("Nhập tên topic: ")
    data_root_dir = f"data/vnexpress/{topic}/"

    # Huấn luyện mô hình Doc2Vec
    train_doc2vec(data_root_dir)

    # Lấy vector từ văn bản thứ 0 trong tập huấn luyện
    get_document_vector()

    # Đọc một file cụ thể và lấy vector
    file_path_1 = "data/vnexpress/the-thao/alcaraz-muon-giu-vi-tri-so-mot-toi-het-nam-4530537.txt"
    vector1 = get_vector_from_file(file_path_1, "doc2vec_the-thao.model")
    file_path_2 = "data/vnexpress/du-lich/6-khach-san-viet-vao-top-sang-trong-nhat-dong-nam-a-4529357.txt"
    vector2 = get_vector_from_file(file_path_2, "doc2vec_the-thao.model")
    file_path_3 = "data/vnexpress/du-lich/ben-trong-du-thuyen-dau-gia-hon-35-ty-dong-cua-flc-4527411.txt"
    vector3 = get_vector_from_file(file_path_3, "doc2vec_du-lich.model")

    # So sánh hai vector
    compare_vectors(vector1, vector2)
    compare_vectors(vector3, vector3)
