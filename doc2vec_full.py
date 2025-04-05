import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os
from underthesea import word_tokenize
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def read_text_file(file_path):
    """Đọc dữ liệu từ file văn bản"""
    if os.path.isfile(file_path):
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


def train_doc2vec_for_each_topic(root_folder):
    """Huấn luyện mô hình Doc2Vec cho từng chủ đề trong thư mục gốc và hiển thị số lượng file huấn luyện"""
    total_files = 0  # Biến đếm tổng số file

    for topic in os.listdir(root_folder):
        topic_path = os.path.join(root_folder, topic)
        if os.path.isdir(topic_path):  # Kiểm tra nếu là thư mục con (chủ đề)
            print(f"🔹 Đang huấn luyện mô hình cho chủ đề: {topic}")
            texts = read_files_from_folder(topic_path)

            num_files = len(texts)  # Đếm số file trong chủ đề này
            total_files += num_files  # Cộng dồn vào tổng số file

            if num_files == 0:
                print(f"⚠️ Bỏ qua chủ đề {topic} vì không có dữ liệu.")
                continue

            print(f"📂 Số file huấn luyện trong chủ đề '{topic}': {num_files}")

            documents = [TaggedDocument(words=text.split(), tags=[f"{topic}_doc_{i}"]) for i, text in enumerate(texts)]
            model = Doc2Vec(documents, vector_size=500, window=5, min_count=1, workers=4, epochs=20)

            model_path = f"doc2vec_{topic}.model"
            model.save(model_path)
            print(f"✅ Mô hình cho chủ đề '{topic}' đã được lưu: {model_path}\n")

    print(f"🔥 Tổng số file đã huấn luyện trên tất cả các chủ đề: {total_files}")


def get_vector_from_file(file_path, model_path):
    """Lấy vector từ một file cụ thể bằng mô hình Doc2Vec"""
    text = read_text_file(file_path)
    if text:
        processed_text = preprocess_text(text)
        model = Doc2Vec.load(model_path)
        vector = model.infer_vector(processed_text.split())
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
    data_root_dir = "data/vnexpress/"  # Thư mục gốc chứa các chủ đề
    train_doc2vec_for_each_topic(data_root_dir)

    # Đọc một file cụ thể và lấy vector
    file_path_1 = "data/vnexpress_data/the-thao/7_sai_lầm_runner_hay_mắc_phải_khi_chấn_thương.txt"
    vector1 = get_vector_from_file(file_path_1,"doc2vec_the-thao.model")
    file_path_2 = "data/vnexpress_data/suc-khoe/4_món_ăn_uống_không_tốt_cho_tinh_binh.txt"
    vector2 = get_vector_from_file(file_path_2,"doc2vec_suc-khoe.model")
    file_path_3 = "data/vnexpress_data/suc-khoe/7_hiểu_lầm_về_bệnh_sởi_ở_người_lớn.txt"
    vector3 = get_vector_from_file(file_path_3,"doc2vec_suc-khoe.model")

    # So sánh hai vector
    compare_vectors(vector1, vector2)
    compare_vectors(vector3, vector3)
