import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os
from underthesea import word_tokenize


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
    """Huấn luyện mô hình Doc2Vec cho từng chủ đề trong thư mục gốc"""
    for topic in os.listdir(root_folder):
        topic_path = os.path.join(root_folder, topic)
        if os.path.isdir(topic_path):  # Kiểm tra nếu là thư mục con (chủ đề)
            print(f"Đang huấn luyện mô hình cho chủ đề: {topic}")
            texts = read_files_from_folder(topic_path)

            if not texts:
                print(f"Bỏ qua chủ đề {topic} vì không có dữ liệu.")
                continue

            documents = [TaggedDocument(words=text.split(), tags=[f"{topic}_doc_{i}"]) for i, text in enumerate(texts)]
            model = Doc2Vec(documents, vector_size=500, window=5, min_count=1, workers=4, epochs=20)

            model_path = f"doc2vec_{topic}.model"
            model.save(model_path)
            print(f"Mô hình cho chủ đề {topic} đã được lưu: {model_path}")


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


if __name__ == "__main__":
    data_root_dir = "data/vnexpress/"  # Thư mục gốc chứa các chủ đề
    train_doc2vec_for_each_topic(data_root_dir)

    # Ví dụ lấy vector từ một file trong chủ đề "the-thao"
    file_path = "data/vnexpress/the-thao/alcaraz-muon-giu-vi-tri-so-mot-toi-het-nam-4530537.txt"
    model_path = "doc2vec_the-thao.model"
    get_vector_from_file(file_path, model_path)
