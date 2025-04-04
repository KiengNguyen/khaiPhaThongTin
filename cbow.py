import os
import re
import math
from underthesea import word_tokenize
from gensim.models import Word2Vec
import numpy as np

# Đọc danh sách từ dừng từ file vietnamese-stopwords.txt
with open("vietnamese-stopwords.txt", "r", encoding="utf-8") as f:
    stop_words = set(f.read().splitlines())

# Đọc dữ liệu từ các file .txt trong thư mục 'du-lich'
data_dir = "D:\\KhaiPhaThongTin\\data\\vnexpress\\du-lich"

sentences = []
document_tokens = []  # Lưu trữ tất cả token của từng văn bản

for filename in os.listdir(data_dir):
    if filename.endswith('.txt'):
        with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as file:
            content = file.read()
            # Chuyển về chữ thường và loại bỏ các ký tự đặc biệt
            clean_doc = re.sub(r'[^\w\s]', '', content.lower())
            # Tách từ tiếng Việt sử dụng UnderTheSea
            tokens = word_tokenize(clean_doc)
            tokens = [token.replace(' ', '_') for token in tokens if token not in stop_words]
            sentences.append(tokens)
            document_tokens.append(tokens)

# Huấn luyện mô hình Word2Vec theo phương pháp CBOW
model = Word2Vec(sentences, vector_size=100, window=5, sg=0, min_count=1, epochs=100)


# Lấy vector trung bình của toàn bộ văn bản để làm ngữ cảnh tổng quát

def get_mean_vector(model, words):
    vectors = [model.wv[word] for word in words if word in model.wv]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)


# Tính toán độ tương tự giữa từ và ngữ cảnh tổng quát của toàn bộ văn bản
context_vector = get_mean_vector(model, [word for sentence in sentences for word in sentence])


# Tính toán mức độ quan trọng của từ dựa trên độ tương tự cosine

def cosine_similarity(vec1, vec2):
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# Ghi kết quả vào file data.txt
with open('data.txt', 'w', encoding='utf-8') as output_file:
    for doc_idx, sentence in enumerate(document_tokens):
        output_file.write(f"\n--- Tài liệu số [{doc_idx}] ---\n")

        # Tìm từ có mức độ quan trọng cao nhất
        most_important_word = None
        max_importance = -1

        for token in sentence:
            if token in model.wv:
                word_vector = model.wv[token]
                importance = cosine_similarity(word_vector, context_vector)
                output_file.write(f'Từ: [{token}], Mức độ quan trọng (Cosine Similarity): [{round(importance, 4)}]\n')

                if importance > max_importance:
                    max_importance = importance
                    most_important_word = token

        # Ghi ra từ quan trọng nhất của văn bản hiện tại
        if most_important_word:
            output_file.write(
                f'Từ quan trọng nhất: [{most_important_word}], Mức độ quan trọng cao nhất: [{round(max_importance, 4)}]\n')
