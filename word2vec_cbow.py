import os
import re
import math
from underthesea import word_tokenize
from gensim.models import Word2Vec
import numpy as np

# Đọc danh sách từ dừng từ file vietnamese-stopwords.txt
try:
    with open("vietnamese-stopwords.txt", "r", encoding="utf-8") as f:
        stop_words = set(f.read().splitlines())
except UnicodeDecodeError:
    try:
        with open("vietnamese-stopwords.txt", "r", encoding="latin-1") as f:
            stop_words = set(f.read().splitlines())
    except Exception as e:
        print(f"Lỗi khi đọc file vietnamese-stopwords.txt: {e}")
        stop_words = set()

# Khởi tạo danh sách để lưu trữ câu và token từ tất cả các tài liệu
all_sentences = []           # Dùng để huấn luyện mô hình Word2Vec
all_documents = []           # Lưu thông tin từng tài liệu đã xử lý

# Đường dẫn đến thư mục chứa dữ liệu phân loại theo danh mục
data_dir = 'data/vnexpress_data'

# Đếm số file đã xử lý thành công hoặc thất bại
success_count = 0
error_count = 0

# Các encoding phổ biến để thử khi đọc file
encodings = ['utf-8', 'latin-1', 'cp1252']

# Duyệt qua các thư mục con tương ứng với danh mục bài viết
for category in os.listdir(data_dir):
    category_path = os.path.join(data_dir, category)

    if os.path.isdir(category_path):
        print(f"Đang xử lý thư mục: {category}")

        for filename in os.listdir(category_path):
            if filename.endswith('.txt') and not filename.startswith('._'):
                file_path = os.path.join(category_path, filename)

                content = None
                # Cố gắng mở file với các encoding khác nhau
                for encoding in encodings:
                    try:
                        with open(file_path, 'r', encoding=encoding) as file:
                            content = file.read()
                            break
                    except Exception:
                        continue

                # Nếu mở được file
                if content is not None:
                    # Làm sạch văn bản và tách từ
                    clean_doc = re.sub(r'[^\w\s]', '', content.lower())
                    tokens = word_tokenize(clean_doc)
                    tokens = [token for token in tokens if token not in stop_words]

                    # Nếu còn từ sau khi lọc, lưu lại
                    if tokens:
                        all_sentences.append(tokens)
                        all_documents.append({'category': category, 'filename': filename, 'tokens': tokens})
                        success_count += 1
                else:
                    print(f"Không thể đọc file: {file_path}")
                    error_count += 1

print(f"\nĐã xử lý thành công {success_count} tài liệu, {error_count} lỗi từ {len(os.listdir(data_dir))} thư mục")

# Nếu có dữ liệu, tiến hành huấn luyện mô hình
if len(all_sentences) > 0:
    print("\nĐang huấn luyện mô hình Word2Vec...")
    model = Word2Vec(all_sentences, vector_size=100, window=5, sg=0, min_count=5, epochs=30)
    print("Đã hoàn thành huấn luyện mô hình Word2Vec")

    # Hàm tính vector trung bình từ danh sách từ
    def get_mean_vector(model, words):
        vectors = [model.wv[word] for word in words if word in model.wv]
        if len(vectors) == 0:
            return np.zeros(model.vector_size)
        return np.mean(vectors, axis=0)

    # Tính vector ngữ cảnh tổng quát cho toàn bộ tập dữ liệu
    print("\nĐang tính vector ngữ cảnh tổng quát...")
    all_words = [word for doc in all_documents for word in doc['tokens']]
    context_vector = get_mean_vector(model, all_words)

    # Hàm tính cosine similarity giữa 2 vector
    def cosine_similarity(vec1, vec2):
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    # Ghi chi tiết vào data.txt
    print("\nĐang ghi kết quả vào file data.txt...")
    with open('data.txt', 'w', encoding='utf-8') as output_file:
        for category in sorted(set(doc['category'] for doc in all_documents)):
            output_file.write(f"\n=== DANH MỤC: {category} ===\n")
            category_docs = [doc for doc in all_documents if doc['category'] == category]

            for doc_idx, doc in enumerate(category_docs):
                output_file.write(f"\n--- Tài liệu: {doc['filename']} (#{doc_idx}) ---\n")

                most_important_word = None
                max_importance = -1

                for token in doc['tokens']:
                    if token in model.wv:
                        word_vector = model.wv[token]
                        importance = cosine_similarity(word_vector, context_vector)
                        output_file.write(f'Từ: [{token}], Mức độ quan trọng: [{round(importance, 4)}]\n')

                        if importance > max_importance:
                            max_importance = importance
                            most_important_word = token

                if most_important_word:
                    output_file.write(f'Từ quan trọng nhất: [{most_important_word}], Mức độ quan trọng cao nhất: [{round(max_importance, 4)}]\n')

    # Ghi tóm tắt vào summary.txt
    print("\nĐang ghi tóm tắt vào file summary.txt...")
    with open('summary.txt', 'w', encoding='utf-8') as summary_file:
        summary_file.write("TÓM TẮT PHÂN TÍCH DỮ LIỆU VNEXPRESS\n")
        summary_file.write("=" * 50 + "\n\n")

        # Thống kê số lượng tài liệu theo danh mục
        summary_file.write("THỐNG KÊ SỐ LƯỢNG TÀI LIỆU:\n")
        category_counts = {}
        for doc in all_documents:
            category = doc['category']
            category_counts[category] = category_counts.get(category, 0) + 1

        for category, count in sorted(category_counts.items()):
            summary_file.write(f"- {category}: {count} tài liệu\n")

        # Tìm từ quan trọng nhất của mỗi danh mục
        summary_file.write("\nTỪ QUAN TRỌNG NHẤT THEO DANH MỤC:\n")
        for category in sorted(set(doc['category'] for doc in all_documents)):
            category_words = [word for doc in all_documents if doc['category'] == category for word in doc['tokens']]
            category_vector = get_mean_vector(model, category_words)

            most_important = None
            max_importance = -1
            word_counts = {word: category_words.count(word) for word in category_words}

            for word, count in word_counts.items():
                if count >= 5 and word in model.wv:
                    importance = cosine_similarity(model.wv[word], category_vector)
                    if importance > max_importance:
                        max_importance = importance
                        most_important = word

            if most_important:
                summary_file.write(f"- {category}: [{most_important}] (độ quan trọng: {round(max_importance, 4)})\n")

    print("\nLưu kết quả thành công vào data.txt và summary.txt.")

    # Cho phép người dùng nhập từ để tra cứu tương đồng
    def check_word_similarity(word):
        if word in model.wv:
            print(f"\nTừ [{word}] có trong mô hình Word2Vec.")
            similar_words = model.wv.most_similar(word, topn=10)
            print("\nCác từ tương tự nhất:")
            for similar_word, similarity in similar_words:
                print(f"- {similar_word}: {round(similarity, 4)}")

            word_vector = model.wv[word]
            context_similarity = cosine_similarity(word_vector, context_vector)
            print(f"\nĐộ tương đồng với ngữ cảnh tổng quát: {round(context_similarity, 4)}")
        else:
            print(f"\nTừ [{word}] không có trong mô hình Word2Vec.")

    while True:
        user_input = input("\nNhập một từ để kiểm tra độ tương đồng (hoặc gõ 'exit' để thoát): ").strip()
        if user_input.lower() == "exit":
            break
        check_word_similarity(user_input)

else:
    print("\nKhông có đủ dữ liệu để huấn luyện mô hình")

print("\nPhân tích hoàn tất. Kết quả đã được lưu vào 'data.txt' và 'summary.txt'.")