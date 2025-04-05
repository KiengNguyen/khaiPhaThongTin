import os
import re
from underthesea import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
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

all_documents = []
tagged_documents = []

# Đường dẫn đến thư mục dữ liệu
data_dir = 'data/vnexpress_data'
success_count = 0
error_count = 0
encodings = ['utf-8', 'latin-1', 'cp1252']

for category in os.listdir(data_dir):
    category_path = os.path.join(data_dir, category)
    if os.path.isdir(category_path):
        print(f"Đang xử lý thư mục: {category}")
        for idx, filename in enumerate(os.listdir(category_path)):
            if filename.endswith('.txt') and not filename.startswith('._'):
                file_path = os.path.join(category_path, filename)
                content = None
                for encoding in encodings:
                    try:
                        with open(file_path, 'r', encoding=encoding) as file:
                            content = file.read()
                            break
                    except Exception:
                        continue

                if content is not None:
                    clean_doc = re.sub(r'[^\w\s]', '', content.lower())
                    tokens = word_tokenize(clean_doc)
                    tokens = [token for token in tokens if token not in stop_words]

                    if tokens:
                        tag = f"{category}_{success_count}"
                        tagged_documents.append(TaggedDocument(words=tokens, tags=[tag]))
                        all_documents.append({'category': category, 'filename': filename, 'tokens': tokens, 'tag': tag})
                        success_count += 1
                else:
                    print(f"Không thể đọc file: {file_path}")
                    error_count += 1

print(f"Đã xử lý thành công {success_count} tài liệu, {error_count} lỗi từ {len(os.listdir(data_dir))} thư mục")

if len(tagged_documents) > 0:
    print("\nĐang huấn luyện mô hình Doc2Vec...")
    model = Doc2Vec(tagged_documents, vector_size=100, window=5, min_count=5, epochs=30)
    print("Đã hoàn thành huấn luyện mô hình Doc2Vec")

    def cosine_similarity(vec1, vec2):
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    print("\nĐang tính vector ngữ cảnh tổng quát...")
    all_vectors = [model.dv[doc['tag']] for doc in all_documents if doc['tag'] in model.dv]
    context_vector = np.mean(all_vectors, axis=0)

    print("\nĐang ghi kết quả vào file data.txt...")
    with open('data.txt', 'w', encoding='utf-8') as output_file:
        for category in sorted(set(doc['category'] for doc in all_documents)):
            output_file.write(f"\n=== DANH MỤC: {category} ===\n")
            category_docs = [doc for doc in all_documents if doc['category'] == category]

            for doc_idx, doc in enumerate(category_docs):
                output_file.write(f"\n--- Tài liệu: {doc['filename']} (#{doc_idx}) ---\n")
                doc_vector = model.dv[doc['tag']]

                max_token = None
                max_score = -1

                for token in doc['tokens']:
                    if token in model.wv:
                        similarity = cosine_similarity(model.wv[token], context_vector)
                        output_file.write(f"Từ: [{token}], Mức độ quan trọng: [{round(similarity, 4)}]\n")
                        if similarity > max_score:
                            max_score = similarity
                            max_token = token

                if max_token:
                    output_file.write(f"Từ quan trọng nhất: [{max_token}], Mức độ quan trọng cao nhất: [{round(max_score, 4)}]\n")

    print("\nĐang ghi tóm tắt vào file summary.txt...")
    with open('summary.txt', 'w', encoding='utf-8') as summary_file:
        summary_file.write("TÓM TẮT PHÂN TÍCH DỮ LIỆU VNEXPRESS\n")
        summary_file.write("=" * 50 + "\n\n")

        category_counts = {}
        for doc in all_documents:
            category = doc['category']
            category_counts[category] = category_counts.get(category, 0) + 1

        summary_file.write("THỐNG KÊ SỐ LƯỢNG TÀI LIỆU:\n")
        for category, count in sorted(category_counts.items()):
            summary_file.write(f"- {category}: {count} tài liệu\n")

        summary_file.write("\nTỪ QUAN TRỌNG NHẤT THEO DANH MỤC:\n")
        for category in sorted(set(doc['category'] for doc in all_documents)):
            category_docs = [doc for doc in all_documents if doc['category'] == category]
            cat_vecs = [model.dv[doc['tag']] for doc in category_docs]
            category_vector = np.mean(cat_vecs, axis=0)

            all_tokens = [token for doc in category_docs for token in doc['tokens']]
            word_counts = {word: all_tokens.count(word) for word in set(all_tokens)}

            best_word = None
            best_score = -1
            for word, count in word_counts.items():
                if count >= 5 and word in model.wv:
                    similarity = cosine_similarity(model.wv[word], category_vector)
                    if similarity > best_score:
                        best_score = similarity
                        best_word = word

            if best_word:
                summary_file.write(f"- {category}: [{best_word}] (độ quan trọng: {round(best_score, 4)})\n")

    print("\nPhân tích hoàn tất. Kết quả đã được lưu vào 'data.txt' và 'summary.txt'.")
    model.save("doc2vec.model")
    print("Mô hình đã được lưu vào 'doc2vec.model'")
    model_w2v = model
else:
    print("Không có đủ dữ liệu để huấn luyện mô hình")

def document_vector(model, document):
    """
    Hàm tính vector trung bình của các từ trong tài liệu dựa trên mô hình Word2Vec.
    """
    vectors = [model.wv[word] for word in document if word in model.wv]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)  # Nếu không có từ nào trong mô hình, trả về vector toàn 0
    return np.mean(vectors, axis=0)
