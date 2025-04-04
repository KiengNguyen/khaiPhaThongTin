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
all_sentences = []
all_documents = []

# Đường dẫn đến thư mục chính
data_dir = "D:\\KhaiPhaThongTin\\data\\vnexpress"

# Đếm số file đã xử lý thành công
success_count = 0
error_count = 0

# Các encoding phổ biến để thử
encodings = ['utf-8', 'latin-1', 'cp1252']

# Duyệt qua các thư mục con trong thư mục vnexpress_data
for category in os.listdir(data_dir):
    category_path = os.path.join(data_dir, category)
    
    # Kiểm tra nếu là thư mục
    if os.path.isdir(category_path):
        print(f"Đang xử lý thư mục: {category}")
        
        # Duyệt qua các file trong thư mục con
        for filename in os.listdir(category_path):
            if filename.endswith('.txt') and not filename.startswith('._'):  # Bỏ qua các file ẩn
                file_path = os.path.join(category_path, filename)
                
                # Thử các encoding khác nhau
                content = None
                for encoding in encodings:
                    try:
                        with open(file_path, 'r', encoding=encoding) as file:
                            content = file.read()
                            break  # Nếu đọc thành công thì thoát vòng lặp
                    except Exception:
                        continue  # Thử encoding tiếp theo
                
                if content is not None:
                    # Chuyển về chữ thường và loại bỏ các ký tự đặc biệt
                    clean_doc = re.sub(r'[^\w\s]', '', content.lower())
                    
                    # Tách từ tiếng Việt sử dụng UnderTheSea
                    tokens = word_tokenize(clean_doc)
                    
                    # Loại bỏ từ dừng và thay thế khoảng trắng bằng gạch dưới
                    tokens = [token.replace(' ', '_') for token in tokens if token not in stop_words]
                    
                    # Kiểm tra nếu tokens không rỗng
                    if tokens:
                        # Thêm vào danh sách các câu để huấn luyện Word2Vec
                        all_sentences.append(tokens)
                        
                        # Lưu thông tin về tài liệu (thư mục con, tên file và tokens)
                        all_documents.append({
                            'category': category,
                            'filename': filename,
                            'tokens': tokens
                        })
                        success_count += 1
                else:
                    print(f"Không thể đọc file: {file_path} với bất kỳ encoding nào")
                    error_count += 1

# Kiểm tra số lượng tài liệu đã xử lý
print(f"Đã xử lý thành công {success_count} tài liệu, {error_count} lỗi từ {len(os.listdir(data_dir))} thư mục")

# Kiểm tra nếu có đủ dữ liệu để huấn luyện
if len(all_sentences) > 0:
    # Huấn luyện mô hình Word2Vec theo phương pháp CBOW
    print("Đang huấn luyện mô hình Word2Vec...")
    model = Word2Vec(all_sentences, vector_size=100, window=5, sg=0, min_count=5, epochs=30)
    print("Đã hoàn thành huấn luyện mô hình Word2Vec")

    # Lấy vector trung bình của toàn bộ văn bản để làm ngữ cảnh tổng quát
    def get_mean_vector(model, words):
        vectors = [model.wv[word] for word in words if word in model.wv]
        if len(vectors) == 0:
            return np.zeros(model.vector_size)
        return np.mean(vectors, axis=0)

    # Tạo vector ngữ cảnh tổng quát từ tất cả các từ trong tất cả các tài liệu
    print("Đang tính vector ngữ cảnh tổng quát...")
    all_words = [word for doc in all_documents for word in doc['tokens']]
    context_vector = get_mean_vector(model, all_words)

    # Tính toán mức độ quan trọng của từ dựa trên độ tương tự cosine
    def cosine_similarity(vec1, vec2):
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    # Ghi kết quả vào file data.txt
    print("Đang ghi kết quả vào file data.txt...")
    with open('data.txt', 'w', encoding='utf-8') as output_file:
        # Phân tích theo từng thư mục con (danh mục)
        for category in sorted(set(doc['category'] for doc in all_documents)):
            output_file.write(f"\n=== DANH MỤC: {category} ===\n")
            
            # Lọc các tài liệu thuộc danh mục này
            category_docs = [doc for doc in all_documents if doc['category'] == category]
            
            # Phân tích từng tài liệu trong danh mục
            for doc_idx, doc in enumerate(category_docs):
                output_file.write(f"\n--- Tài liệu: {doc['filename']} (#{doc_idx}) ---\n")
                
                # Tìm từ có mức độ quan trọng cao nhất
                most_important_word = None
                max_importance = -1
                
                for token in doc['tokens']:
                    if token in model.wv:
                        word_vector = model.wv[token]
                        importance = cosine_similarity(word_vector, context_vector)
                        output_file.write(f'Từ: [{token}], Mức độ quan trọng (Cosine Similarity): [{round(importance, 4)}]\n')
                        
                        if importance > max_importance:
                            max_importance = importance
                            most_important_word = token
                
                # Ghi ra từ quan trọng nhất của văn bản hiện tại
                if most_important_word:
                    output_file.write(f'Từ quan trọng nhất: [{most_important_word}], Mức độ quan trọng cao nhất: [{round(max_importance, 4)}]\n')

    # Tạo một bản tóm tắt tổng quan
    print("Đang ghi tóm tắt vào file summary.txt...")
    with open('summary.txt', 'w', encoding='utf-8') as summary_file:
        summary_file.write("TÓM TẮT PHÂN TÍCH DỮ LIỆU VNEXPRESS\n")
        summary_file.write("=" * 50 + "\n\n")
        
        # Thống kê số lượng tài liệu theo danh mục
        category_counts = {}
        for doc in all_documents:
            category = doc['category']
            if category in category_counts:
                category_counts[category] += 1
            else:
                category_counts[category] = 1
        
        summary_file.write("THỐNG KÊ SỐ LƯỢNG TÀI LIỆU:\n")
        for category, count in sorted(category_counts.items()):
            summary_file.write(f"- {category}: {count} tài liệu\n")
        
        summary_file.write("\nTỪ QUAN TRỌNG NHẤT THEO DANH MỤC:\n")
        
        # Tìm từ quan trọng nhất cho mỗi danh mục
        for category in sorted(set(doc['category'] for doc in all_documents)):
            category_words = [word for doc in all_documents if doc['category'] == category for word in doc['tokens']]
            category_vector = get_mean_vector(model, category_words)
            
            # Tìm từ quan trọng nhất trong danh mục
            most_important = None
            max_importance = -1
            
            # Chỉ xét các từ xuất hiện ít nhất 5 lần trong danh mục này
            word_counts = {}
            for word in category_words:
                if word in word_counts:
                    word_counts[word] += 1
                else:
                    word_counts[word] = 1
            
            for word, count in word_counts.items():
                if count >= 5 and word in model.wv:
                    importance = cosine_similarity(model.wv[word], category_vector)
                    if importance > max_importance:
                        max_importance = importance
                        most_important = word
            
            if most_important:
                summary_file.write(f"- {category}: [{most_important}] (độ quan trọng: {round(max_importance, 4)})\n")
else:
    print("Không có đủ dữ liệu để huấn luyện mô hình")

print("Phân tích hoàn tất. Kết quả đã được lưu vào file 'data.txt' và 'summary.txt'")