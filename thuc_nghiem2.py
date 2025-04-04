import numpy as np
import pandas as pd
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from sklearn.linear_model import SGDClassifier
from scipy.spatial import distance

# Load dataset từ thư mục data
data_dir = 'data/vnexpress/'  # Đổi thành đường dẫn thư mục của bạn
texts, labels = [], []

for label in os.listdir(data_dir):
    label_path = os.path.join(data_dir, label)
    if os.path.isdir(label_path):
        for file in os.listdir(label_path):
            file_path = os.path.join(label_path, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                texts.append(f.read().strip())
                labels.append(label)

data = pd.DataFrame({'text': texts, 'label': labels})

# Chia tập dữ liệu
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# 1. Biểu diễn bằng TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# 2. Biểu diễn bằng Word2Vec
w2v_model = Word2Vec(sentences=[t.split() for t in X_train], vector_size=100, window=5, min_count=1, workers=4)


def avg_word2vec(text, model):
    vectors = [model.wv[word] for word in text.split() if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)


X_train_w2v = np.array([avg_word2vec(text, w2v_model) for text in X_train])
X_test_w2v = np.array([avg_word2vec(text, w2v_model) for text in X_test])

# 3. Biểu diễn bằng Doc2Vec
doc2vec_model = Doc2Vec([TaggedDocument(words=t.split(), tags=[str(i)]) for i, t in enumerate(X_train)],
                        vector_size=100, window=5, min_count=1, workers=4)


def doc2vec_vector(text, model):
    return model.infer_vector(text.split())


X_train_d2v = np.array([doc2vec_vector(text, doc2vec_model) for text in X_train])
X_test_d2v = np.array([doc2vec_vector(text, doc2vec_model) for text in X_test])


# Hàm đánh giá hiệu suất mô hình
def evaluate_model(model, X_train, X_test, name):
    start_time = time.time()
    model.fit(X_train, y_train_enc)
    y_pred = model.predict(X_test)
    elapsed_time = time.time() - start_time

    print(f'Classification report for {name}:')
    print(classification_report(y_test_enc, y_pred))
    print(f'Training time: {elapsed_time:.4f} seconds')
    return precision_score(y_test_enc, y_pred, average='weighted'), \
        recall_score(y_test_enc, y_pred, average='weighted'), \
        f1_score(y_test_enc, y_pred, average='weighted')


# Huấn luyện và đánh giá với KNN
knn = KNeighborsClassifier(n_neighbors=3)
k_precision, k_recall, k_f1 = evaluate_model(knn, X_train_tfidf, X_test_tfidf, 'KNN with TF-IDF')

# Huấn luyện và đánh giá với Naïve Bayes
nb = MultinomialNB()
nb_precision, nb_recall, nb_f1 = evaluate_model(nb, X_train_tfidf, X_test_tfidf, 'Naïve Bayes with TF-IDF')

# Huấn luyện và đánh giá với Rocchio (SVM với loss='hinge' mô phỏng Rocchio)
rocchio = SGDClassifier(loss='hinge', alpha=0.01, max_iter=1000, random_state=42)
r_precision, r_recall, r_f1 = evaluate_model(rocchio, X_train_tfidf, X_test_tfidf, 'Rocchio with TF-IDF')

# So sánh kết quả
print('\nComparison of classification methods:')
print(f'{"Model":<20} {"Precision":<10} {"Recall":<10} {"F1-score":<10}')
print(f'{"KNN with TF-IDF":<20} {k_precision:.4f} {k_recall:.4f} {k_f1:.4f}')
print(f'{"Naïve Bayes with TF-IDF":<20} {nb_precision:.4f} {nb_recall:.4f} {nb_f1:.4f}')
print(f'{"Rocchio with TF-IDF":<20} {r_precision:.4f} {r_recall:.4f} {r_f1:.4f}')

# Xác định K tài liệu gần nhất và chủ đề dự đoán
k = 5
test_doc = "Hormone căng thẳng tăng cao, nhất là cortisol"  # Văn bản kiểm thử
test_doc_tfidf = tfidf_vectorizer.transform([test_doc]).toarray().ravel()

doc_idx_sim_dict = {}
for idx, train_vec in enumerate(X_train_tfidf):
    cs_sim = 1 - distance.cosine(train_vec.toarray().ravel(), test_doc_tfidf)
    doc_idx_sim_dict[idx] = cs_sim

sorted_docs = sorted(doc_idx_sim_dict.items(), key=lambda x: x[1], reverse=True)[:k]
print('\nTop K nearest documents:')
for doc_idx, sim_score in sorted_docs:
    print(f'Tài liệu [{doc_idx}] - chủ đề: [{y_train.iloc[doc_idx]}], độ tương đồng: [{sim_score:.6f}]')

predicted_label = Counter(y_train.iloc[idx] for idx, _ in sorted_docs).most_common(1)[0][0]
print(f'Chủ đề/lớp dự đoán cho văn bản kiểm thử: [{predicted_label}]')
