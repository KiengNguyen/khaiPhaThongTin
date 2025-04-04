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

# Load dataset tu thu muc data
data_dir = 'data/vnexpress/'  # Doi thanh duong dan thu muc cua ban
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

# Chia tap du lieu
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# 1. Bieu dien bang TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# 2. Bieu dien bang Word2Vec
w2v_model = Word2Vec(sentences=[t.split() for t in X_train], vector_size=100, window=5, min_count=1, workers=4)

def avg_word2vec(text, model):
    vectors = [model.wv[word] for word in text.split() if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

X_train_w2v = np.array([avg_word2vec(text, w2v_model) for text in X_train])
X_test_w2v = np.array([avg_word2vec(text, w2v_model) for text in X_test])

# 3. Bieu dien bang Doc2Vec
doc2vec_model = Doc2Vec([TaggedDocument(words=t.split(), tags=[str(i)]) for i, t in enumerate(X_train)],
                        vector_size=100, window=5, min_count=1, workers=4)

def doc2vec_vector(text, model):
    return model.infer_vector(text.split())

X_train_d2v = np.array([doc2vec_vector(text, doc2vec_model) for text in X_train])
X_test_d2v = np.array([doc2vec_vector(text, doc2vec_model) for text in X_test])

# Ham danh gia hieu suat mo hinh
def evaluate_model(model, X_train, X_test, name):
    start_time = time.time()
    model.fit(X_train, y_train_enc)
    y_pred = model.predict(X_test)
    elapsed_time = time.time() - start_time

    print(f'Bao cao phan lop cho {name}:')
    print(classification_report(y_test_enc, y_pred))
    print(f'Thời gian train: {elapsed_time:.4f} giay')
    return precision_score(y_test_enc, y_pred, average='weighted'), \
        recall_score(y_test_enc, y_pred, average='weighted'), \
        f1_score(y_test_enc, y_pred, average='weighted')

# Huan luyen va danh gia voi KNN
knn = KNeighborsClassifier(n_neighbors=3)
k_precision, k_recall, k_f1 = evaluate_model(knn, X_train_tfidf, X_test_tfidf, 'KNN voi TF-IDF')

# Huan luyen va danh gia voi Naive Bayes
nb = MultinomialNB()
nb_precision, nb_recall, nb_f1 = evaluate_model(nb, X_train_tfidf, X_test_tfidf, 'Naive Bayes voi TF-IDF')

# Huan luyen va danh gia voi Rocchio (SVM voi loss='hinge' mo phong Rocchio)
rocchio = SGDClassifier(loss='hinge', alpha=0.01, max_iter=1000, random_state=42)
r_precision, r_recall, r_f1 = evaluate_model(rocchio, X_train_tfidf, X_test_tfidf, 'Rocchio voi TF-IDF')

# So sanh ket qua
print('\nSo sanh cac phuong phap phan lop:')
print(f'{"Mo hinh":<20} {"Precision":<10} {"Recall":<10} {"F1-score":<10}')
print(f'{"KNN voi TF-IDF":<20} {k_precision:.4f} {k_recall:.4f} {k_f1:.4f}')
print(f'{"Naive Bayes voi TF-IDF":<20} {nb_precision:.4f} {nb_recall:.4f} {nb_f1:.4f}')
print(f'{"Rocchio voi TF-IDF":<20} {r_precision:.4f} {r_recall:.4f} {r_f1:.4f}')

# Xuat ma tran TF-IDF thanh file CSV
feature_names = tfidf_vectorizer.get_feature_names_out()
train_tfidf_df = pd.DataFrame(X_train_tfidf.toarray(), columns=feature_names)
train_tfidf_df.to_csv('tfidf_train.csv', index=False)
print('Da xuat ma tran TF-IDF thanh file tfidf_train.csv')
print("\n📊 Mot phan ma tran TF-IDF (5 dong dau):")
print(train_tfidf_df.head())

# Xac dinh K tai lieu gan nhat va chu de du doan
k = 5
while True:
    test_doc = input("🔍 Nhap van ban kiem thu: ")
    # Kiểm tra độ dài của văn bản
    if len(test_doc.split()) < 3:  # Ít nhất 3 từ
        print("Vui lòng nhập một văn bản có ít nhất 3 từ.")
    else:
        test_doc_tfidf = tfidf_vectorizer.transform([test_doc]).toarray().ravel()

        doc_idx_sim_dict = {}
        for idx, train_vec in enumerate(X_train_tfidf):
            cs_sim = 1 - distance.cosine(train_vec.toarray().ravel(), test_doc_tfidf)
            doc_idx_sim_dict[idx] = cs_sim

        sorted_docs = sorted(doc_idx_sim_dict.items(), key=lambda x: x[1], reverse=True)[:k]
        print('\nTop K tai lieu gan nhat:')
        for doc_idx, sim_score in sorted_docs:
            print(f'Tai lieu [{doc_idx}] - chu de: [{y_train.iloc[doc_idx]}], do tuong dong: [{sim_score:.6f}]')

        predicted_label = Counter(y_train.iloc[idx] for idx, _ in sorted_docs).most_common(1)[0][0]
        print(f'Chu de/ lop du doan cho van ban kiem thu: [{predicted_label}]')
