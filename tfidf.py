# tfidf.py

from sklearn.feature_extraction.text import TfidfVectorizer

def compute_tfidf_vectors(documents, max_features=1000):
    """
    Nhận vào danh sách văn bản, trả về ma trận TF-IDF dạng (n_samples, n_features)
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(documents)
    return X
