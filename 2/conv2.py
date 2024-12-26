import pandas as pd
import re
from gensim.models import Word2Vec, FastText
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Загрузка данных
data = pd.read_csv("D:/ВУЗ/Атаева/Домашняя Работа/HomeWorkAnalyz/work_1/habr_articles.csv")
df = data.drop(columns=["Просмотры", "Время чтения", "Ссылка"]).head()

# Предобработка текста
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Zа-яА-Я0-9\s]', '', text.lower())  # Убираем спецсимволы и приводим к нижнему регистру
    return text

df["Текст"] = df["Текст"].fillna("").astype(str).apply(preprocess_text)

# Токенизация
tokenized_texts = df["Текст"].apply(lambda x: x.split())

# Обучение Word2Vec и FastText моделей
word2vec = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=2, workers=4)
fasttext = FastText(sentences=tokenized_texts, vector_size=100, window=5, min_count=2, workers=4)

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english', max_df=0.85, min_df=2)
tfidf_matrix = tfidf.fit_transform(df["Текст"])
tfidf_vocab = tfidf.get_feature_names_out()
tfidf_scores = {word: score for word, score in zip(tfidf_vocab, tfidf.idf_)}

# Функция для взвешенного векторного представления текста
def calculate_weighted_vector(tokens, model, tfidf_scores):
    vectors = [
        model.wv[word] * tfidf_scores[word] 
        for word in tokens if word in model.wv and word in tfidf_scores
    ]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

weighted_vectors_word2vec = tokenized_texts.apply(lambda x: calculate_weighted_vector(x, word2vec, tfidf_scores))
weighted_vectors_fasttext = tokenized_texts.apply(lambda x: calculate_weighted_vector(x, fasttext, tfidf_scores))

matrix_word2vec = np.array(weighted_vectors_word2vec.tolist())
matrix_fasttext = np.array(weighted_vectors_fasttext.tolist())

print("Размеры матриц:", matrix_word2vec.shape, matrix_fasttext.shape)

# Определение оптимального числа кластеров (локтевой метод)
def find_optimal_clusters(data, max_clusters=5):
    inertia_values = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(data)
        inertia_values.append(kmeans.inertia_)
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_clusters + 1), inertia_values, marker='o')
    plt.title("Оптимальное число кластеров")
    plt.xlabel("Число кластеров")
    plt.ylabel("Инерция")
    plt.show()

find_optimal_clusters(matrix_fasttext)

# Кластеризация
optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, n_init=10, random_state=42)
df["Кластер"] = kmeans.fit_predict(matrix_fasttext)

# Вывод информации о кластерах
print(df["Кластер"].value_counts())

# Примеры текстов из кластеров
for cluster in range(optimal_clusters):
    print(f"\nПримеры текстов из кластера {cluster}:")
    cluster_examples = df[df["Кластер"] == cluster].head(2)
    for _, row in cluster_examples.iterrows():
        print(f"- {row['Текст']}")