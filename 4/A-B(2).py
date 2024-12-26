import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD, NMF

# Настройка NLTK
nltk.data.path.append('C:/Users/Slyexistence/nltk_data')
nltk.download('stopwords', download_dir='C:/Users/Slyexistence/AppData/Roaming/nltk_data')

# Загрузка данных
data_path = "D:/ВУЗ/Атаева/Домашняя Работа/HomeWorkAnalyz/work_1/habr_articles.csv"
data = pd.read_csv(data_path)
texts = data['Текст'].dropna().tolist()

# Настройка обработки текста
stop_words = set(stopwords.words('russian'))
tokenizer = RegexpTokenizer(r'\w+')

def clean_text(text):
    """Очищает текст от спецсимволов, приводит к нижнему регистру и удаляет стоп-слова."""
    text = re.sub(r'[^\w\s]', '', text.lower())
    words = tokenizer.tokenize(text)
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

# Предобработка текстов
processed_texts = list(map(clean_text, texts))

# Преобразование текстов в матрицу частот
count_vectorizer = CountVectorizer(max_df=0.85, min_df=10)
count_matrix = count_vectorizer.fit_transform(processed_texts)

# LDA (Latent Dirichlet Allocation)
lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
lda_model.fit(count_matrix)

print("Темы, обнаруженные LDA:")
for idx, topic in enumerate(lda_model.components_):
    top_words = [count_vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]
    print(f"Тема {idx + 1}: {top_words}")

# TF-IDF для дальнейших методов анализа
tfidf_vectorizer = TfidfVectorizer(max_df=0.85, min_df=10)
tfidf_matrix = tfidf_vectorizer.fit_transform(processed_texts)

# LSA (Latent Semantic Analysis)
lsa_model = TruncatedSVD(n_components=5, random_state=42)
lsa_model.fit(tfidf_matrix)

print("\nТемы, обнаруженные LSA:")
for idx, topic in enumerate(lsa_model.components_):
    top_words = [tfidf_vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]
    print(f"Тема {idx + 1}: {top_words}")

# NMF (Non-Negative Matrix Factorization)
nmf_model = NMF(n_components=5, random_state=42)
nmf_model.fit(tfidf_matrix)

print("\nТемы, обнаруженные NMF:")
for idx, topic in enumerate(nmf_model.components_):
    top_words = [tfidf_vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]
    print(f"Тема {idx + 1}: {top_words}")