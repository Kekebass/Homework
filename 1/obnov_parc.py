import requests
from bs4 import BeautifulSoup as bs
import csv

base_url = "https://habr.com/ru/flows/develop/articles/"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
}

# Переменные для хранения данных
article_count = 0
max_pages = 50
collected_data = []

# Обход страниц
for page in range(1, max_pages + 1):
    page_url = f"{base_url}page{page}/" if page > 1 else base_url
    print(f"Собираем данные с: {page_url}")
    
    response = requests.get(page_url, headers=headers)
    if response.status_code != 200:
        print(f"Ошибка при запросе страницы {page_url}: код {response.status_code}")
        break
    
    soup = bs(response.text, "html.parser")
    articles = soup.find_all("article", class_="tm-articles-list__item")
    
    for article in articles:
        article_count += 1
        # Заголовок статьи
        title_tag = article.find("h2")
        title = title_tag.text.strip() if title_tag else "Заголовок отсутствует"
        
        # Время чтения
        reading_time_tag = article.find("span", class_="tm-article-reading-time__label")
        reading_time = reading_time_tag.text.strip() if reading_time_tag else "Не указано"
        
        # Количество просмотров
        views_tag = article.find("span", class_="tm-data-icons__item")
        views = views_tag.text.strip() if views_tag else "Просмотры неизвестны"
        
        # Ключевые слова
        keywords = [
            keyword.text.strip()
            for keyword in article.find_all("a", class_="tm-publication-hub__link")
        ]
        
        # Ссылка на статью
        link_tag = title_tag.find("a")
        article_url = "https://habr.com" + link_tag["href"] if link_tag else "Ссылка отсутствует"
        
        # Попытка получить текст статьи
        try:
            article_response = requests.get(article_url, headers=headers)
            if article_response.status_code == 200:
                article_soup = bs(article_response.text, "html.parser")
                first_paragraph_tag = article_soup.find("p")
                first_paragraph = first_paragraph_tag.text.strip() if first_paragraph_tag else "Текст отсутствует"
            else:
                first_paragraph = "Ошибка загрузки текста"
        except Exception as e:
            first_paragraph = f"Ошибка: {e}"
        
        # Вывод данных
        print(f"Статья {article_count}:")
        print(f"Название: {title}")
        print(f"Время чтения: {reading_time}")
        print(f"Просмотры: {views}")
        print(f"Текст: {first_paragraph}")
        print(f"Ключевые слова: {', '.join(keywords)}")
        print(f"Ссылка: {article_url}")
        print("-" * 50)
        
        # Добавляем данные в список
        collected_data.append({
            "Название": title,
            "Время чтения": reading_time,
            "Просмотры": views,
            "Текст": first_paragraph,
            "Ключевые слова": ", ".join(keywords),
            "Ссылка": article_url
        })

# Сохранение данных в CSV
csv_file = "habr_articles.csv"
with open(csv_file, mode="w", encoding="utf-8", newline="") as file:
    csv_writer = csv.DictWriter(file, fieldnames=["Название", "Время чтения", "Просмотры", "Текст", "Ключевые слова", "Ссылка"])
    csv_writer.writeheader()
    csv_writer.writerows(collected_data)

print(f"Сбор данных завершен. Всего статей: {article_count}. Данные сохранены в {csv_file}")