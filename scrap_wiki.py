import wikipedia
import random
import time

def download_wikipedia_articles(num_articles):
    articles = []
    for _ in range(num_articles):
        article_title = wikipedia.random()
        try:
            article = wikipedia.page(article_title).content
            articles.append(article)
            print(f"Fetched article: {article_title}")
        except wikipedia.exceptions.DisambiguationError as e:
            print(f"Disambiguation error: {e}")
        except wikipedia.exceptions.PageError as e:
            print(f"Page error: {e}")
        time.sleep(1)  # wait 1 second between requests
    return "\n\n".join(articles)

num_articles = 10  # fetch 10 articles for testing
text = download_wikipedia_articles(num_articles)

with open("wikipedia_articles.txt", "w", encoding="utf-8") as f:
    f.write(text)
print("Articles saved to wikipedia_articles.txt")