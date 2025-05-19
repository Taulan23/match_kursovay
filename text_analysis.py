import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk

nltk.download('punkt')
nltk.download('stopwords')

def check_plagiarism(input_text, texts_db):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([input_text] + texts_db)
    similarity_matrix = cosine_similarity(vectors)
    return similarity_matrix[0, 1:]

def generate_word_cloud(texts):
    text = ' '.join(texts)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    return wordcloud

def plot_text_length_distribution(texts):
    text_lengths = [len(text.split()) for text in texts]
    plt.figure(figsize=(10, 6))
    sns.histplot(text_lengths, kde=True)
    plt.xlabel("Длина текста (количество слов)")
    plt.ylabel("Частота")
    plt.title("Распределение длин текстов")
    return plt.gcf()

def analyze_text_complexity(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    avg_sentence_length = len(words) / len(sentences)
    avg_word_length = sum(len(word) for word in words) / len(words)
    unique_words = len(set(words))
    lexical_diversity = unique_words / len(words)
    
    return {
        "avg_sentence_length": avg_sentence_length,
        "avg_word_length": avg_word_length,
        "unique_words": unique_words,
        "lexical_diversity": lexical_diversity
    }

def analyze_readability(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    syllables = sum(count_syllables(word) for word in words)
    
    # Flesch Reading Ease
    fre = 206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (syllables / len(words))
    
    # Flesch-Kincaid Grade Level
    fkgl = 0.39 * (len(words) / len(sentences)) + 11.8 * (syllables / len(words)) - 15.59
    
    return {
        "flesch_reading_ease": fre,
        "flesch_kincaid_grade_level": fkgl
    }

def count_syllables(word):
    # This is a simple syllable counter and may not be entirely accurate for all words
    return len(re.findall('[aeiouаеёиоуыэюя]', word.lower()))

def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment
    return {
        "polarity": sentiment.polarity,
        "subjectivity": sentiment.subjectivity
    }

def extract_keywords(text, top_n=10):
    stop_words = set(stopwords.words('russian'))
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum() and word not in stop_words]
    
    freq_dist = nltk.FreqDist(words)
    return freq_dist.most_common(top_n)

def analyze_text_structure(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    paragraphs = text.split('\n\n')
    
    return {
        "num_sentences": len(sentences),
        "num_words": len(words),
        "num_paragraphs": len(paragraphs),
        "avg_words_per_sentence": len(words) / len(sentences),
        "avg_sentences_per_paragraph": len(sentences) / len(paragraphs)
    }

def perform_pos_tagging(text):
    nltk.download('averaged_perceptron_tagger')
    words = word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    return pos_tags

def extract_named_entities(text):
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
    words = word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    named_entities = nltk.ne_chunk(pos_tags)
    return named_entities

def analyze_word_frequency(text):
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('russian'))
    words = [word for word in words if word.isalnum() and word not in stop_words]
    freq_dist = nltk.FreqDist(words)
    return freq_dist

def generate_ngrams(text, n=2):
    words = word_tokenize(text.lower())
    ngrams = list(nltk.ngrams(words, n))
    return ngrams

def stem_text(text):
    stemmer = SnowballStemmer("russian")
    words = word_tokenize(text.lower())
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

def remove_stopwords(text):
    stop_words = set(stopwords.words('russian'))
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def calculate_text_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

def plot_word_frequency(text, top_n=20):
    freq_dist = analyze_word_frequency(text)
    top_words = freq_dist.most_common(top_n)
    words, frequencies = zip(*top_words)
    
    plt.figure(figsize=(12, 6))
    plt.bar(words, frequencies)
    plt.xlabel("Слова")
    plt.ylabel("Частота")
    plt.title(f"Топ-{top_n} наиболее частых слов")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return plt.gcf()

def analyze_sentence_length(text):
    sentences = sent_tokenize(text)
    sentence_lengths = [len(word_tokenize(sentence)) for sentence in sentences]
    return {
        "min_length": min(sentence_lengths),
        "max_length": max(sentence_lengths),
        "avg_length": sum(sentence_lengths) / len(sentence_lengths)
    }

def extract_phrases(text, phrase_length=3):
    words = word_tokenize(text.lower())
    phrases = [' '.join(words[i:i+phrase_length]) for i in range(len(words)-phrase_length+1)]
    return nltk.FreqDist(phrases)

# Добавьте другие функции анализа текста по необходимости
