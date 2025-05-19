import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QTextEdit, QFileDialog, QMessageBox, 
                             QTabWidget, QComboBox, QProgressBar, QListWidget, QSplitter,
                             QTreeView, QHeaderView, QAbstractItemView, QLineEdit, QCheckBox,
                             QStyleFactory, QDialog, QFormLayout, QRadioButton, QGroupBox,
                             QSpinBox, QDoubleSpinBox, QTableWidget, QTableWidgetItem)
from PyQt5.QtGui import QFont, QColor, QPalette, QStandardItemModel, QStandardItem, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSortFilterProxyModel
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_curve, roc_curve, auc
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.manifold import TSNE
from gensim import corpora
from gensim.models import LdaModel, Word2Vec
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk
import re
import time
import json
import pickle
from collections import Counter
from itertools import cycle
from scipy.interpolate import interp1d
import seaborn as sns
from wordcloud import WordCloud
import speech_recognition as sr
import requests
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import matplotlib.animation as animation
from textblob import TextBlob
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import networkx as nx

class AdvancedTextAnalysis:
    def __init__(self):
        self.nlp = None  # Placeholder for spaCy model
        self.word2vec_model = None
        self.lda_model = None

    def load_spacy_model(self):
        import spacy
        self.nlp = spacy.load("ru_core_news_sm")

    def named_entity_recognition(self, text):
        if not self.nlp:
            self.load_spacy_model()
        doc = self.nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities

    def dependency_parsing(self, text):
        if not self.nlp:
            self.load_spacy_model()
        doc = self.nlp(text)
        dependencies = [(token.text, token.dep_, token.head.text) for token in doc]
        return dependencies

    def train_word2vec(self, texts, vector_size=100, window=5, min_count=1):
        tokenized_texts = [word_tokenize(text.lower()) for text in texts]
        self.word2vec_model = Word2Vec(sentences=tokenized_texts, vector_size=vector_size, window=window, min_count=min_count)

    def get_word_embeddings(self, word):
        if not self.word2vec_model:
            raise ValueError("Word2Vec model not trained. Call train_word2vec first.")
        return self.word2vec_model.wv[word]

    def find_similar_words(self, word, topn=10):
        if not self.word2vec_model:
            raise ValueError("Word2Vec model not trained. Call train_word2vec first.")
        return self.word2vec_model.wv.most_similar(word, topn=topn)

    def train_lda_model(self, texts, num_topics=5):
        tokenized_texts = [word_tokenize(text.lower()) for text in texts]
        dictionary = corpora.Dictionary(tokenized_texts)
        corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
        self.lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42)

    def get_document_topics(self, text):
        if not self.lda_model:
            raise ValueError("LDA model not trained. Call train_lda_model first.")
        bow = self.lda_model.id2word.doc2bow(word_tokenize(text.lower()))
        return self.lda_model.get_document_topics(bow)

class AdvancedVisualization:
    def __init__(self):
        self.figure = plt.figure(figsize=(10, 6))
        self.ax = self.figure.add_subplot(111)

    def plot_word_frequencies(self, text, top_n=20):
        words = word_tokenize(text.lower())
        word_freq = Counter(words)
        top_words = dict(word_freq.most_common(top_n))
        
        self.ax.clear()
        self.ax.bar(top_words.keys(), top_words.values())
        self.ax.set_xlabel('Words')
        self.ax.set_ylabel('Frequency')
        self.ax.set_title(f'Top {top_n} Most Frequent Words')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

    def plot_sentiment_distribution(self, texts):
        sentiments = [TextBlob(text).sentiment.polarity for text in texts]
        
        self.ax.clear()
        self.ax.hist(sentiments, bins=20, edgecolor='black')
        self.ax.set_xlabel('Sentiment Polarity')
        self.ax.set_ylabel('Frequency')
        self.ax.set_title('Sentiment Distribution')
        plt.tight_layout()

    def plot_topic_distribution(self, lda_model, num_topics):
        topic_weights = [topic[1] for topic in lda_model.print_topics(num_topics)]
        topics = range(1, num_topics + 1)
        
        self.ax.clear()
        self.ax.bar(topics, topic_weights)
        self.ax.set_xlabel('Topics')
        self.ax.set_ylabel('Weight')
        self.ax.set_title('Topic Distribution')
        plt.tight_layout()

    def plot_tsne_cluster(self, vectors, labels):
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(vectors)
        
        self.ax.clear()
        scatter = self.ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis')
        self.ax.set_xlabel('t-SNE 1')
        self.ax.set_ylabel('t-SNE 2')
        self.ax.set_title('t-SNE Visualization of Clusters')
        plt.colorbar(scatter)
        plt.tight_layout()

class TextGenerationModels:
    def __init__(self):
        self.markov_chain = None
        self.lstm_model = None
        self.transformer_model = None

    def train_markov_chain(self, text, n=2):
        words = text.split()
        self.markov_chain = {}
        for i in range(len(words) - n):
            state = tuple(words[i:i+n])
            next_word = words[i+n]
            if state not in self.markov_chain:
                self.markov_chain[state] = {}
            if next_word not in self.markov_chain[state]:
                self.markov_chain[state][next_word] = 0
            self.markov_chain[state][next_word] += 1

    def generate_text_markov_chain(self, start_words, length=100):
        if not self.markov_chain:
            raise ValueError("Markov chain not trained. Call train_markov_chain first.")
        
        n = len(list(self.markov_chain.keys())[0])
        current_state = tuple(start_words.split()[-n:])
        result = list(current_state)
        
        for _ in range(length):
            if current_state not in self.markov_chain:
                break
            next_word = max(self.markov_chain[current_state], key=self.markov_chain[current_state].get)
            result.append(next_word)
            current_state = tuple(result[-n:])
        
        return ' '.join(result)

    def train_lstm(self, texts):
        # Placeholder for LSTM model training
        # This would typically involve tokenization, padding, and training a Keras/TensorFlow model
        pass

    def generate_text_lstm(self, start_text, length=100):
        # Placeholder for LSTM text generation
        pass

    def train_transformer(self, texts):
        # Placeholder for Transformer model training
        # This would typically involve using a library like Hugging Face's transformers
        pass

    def generate_text_transformer(self, start_text, length=100):
        # Placeholder for Transformer text generation
        pass

class AdvancedPreprocessing:
    def __init__(self):
        self.stemmer = SnowballStemmer("russian")
        self.stop_words = set(stopwords.words("russian"))

    def preprocess_text(self, text, remove_stopwords=True, stem=True):
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Stem
        if stem:
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        return ' '.join(tokens)

    def extract_ngrams(self, text, n=2):
        tokens = word_tokenize(text)
        ngrams = zip(*[tokens[i:] for i in range(n)])
        return [' '.join(ngram) for ngram in ngrams]

    def extract_noun_phrases(self, text):
        if not hasattr(self, 'nlp'):
            import spacy
            self.nlp = spacy.load("ru_core_news_sm")
        
        doc = self.nlp(text)
        return [chunk.text for chunk in doc.noun_chunks]

    def lemmatize_text(self, text):
        if not hasattr(self, 'nlp'):
            import spacy
            self.nlp = spacy.load("ru_core_news_sm")
        
        doc = self.nlp(text)
        return ' '.join([token.lemma_ for token in doc])

class AdvancedFeatureExtraction:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.count_vectorizer = None

    def extract_tfidf_features(self, texts, max_features=1000):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
        return self.tfidf_vectorizer.fit_transform(texts)

    def extract_bow_features(self, texts, max_features=1000):
        self.count_vectorizer = CountVectorizer(max_features=max_features)
        return self.count_vectorizer.fit_transform(texts)

    def extract_pos_features(self, texts):
        if not hasattr(self, 'nlp'):
            import spacy
            self.nlp = spacy.load("ru_core_news_sm")
        
        pos_features = []
        for text in texts:
            doc = self.nlp(text)
            pos_counts = Counter([token.pos_ for token in doc])
            pos_features.append(pos_counts)
        
        return pd.DataFrame(pos_features)

    def extract_named_entity_features(self, texts):
        if not hasattr(self, 'nlp'):
            import spacy
            self.nlp = spacy.load("ru_core_news_sm")
        
        ne_features = []
        for text in texts:
            doc = self.nlp(text)
            ne_counts = Counter([ent.label_ for ent in doc.ents])
            ne_features.append(ne_counts)
        
        return pd.DataFrame(ne_features)

class AdvancedModelTraining:
    def __init__(self):
        self.model = None
        self.vectorizer = None

    def train_model(self, X, y, model_type='svm', vectorizer_type='tfidf'):
        # Vectorize the text data
        if vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer()
        elif vectorizer_type == 'count':
            self.vectorizer = CountVectorizer()
        else:
            raise ValueError("Invalid vectorizer type. Choose 'tfidf' or 'count'.")
        
        X_vectorized = self.vectorizer.fit_transform(X)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
        
        # Train the model
        if model_type == 'svm':
            self.model = SVC(probability=True)
        elif model_type == 'naive_bayes':
            self.model = MultinomialNB()
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression()
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier()
        else:
            raise ValueError("Invalid model type. Choose 'svm', 'naive_bayes', 'logistic_regression', or 'random_forest'.")
        
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        return accuracy, report

    def predict(self, text):
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained. Call train_model first.")
        
        X_vectorized = self.vectorizer.transform([text])
        prediction = self.model.predict(X_vectorized)
        probabilities = self.model.predict_proba(X_vectorized)
        
        return prediction[0], probabilities[0]

    def perform_grid_search(self, X, y, param_grid):
        X_vectorized = self.vectorizer.fit_transform(X)
        grid_search = GridSearchCV(
            self.model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_vectorized, y)
        
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        return best_params, best_score

class AdvancedTextSummarization:
    def __init__(self):
        self.summarizer = None

    def extractive_summarization(self, text, sentences_count=3, algorithm='lsa'):
        parser = PlaintextParser.from_string(text, Tokenizer("russian"))
        
        if algorithm == 'lsa':
            self.summarizer = LsaSummarizer()
        elif algorithm == 'textrank':
            self.summarizer = TextRankSummarizer()
        elif algorithm == 'lexrank':
            self.summarizer = LexRankSummarizer()
        else:
            raise ValueError("Invalid algorithm. Choose 'lsa', 'textrank', or 'lexrank'.")
        
        summary = self.summarizer(parser.document, sentences_count)
        return " ".join([str(sentence) for sentence in summary])

    def abstractive_summarization(self, text, max_length=150, min_length=50):
        # This is a placeholder for abstractive summarization
        # In a real implementation, you would use a pre-trained model like T5 or BART
        # For now, we'll just return a simple extractive summary
        return self.extractive_summarization(text, sentences_count=3)

class AdvancedTextClustering:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.model = None

    def preprocess_and_vectorize(self, texts):
        return self.vectorizer.fit_transform(texts)

    def perform_kmeans_clustering(self, texts, n_clusters=5):
        X = self.preprocess_and_vectorize(texts)
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = self.model.fit_predict(X)
        return labels

    def perform_dbscan_clustering(self, texts, eps=0.5, min_samples=5):
        X = self.preprocess_and_vectorize(texts)
        self.model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = self.model.fit_predict(X)
        return labels

    def perform_agglomerative_clustering(self, texts, n_clusters=5):
        X = self.preprocess_and_vectorize(texts)
        self.model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = self.model.fit_predict(X.toarray())
        return labels

    def get_cluster_centers(self):
        if isinstance(self.model, KMeans):
            centers = self.model.cluster_centers_
            return self.vectorizer.inverse_transform(centers)
        else:
            raise ValueError("Cluster centers are only available for KMeans clustering.")

class AdvancedTopicModeling:
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.model = None

    def preprocess_and_vectorize(self, texts):
        return self.vectorizer.fit_transform(texts)

    def perform_lda(self, texts, n_topics=5, random_state=42):
        X = self.preprocess_and_vectorize(texts)
        self.model = LatentDirichletAllocation(n_components=n_topics, random_state=random_state)
        self.model.fit(X)
        return self.model

    def perform_nmf(self, texts, n_topics=5, random_state=42):
        X = self.preprocess_and_vectorize(texts)
        self.model = NMF(n_components=n_topics, random_state=random_state)
        self.model.fit(X)
        return self.model

    def print_top_words(self, n_top_words=10):
        feature_names = self.vectorizer.get_feature_names_out()
        for topic_idx, topic in enumerate(self.model.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
            print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")

    def get_document_topics(self, text):
        X = self.vectorizer.transform([text])
        return self.model.transform(X)[0]

class AdvancedSentimentAnalysis:
    def __init__(self):
        self.nlp = None

    def load_spacy_model(self):
        import spacy
        self.nlp = spacy.load("ru_core_news_sm")

    def analyze_sentiment(self, text):
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        if polarity > 0:
            sentiment = "Positive"
        elif polarity < 0:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        
        return {
            "sentiment": sentiment,
            "polarity": polarity,
            "subjectivity": subjectivity
        }

    def analyze_sentence_sentiments(self, text):
        if not self.nlp:
            self.load_spacy_model()
        
        doc = self.nlp(text)
        sentence_sentiments = []
        
        for sent in doc.sents:
            sentiment = self.analyze_sentiment(sent.text)
            sentence_sentiments.append({
                "sentence": sent.text,
                "sentiment": sentiment
            })
        
        return sentence_sentiments

    def get_emotion_lexicon(self):
        # This is a placeholder for a more comprehensive emotion lexicon
        return {
            "радость": ["счастливый", "радостный", "веселый"],
            "грусть": ["печальный", "грустный", "унылый"],
            "гнев": ["злой", "раздраженный", "возмущенный"],
            "страх": ["испуганный", "тревожный", "обеспокоенный"],
            "удивление": ["удивленный", "изумленный", "пораженный"]
        }

    def analyze_emotions(self, text):
        if not self.nlp:
            self.load_spacy_model()
        
        doc = self.nlp(text)
        emotion_lexicon = self.get_emotion_lexicon()
        emotion_counts = {emotion: 0 for emotion in emotion_lexicon}
        
        for token in doc:
            for emotion, words in emotion_lexicon.items():
                if token.lemma_.lower() in words:
                    emotion_counts[emotion] += 1
        
        total_emotions = sum(emotion_counts.values())
        emotion_percentages = {emotion: count / total_emotions * 100 if total_emotions > 0 else 0 
                               for emotion, count in emotion_counts.items()}
        
        return emotion_percentages

class AdvancedReadabilityAnalysis:
    def __init__(self):
        self.nlp = None

    def load_spacy_model(self):
        import spacy
        self.nlp = spacy.load("ru_core_news_sm")

    def calculate_readability_metrics(self, text):
        if not self.nlp:
            self.load_spacy_model()
        
        doc = self.nlp(text)
        
        # Basic text statistics
        num_sentences = len(list(doc.sents))
        num_words = len([token for token in doc if not token.is_punct])
        num_syllables = sum([self.count_syllables(token.text) for token in doc if not token.is_punct])
        num_characters = len(text)
        
        # Calculate various readability scores
        flesch_kincaid_grade = self.flesch_kincaid_grade(num_sentences, num_words, num_syllables)
        flesch_reading_ease = self.flesch_reading_ease(num_sentences, num_words, num_syllables)
        gunning_fog = self.gunning_fog(doc)
        smog = self.smog(doc)
        coleman_liau = self.coleman_liau(num_sentences, num_words, num_characters)
        
        return {
            "flesch_kincaid_grade": flesch_kincaid_grade,
            "flesch_reading_ease": flesch_reading_ease,
            "gunning_fog": gunning_fog,
            "smog": smog,
            "coleman_liau": coleman_liau
        }

    def count_syllables(self, word):
        # This is a simple syllable counter and may not be entirely accurate for all Russian words
        return len([char for char in word if char.lower() in 'аеёиоуыэюя'])

    def flesch_kincaid_grade(self, num_sentences, num_words, num_syllables):
        return 0.39 * (num_words / num_sentences) + 11.8 * (num_syllables / num_words) - 15.59

    def flesch_reading_ease(self, num_sentences, num_words, num_syllables):
        return 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (num_syllables / num_words)

    def gunning_fog(self, doc):
        complex_words = len([token for token in doc if len(token.text) > 6])
        num_sentences = len(list(doc.sents))
        num_words = len([token for token in doc if not token.is_punct])
        return 0.4 * ((num_words / num_sentences) + 100 * (complex_words / num_words))

    def smog(self, doc):
        complex_words = len([token for token in doc if len(token.text) > 6])
        num_sentences = len(list(doc.sents))
        return 1.043 * ((complex_words * (30 / num_sentences)) ** 0.5) + 3.1291

    def coleman_liau(self, num_sentences, num_words, num_characters):
        L = (num_characters / num_words) * 100
        S = (num_sentences / num_words) * 100
        return 0.0588 * L - 0.296 * S - 15.8

class AdvancedTextVisualization:
    def __init__(self):
        self.nlp = None

    def load_spacy_model(self):
        import spacy
        self.nlp = spacy.load("ru_core_news_sm")

    def generate_word_cloud(self, text):
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud')
        plt.show()

    def plot_top_n_grams(self, text, n=2, top_k=20):
        if not self.nlp:
            self.load_spacy_model()
        
        doc = self.nlp(text)
        n_grams = self.get_n_grams(doc, n)
        top_n_grams = dict(sorted(n_grams.items(), key=lambda x: x[1], reverse=True)[:top_k])
        
        plt.figure(figsize=(12, 6))
        plt.bar(top_n_grams.keys(), top_n_grams.values())
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Top {top_k} {n}-grams')
        plt.xlabel(f'{n}-grams')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

    def get_n_grams(self, doc, n):
        n_grams = Counter()
        for sent in doc.sents:
            tokens = [token.text.lower() for token in sent if not token.is_punct]
            n_grams.update(zip(*[tokens[i:] for i in range(n)]))
        return n_grams

    def plot_pos_distribution(self, text):
        if not self.nlp:
            self.load_spacy_model()
        
        doc = self.nlp(text)
        pos_counts = Counter([token.pos_ for token in doc])
        
        plt.figure(figsize=(10, 5))
        plt.bar(pos_counts.keys(), pos_counts.values())
        plt.title('Distribution of Parts of Speech')
        plt.xlabel('Part of Speech')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def plot_named_entity_distribution(self, text):
        if not self.nlp:
            self.load_spacy_model()
        
        doc = self.nlp(text)
        ne_counts = Counter([ent.label_ for ent in doc.ents])
        
        plt.figure(figsize=(10, 5))
        plt.bar(ne_counts.keys(), ne_counts.values())
        plt.title('Distribution of Named Entities')
        plt.xlabel('Named Entity Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

class AdvancedPlagiarismDetection:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def preprocess_texts(self, texts):
        return [self.preprocess_text(text) for text in texts]

    def preprocess_text(self, text):
        # Convert to lowercase and remove punctuation
        text = re.sub(r'[^\w\s]', '', text.lower())
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords
        stop_words = set(stopwords.words('russian'))
        tokens = [token for token in tokens if token not in stop_words]
        return ' '.join(tokens)

    def calculate_similarity(self, text1, text2):
        preprocessed_texts = self.preprocess_texts([text1, text2])
        tfidf_matrix = self.vectorizer.fit_transform(preprocessed_texts)
        return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    def detect_plagiarism(self, suspect_text, reference_texts, threshold=0.8):
        suspect_text = self.preprocess_text(suspect_text)
        reference_texts = self.preprocess_texts(reference_texts)
        
        all_texts = [suspect_text] + reference_texts
        tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
        
        plagiarism_results = []
        for i, similarity in enumerate(similarities):
            if similarity >= threshold:
                plagiarism_results.append({
                    "reference_index": i,
                    "similarity": similarity
                })
        
        return plagiarism_results

    def highlight_similar_passages(self, suspect_text, reference_text, window_size=5):
        suspect_tokens = word_tokenize(suspect_text)
        reference_tokens = word_tokenize(reference_text)
        
        similar_passages = []
        for i in range(len(suspect_tokens) - window_size + 1):
            suspect_window = suspect_tokens[i:i+window_size]
            for j in range(len(reference_tokens) - window_size + 1):
                reference_window = reference_tokens[j:j+window_size]
                if suspect_window == reference_window:
                    similar_passages.append({
                        "suspect_start": i,
                        "suspect_end": i + window_size,
                        "reference_start": j,
                        "reference_end": j + window_size
                    })
        
        return similar_passages

class AdvancedTextGeneration:
    def __init__(self):
        self.model = None

    def load_gpt2_model(self):
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def generate_text(self, prompt, max_length=100, temperature=1.0, top_k=50, top_p=0.95):
        if not self.model:
            self.load_gpt2_model()

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        output = self.model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True
        )

        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text

class AdvancedTextClassification:
    def __init__(self):
        self.model = None
        self.vectorizer = None

    def train_model(self, texts, labels, model_type='svm'):
        self.vectorizer = TfidfVectorizer()
        X = self.vectorizer.fit_transform(texts)
        
        if model_type == 'svm':
            self.model = SVC(probability=True)
        elif model_type == 'naive_bayes':
            self.model = MultinomialNB()
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression()
        else:
            raise ValueError("Invalid model type. Choose 'svm', 'naive_bayes', or 'logistic_regression'.")
        
        self.model.fit(X, labels)

    def classify_text(self, text):
        if not self.model or not self.vectorizer:
            raise ValueError("Model not trained. Call train_model first.")
        
        X = self.vectorizer.transform([text])
        prediction = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        return prediction[0], probabilities[0]

    def evaluate_model(self, test_texts, test_labels):
        if not self.model or not self.vectorizer:
            raise ValueError("Model not trained. Call train_model first.")
        
        X_test = self.vectorizer.transform(test_texts)
        y_pred = self.model.predict(X_test)
        
        accuracy = accuracy_score(test_labels, y_pred)
        report = classification_report(test_labels, y_pred)
        
        return accuracy, report

class AdvancedNamedEntityRecognition:
    def __init__(self):
        self.nlp = None

    def load_spacy_model(self):
        import spacy
        self.nlp = spacy.load("ru_core_news_sm")

    def recognize_entities(self, text):
        if not self.nlp:
            self.load_spacy_model()
        
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
                "label": ent.label_
            })
        return entities

    def visualize_entities(self, text):
        if not self.nlp:
            self.load_spacy_model()
        
        doc = self.nlp(text)
        from spacy import displacy
        displacy.render(doc, style="ent", jupyter=True)

class AdvancedSyntacticAnalysis:
    def __init__(self):
        self.nlp = None

    def load_spacy_model(self):
        import spacy
        self.nlp = spacy.load("ru_core_news_sm")

    def analyze_syntax(self, text):
        if not self.nlp:
            self.load_spacy_model()
        
        doc = self.nlp(text)
        analysis = []
        for token in doc:
            analysis.append({
                "text": token.text,
                "lemma": token.lemma_,
                "pos": token.pos_,
                "tag": token.tag_,
                "dep": token.dep_,
                "shape": token.shape_,
                "is_alpha": token.is_alpha,
                "is_stop": token.is_stop
            })
        return analysis

    def visualize_dependency_tree(self, text):
        if not self.nlp:
            self.load_spacy_model()
        
        doc = self.nlp(text)
        from spacy import displacy
        displacy.render(doc, style="dep", jupyter=True)

class AdvancedTextSummarization:
    def __init__(self):
        self.nlp = None

    def load_spacy_model(self):
        import spacy
        self.nlp = spacy.load("ru_core_news_sm")

    def extractive_summarization(self, text, num_sentences=3):
        if not self.nlp:
            self.load_spacy_model()
        
        doc = self.nlp(text)
        sentences = [sent.text for sent in doc.sents]
        
        # Calculate sentence scores based on word frequency
        word_freq = Counter([token.text.lower() for token in doc if not token.is_stop and token.is_alpha])
        sentence_scores = []
        for sentence in sentences:
            score = sum(word_freq[word.lower()] for word in word_tokenize(sentence) if word.lower() in word_freq)
            sentence_scores.append((sentence, score))
        
        # Sort sentences by score and select top N
        summary_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:num_sentences]
        summary = " ".join([sentence for sentence, score in summary_sentences])
        
        return summary

    def abstractive_summarization(self, text, max_length=150, min_length=50):
        from transformers import pipeline
        summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
        return summary

class AdvancedSentimentAnalysis:
    def __init__(self):
        self.nlp = None
        self.sia = None

    def load_spacy_model(self):
        import spacy
        self.nlp = spacy.load("ru_core_news_sm")

    def load_vader_sentiment_analyzer(self):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self.sia = SentimentIntensityAnalyzer()

    def analyze_sentiment(self, text):
        if not self.nlp:
            self.load_spacy_model()
        if not self.sia:
            self.load_vader_sentiment_analyzer()
        
        doc = self.nlp(text)
        overall_sentiment = self.sia.polarity_scores(text)
        
        sentence_sentiments = []
        for sent in doc.sents:
            sentiment = self.sia.polarity_scores(sent.text)
            sentence_sentiments.append({
                "sentence": sent.text,
                "sentiment": sentiment
            })
        
        return {
            "overall_sentiment": overall_sentiment,
            "sentence_sentiments": sentence_sentiments
        }

    def visualize_sentiment(self, text):
        analysis = self.analyze_sentiment(text)
        
        # Visualize overall sentiment
        overall = analysis["overall_sentiment"]
        plt.figure(figsize=(10, 5))
        plt.bar(overall.keys(), overall.values())
        plt.title("Overall Sentiment")
        plt.show()
        
        # Visualize sentence sentiments
        sentences = [s["sentence"] for s in analysis["sentence_sentiments"]]
        compound_scores = [s["sentiment"]["compound"] for s in analysis["sentence_sentiments"]]
        
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(sentences)), compound_scores, marker='o')
        plt.title("Sentence-level Sentiment")
        plt.xlabel("Sentence")
        plt.ylabel("Compound Sentiment Score")
        plt.xticks(range(len(sentences)), sentences, rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

class AdvancedTopicModeling:
    def __init__(self):
        self.lda_model = None
        self.dictionary = None
        self.corpus = None

    def preprocess_texts(self, texts):
        nltk.download('punkt')
        nltk.download('stopwords')
        stop_words = set(stopwords.words('russian'))
        tokenized_texts = [word_tokenize(text.lower()) for text in texts]
        tokenized_texts = [[token for token in text if token not in stop_words] for text in tokenized_texts]
        return tokenized_texts

    def train_lda_model(self, texts, num_topics=5):
        tokenized_texts = self.preprocess_texts(texts)
        self.dictionary = corpora.Dictionary(tokenized_texts)
        self.corpus = [self.dictionary.doc2bow(text) for text in tokenized_texts]
        self.lda_model = LdaModel(corpus=self.corpus, id2word=self.dictionary, num_topics=num_topics, random_state=42)

    def get_document_topics(self, text):
        if not self.lda_model:
            raise ValueError("LDA model not trained. Call train_lda_model first.")
        bow = self.dictionary.doc2bow(word_tokenize(text.lower()))
        return self.lda_model.get_document_topics(bow)

    def print_top_words(self, num_words=10):
        if not self.lda_model:
            raise ValueError("LDA model not trained. Call train_lda_model first.")
        for topic_idx, topic in self.lda_model.print_topics(num_words=num_words):
            print(f"Topic {topic_idx + 1}: {topic}")

    def visualize_topics(self, num_topics=5, num_words=10):
        if not self.lda_model:
            raise ValueError("LDA model not trained. Call train_lda_model first.")
        topics = self.lda_model.print_topics(num_topics=num_topics, num_words=num_words)
        for topic in topics:
            print(topic)

def perform_advanced_analysis(texts, analysis_type):
    if analysis_type == "Облако слов":
        return word_cloud_analysis(texts)
    elif analysis_type == "Распределение длин текстов":
        return text_length_distribution(texts)
    elif analysis_type == "Анализ сложности текста":
        return text_complexity_analysis(texts)
    elif analysis_type == "Оценка читаемости":
        return readability_analysis(texts)
    elif analysis_type == "Анализ тональности":
        return sentiment_analysis(texts)
    elif analysis_type == "Извлечение ключевых слов":
        return keyword_extraction(texts)
    elif analysis_type == "Анализ структуры текста":
        return text_structure_analysis(texts)
    elif analysis_type == "Тематическое моделирование":
        return topic_modeling(texts)
    elif analysis_type == "Анализ стиля письма":
        return writing_style_analysis(texts)
    elif analysis_type == "Обнаружение повторного использования текста":
        return detect_text_reuse(texts)
    elif analysis_type == "Анализ паттернов цитирования":
        return citation_pattern_analysis(texts)
    else:
        return {"text": "Выбранный тип анализа не реализован"}

def word_cloud_analysis(texts):
    text = ' '.join(texts)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    def plot_wordcloud(ax):
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title("Облако слов")
    
    return {
        "text": "Облако слов создано на основе всех текстов.",
        "figure": plot_wordcloud
    }

def text_length_distribution(texts):
    lengths = [len(text.split()) for text in texts]
    
    def plot_distribution(ax):
        ax.hist(lengths, bins=20, edgecolor='black')
        ax.set_xlabel("Длина текста (количество слов)")
        ax.set_ylabel("Частота")
        ax.set_title("Распределение длин текстов")
    
    return {
        "text": f"Средняя длина текста: {np.mean(lengths):.2f} слов\n"
                f"Медианная длина текста: {np.median(lengths):.2f} слов\n"
                f"Минимальная длина: {min(lengths)} слов\n"
                f"Максимальная длина: {max(lengths)} слов",
        "figure": plot_distribution
    }

def text_complexity_analysis(texts):
    # Простой анализ сложности на основе длины слов и предложений
    avg_word_lengths = []
    avg_sentence_lengths = []
    
    for text in texts:
        words = text.split()
        sentences = text.split('.')
        avg_word_lengths.append(np.mean([len(word) for word in words]))
        avg_sentence_lengths.append(np.mean([len(sentence.split()) for sentence in sentences if sentence.strip()]))
    
    def plot_complexity(ax):
        ax.scatter(avg_word_lengths, avg_sentence_lengths)
        ax.set_xlabel("Средняя длина слова")
        ax.set_ylabel("Средняя длина предложения")
        ax.set_title("Анализ сложности текста")
    
    return {
        "text": f"Средняя длина слова: {np.mean(avg_word_lengths):.2f} символов\n"
                f"Средняя длина предложения: {np.mean(avg_sentence_lengths):.2f} слов",
        "figure": plot_complexity
    }

def readability_analysis(texts):
    # Простой анализ читаемости на основе индекса Флеша-Кинкейда
    def flesch_kincaid_grade(text):
        sentences = text.split('.')
        words = text.split()
        if len(sentences) == 0 or len(words) == 0:
            return 0
        return 0.39 * (len(words) / len(sentences)) + 11.8 * (sum(len(word) for word in words) / len(words)) - 15.59
    
    readability_scores = [flesch_kincaid_grade(text) for text in texts]
    
    def plot_readability(ax):
        ax.hist(readability_scores, bins=20, edgecolor='black')
        ax.set_xlabel("Индекс читаемости Флеша-Кинкейда")
        ax.set_ylabel("Частота")
        ax.set_title("Распределение индексов читаемости")
    
    return {
        "text": f"Средний индекс читаемости: {np.mean(readability_scores):.2f}\n"
                f"Медианный индекс читаемости: {np.median(readability_scores):.2f}",
        "figure": plot_readability
    }

def sentiment_analysis(texts):
    sentiments = [TextBlob(text).sentiment.polarity for text in texts]
    
    def plot_sentiment(ax):
        ax.hist(sentiments, bins=20, edgecolor='black')
        ax.set_xlabel("Тональность")
        ax.set_ylabel("Частота")
        ax.set_title("Распределение тональности текстов")
    
    return {
        "text": f"Средняя тональность: {np.mean(sentiments):.2f}\n"
                f"Медианная тональность: {np.median(sentiments):.2f}",
        "figure": plot_sentiment
    }

def keyword_extraction(texts):
    vectorizer = TfidfVectorizer(max_features=100)
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    def plot_keywords(ax):
        ax.bar(range(20), tfidf_matrix.sum(axis=0).A1.argsort()[-20:][::-1])
        ax.set_xticks(range(20))
        ax.set_xticklabels(feature_names[tfidf_matrix.sum(axis=0).A1.argsort()[-20:][::-1]], rotation=45, ha='right')
        ax.set_title("Топ-20 ключевых слов")
    
    return {
        "text": f"Извлечено {len(feature_names)} ключевых слов.",
        "figure": plot_keywords
    }

def text_structure_analysis(texts):
    avg_paragraph_count = np.mean([len(text.split('\n\n')) for text in texts])
    avg_sentence_count = np.mean([len(text.split('.')) for text in texts])
    
    def plot_structure(ax):
        structure_data = [
            np.mean([len(text.split('\n\n')) for text in texts]),
            np.mean([len(text.split('.')) for text in texts]),
            np.mean([len(text.split()) for text in texts])
        ]
        ax.bar(['Абзацы', 'Предложения', 'Слова'], structure_data)
        ax.set_title("Средняя структура текстов")
    
    return {
        "text": f"Среднее количество абзацев: {avg_paragraph_count:.2f}\n"
                f"Среднее количество предложений: {avg_sentence_count:.2f}",
        "figure": plot_structure
    }

def topic_modeling(texts):
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
    lda_output = lda_model.fit_transform(tfidf_matrix)
    
    feature_names = vectorizer.get_feature_names_out()
    
    topic_words = []
    for topic_idx, topic in enumerate(lda_model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
        topic_words.append(', '.join(top_words))
    
    def plot_topics(ax):
        ax.bar(range(5), lda_model.components_.sum(axis=1))
        ax.set_xticks(range(5))
        ax.set_xticklabels([f"Тема {i+1}" for i in range(5)], rotation=45, ha='right')
        ax.set_title("Распределение тем")
    
    return {
        "text": "Выявлено 5 основных тем:\n" + '\n'.join([f"Тема {i+1}: {words}" for i, words in enumerate(topic_words)]),
        "figure": plot_topics
    }

def writing_style_analysis(texts):
    avg_word_length = np.mean([np.mean([len(word) for word in text.split()]) for text in texts])
    avg_sentence_length = np.mean([np.mean([len(sentence.split()) for sentence in text.split('.')]) for text in texts])
    
    def plot_style(ax):
        ax.scatter([np.mean([len(word) for word in text.split()]) for text in texts],
                   [np.mean([len(sentence.split()) for sentence in text.split('.')]) for text in texts])
        ax.set_xlabel("Средняя длина слова")
        ax.set_ylabel("Средняя длина предложения")
        ax.set_title("Анализ стиля письма")
    
    return {
        "text": f"Средняя длина слова: {avg_word_length:.2f} символов\n"
                f"Средняя длина предложения: {avg_sentence_length:.2f} слов",
        "figure": plot_style
    }

def detect_text_reuse(texts):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    def plot_similarity(ax):
        sns.heatmap(similarity_matrix, ax=ax)
        ax.set_title("Матрица сходства текстов")
    
    return {
        "text": f"Средняя схожесть между текстами: {np.mean(similarity_matrix):.2f}",
        "figure": plot_similarity
    }

def citation_pattern_analysis(texts):
    citation_counts = [len(re.findall(r'\[\d+\]', text)) for text in texts]
    
    def plot_citations(ax):
        ax.hist(citation_counts, bins=max(citation_counts), edgecolor='black')
        ax.set_xlabel("Количество цитат")
        ax.set_ylabel("Частота")
        ax.set_title("Распределение количества цитат")
    
    return {
        "text": f"Среднее количество цитат: {np.mean(citation_counts):.2f}\n"
                f"Максимальное количество цитат: {max(citation_counts)}",
        "figure": plot_citations
    }

def visualize_text_similarity(texts, visualization_type, canvas):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    fig = canvas.figure
    fig.clear()
    ax = fig.add_subplot(111)
    
    if visualization_type == "Тепловая карта сходства":
        sns.heatmap(similarity_matrix, ax=ax)
        ax.set_title("Тепловая карта сходства текстов")
    elif visualization_type == "Сеть связей текстов":
        G = nx.Graph()
        for i in range(len(texts)):
            for j in range(i+1, len(texts)):
                if similarity_matrix[i][j] > 0.5:  # Порог схожести
                    G.add_edge(i, j, weight=similarity_matrix[i][j])
        pos = nx.spring_layout(G)
        nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_weight='bold')
        ax.set_title("Сеть связей текстов")
    elif visualization_type == "Временная динамика сходства":
        # Предполагаем, что тексты упорядочены по времени
        avg_similarities = [np.mean(row) for row in similarity_matrix]
        ax.plot(range(len(texts)), avg_similarities)
        ax.set_xlabel("Индекс текста")
        ax.set_ylabel("Средняя схожесть")
        ax.set_title("Временная динамика сходства текстов")
    elif visualization_type == "3D визуализация сходства":
        from mpl_toolkits.mplot3d import Axes3D
        ax = fig.add_subplot(111, projection='3d')
        x, y = np.meshgrid(range(len(texts)), range(len(texts)))
        ax.plot_surface(x, y, similarity_matrix, cmap='viridis')
        ax.set_xlabel("Текст 1")
        ax.set_ylabel("Текст 2")
        ax.set_zlabel("Сходство")
        ax.set_title("3D визуализация сходства текстов")
    
    canvas.draw()

# Дополнительные функции могут быть добавлены здесь по мере необходимости