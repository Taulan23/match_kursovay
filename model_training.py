from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def train_model(texts, labels, model_type="Наивный Байес"):
    """
    Обучает модель на текстовых данных.
    
    Args:
        texts (list): Список текстов для обучения
        labels (list): Список меток для текстов
        model_type (str): Тип модели для обучения
        
    Returns:
        tuple: (модель, векторизатор, точность, матрица ошибок, отчет)
    """
    # Преобразование текста в числовые признаки
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(texts)
    
    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    
    # Выбор модели
    if model_type == "Наивный Байес":
        model = MultinomialNB()
    elif model_type == "Логистическая регрессия":
        model = LogisticRegression(max_iter=1000)
    elif model_type == "Случайный лес":
        model = RandomForestClassifier(n_estimators=100)
    elif model_type == "Градиентный бустинг":
        model = GradientBoostingClassifier()
    elif model_type == "Нейронная сеть":
        model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
    else:
        model = MultinomialNB()  # По умолчанию используем Наивный Байес
    
    # Обучение модели
    model.fit(X_train, y_train)
    
    # Оценка модели
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    return model, vectorizer, accuracy, conf_matrix, class_report

def evaluate_model(model, vectorizer, texts, true_labels):
    """
    Оценивает производительность модели на новых данных.
    
    Args:
        model: Обученная модель
        vectorizer: Векторизатор
        texts (list): Список текстов для оценки
        true_labels (list): Истинные метки
        
    Returns:
        tuple: (точность, матрица ошибок, отчет)
    """
    # Преобразование текстов
    X = vectorizer.transform(texts)
    
    # Предсказание
    y_pred = model.predict(X)
    
    # Метрики
    accuracy = accuracy_score(true_labels, y_pred)
    conf_matrix = confusion_matrix(true_labels, y_pred)
    class_report = classification_report(true_labels, y_pred)
    
    return accuracy, conf_matrix, class_report

# Вспомогательные функции для обучения моделей
def train_lstm_model(texts, max_length=100, vocab_size=5000, epochs=10):
    """
    Заглушка для функции обучения LSTM модели.
    В данной реализации возвращает простую модель на основе словаря частотности слов.
    
    Args:
        texts (list): Список текстов для обучения
        max_length (int): Максимальная длина последовательности
        vocab_size (int): Размер словаря
        epochs (int): Количество эпох обучения
        
    Returns:
        tuple: (модель, токенизатор)
    """
    try:
        print("Создаем упрощенную модель для генерации текста (без TensorFlow)")
        
        # Объединяем все тексты
        all_text = " ".join(texts)
        
        # Создаем простой токенизатор на основе частоты слов
        words = all_text.split()
        word_counts = {}
        for word in words:
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1
        
        # Создаем простой словарь слов (имитация токенизатора)
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        vocabulary = {word: i+1 for i, (word, _) in enumerate(sorted_words[:vocab_size])}
        
        # Создаем простую "модель" - словарь переходов между словами
        transitions = {}
        for i in range(len(words) - 1):
            current_word = words[i]
            next_word = words[i+1]
            
            if current_word in transitions:
                if next_word in transitions[current_word]:
                    transitions[current_word][next_word] += 1
                else:
                    transitions[current_word][next_word] = 1
            else:
                transitions[current_word] = {next_word: 1}
        
        # Нормализуем вероятности
        for word in transitions:
            total = sum(transitions[word].values())
            for next_word in transitions[word]:
                transitions[word][next_word] /= total
        
        # Создаем простую версию модели и токенизатора
        simple_model = {"transitions": transitions}
        simple_tokenizer = {"vocabulary": vocabulary, "index_word": {i+1: word for i, (word, _) in enumerate(sorted_words[:vocab_size])}}
        
        return simple_model, simple_tokenizer
        
    except Exception as e:
        print(f"Ошибка при создании упрощенной модели: {str(e)}")
        return None, None

# Добавьте другие функции обучения моделей по необходимости
