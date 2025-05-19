import sys
import os
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QTextEdit, QFileDialog, QMessageBox, 
                             QTabWidget, QComboBox, QProgressBar, QListWidget, QSplitter,
                             QTreeView, QHeaderView, QAbstractItemView, QLineEdit, QCheckBox,
                             QRadioButton, QButtonGroup, QScrollArea, QGridLayout)
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
import json

from auth import AuthManager, LoginDialog
from text_analysis import (check_plagiarism, generate_word_cloud, plot_text_length_distribution,
                           analyze_text_complexity, analyze_readability, analyze_sentiment,
                           extract_keywords, analyze_text_structure)
from model_training import train_model, evaluate_model
from text_generation import generate_text_markov_chain, generate_text_lstm, train_lstm_model
from voice_recognition import recognize_speech
from advanced_features import (perform_advanced_analysis, visualize_text_similarity,
                               topic_modeling, writing_style_analysis,
                               detect_text_reuse, citation_pattern_analysis)

# Проверяем наличие библиотеки python-docx
try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

class TextProcessingThread(QThread):
    progress_updated = pyqtSignal(int)
    finished = pyqtSignal(list)

    def __init__(self, texts, stemming=False, remove_stopwords=True):
        super().__init__()
        self.texts = texts
        self.stemming = stemming
        self.remove_stopwords = remove_stopwords

    def run(self):
        processed_texts = []
        for i, text in enumerate(self.texts):
            # Здесь должна быть реальная обработка текста
            processed_text = text.lower()  # Пример простой обработки
            processed_texts.append(processed_text)
            self.progress_updated.emit(int((i + 1) / len(self.texts) * 100))
        self.finished.emit(processed_texts)

class PlagiarismCheckerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.auth_manager = AuthManager()
        self.data_model = QStandardItemModel()
        self.texts_database = []
        self.similarities = []
        self.classifier = None
        self.vectorizer = None
        self.username = ""
        self.role = "Гость"

        self.setup_ui()
        self.setWindowTitle("Расширенная система анализа текстов")
        self.setGeometry(100, 100, 1400, 900)

        self.setup_data_model()
        self.show_login_dialog()

    def setup_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        self.setup_data_loading_tab()
        self.setup_text_checking_tab()
        self.setup_advanced_analysis_tab()
        self.setup_data_visualization_tab()
        self.setup_model_training_tab()
        self.setup_text_generation_tab()
        self.setup_voice_recognition_tab()
        self.setup_settings_tab()

        self.statusBar().showMessage("Готово к работе")
        self.setup_menu()

    def setup_data_loading_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        load_csv_button = QPushButton("Загрузить CSV")
        load_csv_button.clicked.connect(self.load_csv)
        layout.addWidget(load_csv_button)
        
        if DOCX_AVAILABLE:
            load_docx_button = QPushButton("Загрузить DOCX")
            load_docx_button.clicked.connect(self.load_docx)
            layout.addWidget(load_docx_button)

        load_txt_button = QPushButton("Загрузить TXT")
        load_txt_button.clicked.connect(self.load_txt)
        layout.addWidget(load_txt_button)
        
        load_dataset_button = QPushButton("Загрузить датасет")
        load_dataset_button.clicked.connect(self.load_dataset)
        layout.addWidget(load_dataset_button)

        self.data_list = QListWidget()
        layout.addWidget(self.data_list)

        self.tabs.addTab(tab, "Загрузка данных")

    def setup_text_checking_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.text_input = QTextEdit()
        layout.addWidget(QLabel("Введите текст для проверки:"))
        layout.addWidget(self.text_input)

        check_button = QPushButton("Проверить на плагиат")
        check_button.clicked.connect(self.check_text)
        layout.addWidget(check_button)

        self.result_label = QLabel()
        layout.addWidget(self.result_label)

        self.similarity_canvas = FigureCanvas(plt.figure(figsize=(8, 6)))
        layout.addWidget(self.similarity_canvas)

        self.tabs.addTab(tab, "Проверка текста")

    def setup_advanced_analysis_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.analysis_type_combo = QComboBox()
        self.analysis_type_combo.addItems([
            "Облако слов", "Распределение длин текстов", "Анализ сложности текста",
            "Оценка читаемости", "Анализ тональности", "Извлечение ключевых слов",
            "Анализ структуры текста", "Тематическое моделирование",
            "Анализ стиля письма", "Обнаружение повторного использования текста",
            "Анализ паттернов цитирования"
        ])
        layout.addWidget(QLabel("Выберите тип анализа:"))
        layout.addWidget(self.analysis_type_combo)

        analyze_button = QPushButton("Выполнить анализ")
        analyze_button.clicked.connect(self.run_advanced_analysis)
        layout.addWidget(analyze_button)

        self.analysis_result = QTextEdit()
        self.analysis_result.setReadOnly(True)
        layout.addWidget(self.analysis_result)

        self.analysis_canvas = FigureCanvas(plt.figure(figsize=(8, 6)))
        layout.addWidget(self.analysis_canvas)

        self.tabs.addTab(tab, "Расширенный анализ")

    def setup_data_visualization_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.visualization_type_combo = QComboBox()
        self.visualization_type_combo.addItems([
            "Тепловая карта сходства", "Сеть связей текстов",
            "Временная динамика сходства", "3D визуализация сходства"
        ])
        layout.addWidget(QLabel("Выберите тип визуализации:"))
        layout.addWidget(self.visualization_type_combo)

        visualize_button = QPushButton("Визуализировать")
        visualize_button.clicked.connect(self.run_visualization)
        layout.addWidget(visualize_button)

        self.visualization_canvas = FigureCanvas(plt.figure(figsize=(10, 8)))
        layout.addWidget(self.visualization_canvas)

        self.tabs.addTab(tab, "Визуализация данных")

    def setup_model_training_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems([
            "Наивный Байес", "Логистическая регрессия", "Случайный лес",
            "Градиентный бустинг", "Нейронная сеть"
        ])
        layout.addWidget(QLabel("Выберите тип модели:"))
        layout.addWidget(self.model_type_combo)

        train_button = QPushButton("Обучить модель")
        train_button.clicked.connect(self.train_model)
        layout.addWidget(train_button)

        evaluate_button = QPushButton("Оценить модель")
        evaluate_button.clicked.connect(self.evaluate_model)
        layout.addWidget(evaluate_button)

        self.model_result_text = QTextEdit()
        self.model_result_text.setReadOnly(True)
        layout.addWidget(self.model_result_text)

        self.model_canvas = FigureCanvas(plt.figure(figsize=(8, 6)))
        layout.addWidget(self.model_canvas)

        self.tabs.addTab(tab, "Обучение модели")

    def setup_text_generation_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.generation_input = QTextEdit()
        layout.addWidget(QLabel("Введите начало текста:"))
        layout.addWidget(self.generation_input)

        self.max_length_input = QLineEdit()
        self.max_length_input.setPlaceholderText("Масимальная длина (по умолчанию: 100)")
        layout.addWidget(QLabel("Максимальная длина генерации:"))
        layout.addWidget(self.max_length_input)

        self.generation_type_combo = QComboBox()
        self.generation_type_combo.addItems(["Марковская цепь", "LSTM"])
        layout.addWidget(QLabel("Выберите метод генерации:"))
        layout.addWidget(self.generation_type_combo)

        generate_button = QPushButton("Сгенерировать текст")
        generate_button.clicked.connect(self.generate_text)
        layout.addWidget(generate_button)

        self.generated_text_output = QTextEdit()
        self.generated_text_output.setReadOnly(True)
        layout.addWidget(QLabel("Сгенерированный текст:"))
        layout.addWidget(self.generated_text_output)

        self.tabs.addTab(tab, "Генерация текста")

    def setup_voice_recognition_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        record_button = QPushButton("Записать голос")
        record_button.clicked.connect(self.record_voice)
        layout.addWidget(record_button)

        self.voice_text_output = QTextEdit()
        self.voice_text_output.setReadOnly(True)
        layout.addWidget(QLabel("Распознанный текст:"))
        layout.addWidget(self.voice_text_output)

        generate_from_voice_button = QPushButton("Сгенерировать текст из голоса")
        generate_from_voice_button.clicked.connect(self.generate_text_from_voice)
        layout.addWidget(generate_from_voice_button)

        self.voice_generated_text_output = QTextEdit()
        self.voice_generated_text_output.setReadOnly(True)
        layout.addWidget(QLabel("Сгенерированный текст из голоса:"))
        layout.addWidget(self.voice_generated_text_output)

        self.tabs.addTab(tab, "Распознавание голоса")

    def setup_settings_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.stemming_checkbox = QCheckBox("Использовать стемминг")
        layout.addWidget(self.stemming_checkbox)

        self.remove_stopwords_checkbox = QCheckBox("Удалять стоп-слова")
        self.remove_stopwords_checkbox.setChecked(True)
        layout.addWidget(self.remove_stopwords_checkbox)

        self.language_combo = QComboBox()
        self.language_combo.addItems(["Русский", "Английский"])
        layout.addWidget(QLabel("Язык анализа:"))
        layout.addWidget(self.language_combo)

        save_settings_button = QPushButton("Сохранить настройки")
        save_settings_button.clicked.connect(self.save_settings)
        layout.addWidget(save_settings_button)

        self.tabs.addTab(tab, "Настройки")

    def setup_menu(self):
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu('Файл')
        load_action = file_menu.addAction('Загрузить данные')
        load_action.triggered.connect(self.load_data)
        save_action = file_menu.addAction('Сохранить результаты')
        save_action.triggered.connect(self.save_results)
        exit_action = file_menu.addAction('Выход')
        exit_action.triggered.connect(self.close)

        edit_menu = menubar.addMenu('Правка')
        clear_action = edit_menu.addAction('Очистить все')
        clear_action.triggered.connect(self.clear_all)

        tools_menu = menubar.addMenu('Инструменты')
        preprocess_action = tools_menu.addAction('Предобработка текстов')
        preprocess_action.triggered.connect(self.preprocess_all_texts)

        help_menu = menubar.addMenu('Помощь')
        about_action = help_menu.addAction('О программе')
        about_action.triggered.connect(self.show_about)

        account_menu = menubar.addMenu('Аккаунт')
        logout_action = account_menu.addAction('Выйти')
        logout_action.triggered.connect(self.logout)

    def setup_data_model(self):
        self.data_model.setHorizontalHeaderLabels(['Тексты'])

    def show_login_dialog(self):
        dialog = LoginDialog(self.auth_manager)
        dialog.login_successful.connect(self.on_login_successful)
        if dialog.exec_() == LoginDialog.Rejected:
            self.close()

    def on_login_successful(self, username, role):
        self.username = username
        self.role = role
        self.update_ui_for_role()
        self.statusBar().showMessage(f"Вы вошли как: {username} ({role})")

    def update_ui_for_role(self):
        if self.role == "Администратор":
            for i in range(self.tabs.count()):
                self.tabs.setTabEnabled(i, True)
        elif self.role == "Пользователь":
            for i in range(self.tabs.count()):
                self.tabs.setTabEnabled(i, i != 7)  # Отключаем вкладку настроек
        else:  # Гость
            for i in range(self.tabs.count()):
                self.tabs.setTabEnabled(i, i < 2)  # Разрешаем только первые две вкладки

    def load_csv(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(self, "Выберите CSV файл", "", "CSV Files (*.csv)")
            if file_path:
                df = pd.read_csv(file_path)
                
                # Проверяем, есть ли столбец 'text'
                if 'text' not in df.columns:
                    # Если нет столбца 'text', но есть хотя бы один столбец,
                    # используем первый столбец как источник текста
                    if len(df.columns) > 0:
                        text_column = df.columns[0]
                        texts = df[text_column].dropna().tolist()
                        QMessageBox.information(self, "Информация", f"Столбец 'text' не найден, используется столбец '{text_column}'")
                    else:
                        QMessageBox.warning(self, "Ошибка", "CSV файл не содержит столбцов с данными")
                        return
                else:
                    texts = df['text'].dropna().tolist()
                
                if texts:
                    self.texts_database.extend(texts)
                    self.update_data_list()
                    QMessageBox.information(self, "Успех", f"Загружено {len(texts)} текстов из CSV файла")
                else:
                    QMessageBox.warning(self, "Предупреждение", "В файле не найдено текстовых данных")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить CSV файл: {str(e)}\n{traceback.format_exc()}")

    def load_docx(self):
        if not DOCX_AVAILABLE:
            QMessageBox.warning(self, "Ошибка", "Библиотека python-docx не установлена. Установите её с помощью 'pip install python-docx'")
            return
        
        try:
            file_path, _ = QFileDialog.getOpenFileName(self, "Выберите DOCX файл", "", "Word Documents (*.docx)")
            if file_path:
                doc = docx.Document(file_path)
                texts = []
                for paragraph in doc.paragraphs:
                    if paragraph.text.strip():  # Игнорировать пустые параграфы
                        texts.append(paragraph.text)
                
                if texts:
                    self.texts_database.extend(texts)
                    self.update_data_list()
                    QMessageBox.information(self, "Успех", f"Загружено {len(texts)} текстовых фрагментов из DOCX файла")
                else:
                    QMessageBox.warning(self, "Предупреждение", "В файле не найдено текстовых параграфов")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить DOCX файл: {str(e)}\n{traceback.format_exc()}")

    def load_txt(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Выберите TXT файл", "", "Text Files (*.txt)")
        if file_name:
            with open(file_name, 'r', encoding='utf-8') as file:
                self.texts_database.append(file.read())
            self.update_data_list()

    def update_data_list(self):
        self.data_list.clear()
        for i, text in enumerate(self.texts_database):
            self.data_list.addItem(f"Текст {i+1}: {text[:50]}...")

    def check_text(self):
        input_text = self.text_input.toPlainText()
        if not input_text or not self.texts_database:
            QMessageBox.warning(self, "Ошибка", "Введите текст и загрузите базу данных")
            return

        self.similarities = check_plagiarism(input_text, self.texts_database)
        max_similarity = max(self.similarities)
        self.result_label.setText(f"Максимальное сходство: {max_similarity:.2f}")
        self.plot_similarity()

    def plot_similarity(self):
        ax = self.similarity_canvas.figure.subplots()
        ax.clear()
        ax.bar(range(len(self.similarities)), self.similarities)
        ax.set_xlabel("Индекс текста")
        ax.set_ylabel("Сходство")
        ax.set_title("Сходство с текстами в базе данных")
        self.similarity_canvas.draw()

    def run_advanced_analysis(self):
        if not self.texts_database:
            QMessageBox.warning(self, "Ошибка", "Загрузите тексты для анализа")
            return

        analysis_type = self.analysis_type_combo.currentText()
        result = perform_advanced_analysis(self.texts_database, analysis_type)
        
        self.analysis_result.setText(result['text'])
        
        if 'figure' in result:
            self.analysis_canvas.figure.clear()
            ax = self.analysis_canvas.figure.add_subplot(111)
            result['figure'](ax)
            self.analysis_canvas.draw()

    def run_visualization(self):
        if not self.texts_database:
            QMessageBox.warning(self, "Ошибка", "Загрузите тексты для визуализации")
            return

        visualization_type = self.visualization_type_combo.currentText()
        visualize_text_similarity(self.texts_database, visualization_type, self.visualization_canvas)

    def train_model(self):
        if not self.texts_database:
            QMessageBox.warning(self, "Ошибка", "Загрузите тексты для обучения модели")
            return

        # Для демонстрации используем случайные метки
        labels = np.random.randint(2, size=len(self.texts_database))
        model_type = self.model_type_combo.currentText()

        self.classifier, self.vectorizer, accuracy, conf_matrix, class_report = train_model(self.texts_database, labels, model_type)

        result = f"Точность модели: {accuracy}\n\n"
        result += "Матрица ошибок:\n"
        result += str(conf_matrix) + "\n\n"
        result += "Отчет о классификации:\n"
        result += class_report

        self.model_result_text.setText(result)

        # Визуализация результатов обучения
        self.model_canvas.figure.clear()
        ax = self.model_canvas.figure.add_subplot(111)
        sns.heatmap(conf_matrix, annot=True, ax=ax)
        ax.set_title("Матрица ошибок")
        self.model_canvas.draw()

    def evaluate_model(self):
        if self.classifier is None:
            QMessageBox.warning(self, "Ошибка", "Сначала обучите модель")
            return

        # Для демонстрации используем случайные метки
        labels = np.random.randint(2, size=len(self.texts_database))
        accuracy, conf_matrix, class_report = evaluate_model(self.classifier, self.vectorizer, self.texts_database, labels)

        result = f"Точность модели при оценке: {accuracy}\n\n"
        result += "Матрица ошибок:\n"
        result += str(conf_matrix) + "\n\n"
        result += "Отчет о классификации:\n"
        result += class_report

        self.model_result_text.setText(result)

        # Визуализация результатов оценки
        self.model_canvas.figure.clear()
        ax = self.model_canvas.figure.add_subplot(111)
        sns.heatmap(conf_matrix, annot=True, ax=ax)
        ax.set_title("Матрица ошибок при оценке")
        self.model_canvas.draw()

    def generate_text(self):
        input_text = self.generation_input.toPlainText()
        if not input_text:
            QMessageBox.warning(self, "Ошибка", "Введите начальный текст для генерации")
            return
        
        max_length = int(self.max_length_input.text()) if self.max_length_input.text() else 100
        generation_type = self.generation_type_combo.currentText()

        try:
            if generation_type == "Марковская цепь":
                # Используем все загруженные тексты как основу для обучения
                if len(self.texts_database) > 0:
                    # Объединяем тексты для лучшей марковской цепи
                    training_text = " ".join(self.texts_database)
                    # Начинаем генерацию с заданного пользователем текста
                    generated_text = generate_text_markov_chain(training_text + " " + input_text, max_length)
                else:
                    # Если нет загруженных текстов, используем только ввод пользователя
                    generated_text = generate_text_markov_chain(input_text, max_length)
            else:  # LSTM
                if not hasattr(self, 'lstm_model') or not hasattr(self, 'lstm_tokenizer'):
                    # Проверяем, достаточно ли данных для обучения LSTM
                    if len(self.texts_database) < 3:
                        QMessageBox.warning(self, "Ошибка", "Недостаточно данных для обучения модели LSTM. Загрузите больше текстов.")
                        return
                        
                    QMessageBox.information(self, "Информация", "Начинаем обучение модели LSTM. Это может занять некоторое время.")
                    
                    # Объединяем все тексты для лучшего обучения LSTM
                    combined_text = " ".join(self.texts_database) + " " + input_text
                    self.lstm_model, self.lstm_tokenizer = train_lstm_model([combined_text], epochs=5)
                    
                    if not self.lstm_model or not self.lstm_tokenizer:
                        QMessageBox.warning(self, "Ошибка", "Не удалось обучить LSTM модель")
                        return
                        
                generated_text = generate_text_lstm(input_text, max_length, self.lstm_model, self.lstm_tokenizer)

            # Проверяем результат и показываем его
            if generated_text.startswith("Ошибка"):
                QMessageBox.warning(self, "Ошибка генерации", generated_text)
            else:
                self.generated_text_output.setText(generated_text)
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Произошла ошибка при генерации текста: {str(e)}\n{traceback.format_exc()}")

    def record_voice(self):
        recognized_text = recognize_speech()
        self.voice_text_output.setText(recognized_text)

    def generate_text_from_voice(self):
        voice_text = self.voice_text_output.toPlainText()
        if not voice_text:
            QMessageBox.warning(self, "Ошибка", "Сначала запишите голос")
            return

        max_length = int(self.max_length_input.text()) if self.max_length_input.text() else 100
        generated_text = generate_text_markov_chain(voice_text, max_length)
        self.voice_generated_text_output.setText(generated_text)

    def save_settings(self):
        # Здесь можно добавить логику сохранения настроек
        QMessageBox.information(self, "Настройки", "Настройки сохранены")

    def load_data(self):
        # Логика загрузки данных
        self.load_dataset()

    def save_results(self):
        # Логика сохранения результатов
        pass

    def clear_all(self):
        self.texts_database.clear()
        self.update_data_list()
        self.text_input.clear()
        self.result_label.clear()
        self.similarity_canvas.figure.clear()
        self.similarity_canvas.draw()
        self.analysis_result.clear()
        self.analysis_canvas.figure.clear()
        self.analysis_canvas.draw()
        self.model_result_text.clear()
        self.model_canvas.figure.clear()
        self.model_canvas.draw()
        self.generated_text_output.clear()
        self.voice_text_output.clear()
        self.voice_generated_text_output.clear()

    def preprocess_all_texts(self):
        if not self.texts_database:
            QMessageBox.warning(self, "Ошибка", "Загрузите тексты для предобработки")
            return

        self.preprocessing_thread = TextProcessingThread(
            self.texts_database,
            stemming=self.stemming_checkbox.isChecked(),
            remove_stopwords=self.remove_stopwords_checkbox.isChecked()
        )
        self.preprocessing_thread.progress_updated.connect(self.update_preprocessing_progress)
        self.preprocessing_thread.finished.connect(self.preprocessing_finished)
        self.preprocessing_thread.start()

        self.preprocessing_progress = QProgressBar(self)
        self.preprocessing_progress.setRange(0, 100)
        self.statusBar().addWidget(self.preprocessing_progress)

    def update_preprocessing_progress(self, value):
        self.preprocessing_progress.setValue(value)

    def preprocessing_finished(self, processed_texts):
        self.texts_database = processed_texts
        self.update_data_list()
        self.statusBar().removeWidget(self.preprocessing_progress)
        QMessageBox.information(self, "Успех", "Предобработка текстов завершена")

    def show_about(self):
        QMessageBox.about(self, "О программе", "Расширенная система анализа текста и проверка текстов на плагиат\nВерсия 1.0\n© 2025 Егор Булатов")

    def logout(self):
        self.auth_manager.logout()
        self.show_login_dialog()

    def load_dataset(self):
        """Загружает датасет из различных форматов файлов"""
        file_name, file_filter = QFileDialog.getOpenFileName(
            self, 
            "Выберите файл датасета", 
            "", 
            "Все поддерживаемые (*.csv *.txt *.json *.jsonl);;CSV (*.csv);;TXT (*.txt);;JSON (*.json);;JSONL (*.jsonl)"
        )
        
        if not file_name:
            return
        
        try:
            file_extension = os.path.splitext(file_name)[1].lower()
            
            if file_extension == '.csv':
                # Пробуем разные варианты колонок для CSV
                df = pd.read_csv(file_name)
                
                # Проверяем различные имена столбцов, которые могут содержать текст
                text_columns = ['text', 'Text', 'content', 'Content', 'текст', 'Текст', 'body', 'Body']
                found_column = None
                
                for column in text_columns:
                    if column in df.columns:
                        found_column = column
                        break
                
                if found_column:
                    self.texts_database.extend(df[found_column].dropna().tolist())
                else:
                    # Если не нашли стандартную колонку, используем первую текстовую колонку
                    text_cols = df.select_dtypes(include=['object']).columns
                    if len(text_cols) > 0:
                        self.texts_database.extend(df[text_cols[0]].dropna().tolist())
                    else:
                        raise ValueError("Не найдена текстовая колонка в CSV файле")
                    
            elif file_extension == '.txt':
                with open(file_name, 'r', encoding='utf-8') as f:
                    # Пробуем разделить на строки или абзацы
                    content = f.read()
                    lines = content.split('\n\n')  # Разделяем по пустым строкам
                    
                    # Если получилась только одна запись, пробуем разделить по строкам
                    if len(lines) <= 1:
                        lines = content.split('\n')
                    
                    # Удаляем пустые строки
                    lines = [line.strip() for line in lines if line.strip()]
                    if lines:
                        self.texts_database.extend(lines)
                    else:
                        raise ValueError("Файл не содержит текста")
                    
            elif file_extension in ['.json', '.jsonl']:
                with open(file_name, 'r', encoding='utf-8') as f:
                    if file_extension == '.json':
                        # Пробуем разные форматы JSON
                        try:
                            # Сначала пробуем как массив объектов
                            data = json.load(f)
                            if isinstance(data, list):
                                for item in data:
                                    # Ищем текстовые поля
                                    for field in ['text', 'content', 'body', 'текст']:
                                        if field in item and item[field]:
                                            self.texts_database.append(str(item[field]))
                                            break
                            elif isinstance(data, dict):
                                # Проверяем, есть ли массив данных
                                for field in ['data', 'texts', 'items', 'records']:
                                    if field in data and isinstance(data[field], list):
                                        for item in data[field]:
                                            for text_field in ['text', 'content', 'body', 'текст']:
                                                if text_field in item and item[text_field]:
                                                    self.texts_database.append(str(item[text_field]))
                                                    break
                        except json.JSONDecodeError:
                            # Если не удалось декодировать как JSON, пробуем как JSONL
                            f.seek(0)
                            lines = f.readlines()
                            for line in lines:
                                try:
                                    item = json.loads(line.strip())
                                    for field in ['text', 'content', 'body', 'текст']:
                                        if field in item and item[field]:
                                            self.texts_database.append(str(item[field]))
                                            break
                                except json.JSONDecodeError:
                                    continue
                    else:  # JSONL
                        lines = f.readlines()
                        for line in lines:
                            try:
                                item = json.loads(line.strip())
                                for field in ['text', 'content', 'body', 'текст']:
                                    if field in item and item[field]:
                                        self.texts_database.append(str(item[field]))
                                        break
                            except json.JSONDecodeError:
                                continue
            
            # Обновляем список текстов
            self.update_data_list()
            
            if not self.texts_database:
                QMessageBox.warning(self, "Предупреждение", "Из файла не удалось извлечь текстовые данные")
            else:
                QMessageBox.information(self, "Успех", f"Загружено {len(self.texts_database)} текстов")
            
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Не удалось загрузить файл: {str(e)}\n{traceback.format_exc()}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PlagiarismCheckerApp()
    window.show()
    sys.exit(app.exec_())