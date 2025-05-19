import numpy as np
import os
import pickle

def generate_text_markov_chain(input_text, max_length):
    """
    Генерирует текст на основе цепей Маркова.
    
    Args:
        input_text (str): Исходный текст для генерации
        max_length (int): Максимальная длина генерируемого текста
        
    Returns:
        str: Сгенерированный текст
    """
    try:
        # Проверка входных данных
        if not input_text or len(input_text.strip()) == 0:
            return "Ошибка: Входной текст не может быть пустым"
        
        words = input_text.split()
        if len(words) < 2:
            return "Ошибка: Входной текст слишком короткий для генерации. Требуется минимум 2 слова."
        
        # Создаем словарь для цепи Маркова (n-граммы, где n=1)
        markov_dict = {}
        for i in range(len(words) - 1):
            # Используем кортеж из 1 слова в качестве ключа для лучшей работы с русским текстом
            key = (words[i],)
            if key in markov_dict:
                markov_dict[key].append(words[i+1])
            else:
                markov_dict[key] = [words[i+1]]
        
        # Начинаем с случайного слова
        start_idx = np.random.randint(0, len(words) - 1)
        current = (words[start_idx],)
        result = list(current)
        
        # Генерируем текст
        for _ in range(max_length - len(result)):
            if current in markov_dict:
                next_word = np.random.choice(markov_dict[current])
                result.append(next_word)
                
                # Обновляем текущее состояние
                current = (next_word,)
            else:
                # Если нет продолжения, выбираем случайное новое начало
                if len(words) > 0:
                    start_idx = np.random.randint(0, len(words) - 1)
                    current = (words[start_idx],)
                    # Не добавляем в результат, чтобы избежать повторений
                else:
                    break
                
            # Ограничение длины
            if len(result) >= max_length:
                break
        
        return ' '.join(result)
    except Exception as e:
        return f"Ошибка при генерации текста Марковской цепью: {str(e)}"

def generate_text_lstm(input_text, max_length, model=None, tokenizer=None):
    """
    Генерирует текст с использованием простой модели, заменяющей LSTM.
    
    Args:
        input_text (str): Исходный текст для генерации
        max_length (int): Максимальная длина генерируемого текста
        model: Обученная модель (словарь переходов)
        tokenizer: Токенизатор для преобразования текста (словарь)
        
    Returns:
        str: Сгенерированный текст
    """
    try:
        if not input_text or len(input_text.strip()) == 0:
            return "Ошибка: Входной текст не может быть пустым"
            
        if model is None or tokenizer is None:
            # Пытаемся загрузить сохраненную модель и токенизатор
            if os.path.exists('tokenizer.pickle'):
                try:
                    with open('tokenizer.pickle', 'rb') as handle:
                        tokenizer = pickle.load(handle)
                    
                    # Создаем заглушку для модели
                    model = {'transitions': {}}
                    return "Модель загружена из файла, но не поддерживается в этой версии. Используйте Марковскую цепь."
                except Exception as e:
                    return f"Не удалось загрузить сохраненную модель: {str(e)}"
            else:
                return "Модель не обучена. Пожалуйста, сначала обучите модель."
        
        # Используем простую модель на основе переходов между словами
        if isinstance(model, dict) and 'transitions' in model:
            transitions = model['transitions']
            
            seed_text = input_text
            words = seed_text.split()
            generated_text = seed_text
            
            current_word = words[-1] if words else ""
            
            # Генерация текста
            for _ in range(max_length):
                if current_word in transitions:
                    # Выбираем следующее слово на основе вероятностей переходов
                    next_words = list(transitions[current_word].keys())
                    probs = list(transitions[current_word].values())
                    
                    next_word = np.random.choice(next_words, p=probs)
                    generated_text += ' ' + next_word
                    current_word = next_word
                else:
                    # Если нет переходов для текущего слова, выбираем случайное слово
                    if len(transitions) > 0:
                        random_word = np.random.choice(list(transitions.keys()))
                        generated_text += ' ' + random_word
                        current_word = random_word
                    else:
                        break
            
            return generated_text
        else:
            return "Ошибка: Неподдерживаемый формат модели. Используйте Марковскую цепь."
    except Exception as e:
        return f"Ошибка при генерации текста: {str(e)}"

def train_lstm_model(texts, max_length=100, vocab_size=5000, epochs=10):
    """
    Создает простую модель на основе переходов между словами.
    
    Args:
        texts (list): Список текстов для обучения
        max_length (int): Максимальная длина последовательности
        vocab_size (int): Размер словаря
        epochs (int): Количество эпох обучения (не используется)
        
    Returns:
        tuple: (модель, токенизатор)
    """
    try:
        print("Создаем упрощенную модель для генерации текста")
        
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
        
        # Сохраняем токенизатор
        with open('tokenizer.pickle', 'wb') as handle:
            pickle.dump(simple_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        return simple_model, simple_tokenizer
        
    except Exception as e:
        print(f"Ошибка при создании упрощенной модели: {str(e)}")
        return None, None

# Добавьте другие функции генерации текста по необходимости
