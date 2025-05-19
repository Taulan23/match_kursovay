#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import argparse
from text_generation import generate_text_markov_chain, generate_text_lstm, train_lstm_model

def print_help():
    """Выводит справку по использованию программы"""
    print("""
Генератор текста - программа для генерации текста на основе различных алгоритмов.

Использование:
    python text_generator.py [опции]

Опции:
    --help, -h              Показать эту справку
    --input, -i ФАЙЛ        Указать входной файл с текстом
    --output, -o ФАЙЛ       Указать выходной файл для сохранения результата
    --method, -m МЕТОД      Метод генерации текста (markov или lstm)
    --length, -l ДЛИНА      Максимальная длина генерируемого текста
    --train, -t             Обучить модель LSTM перед генерацией
    --epochs, -e ЭПОХИ      Количество эпох для обучения LSTM модели

Примеры:
    python text_generator.py -i input.txt -o output.txt -m markov -l 100
    python text_generator.py -i input.txt -m lstm -t -e 5 -l 200
    """)

def main():
    """Основная функция программы"""
    parser = argparse.ArgumentParser(description='Генератор текста')
    parser.add_argument('--input', '-i', type=str, help='Входной файл с текстом')
    parser.add_argument('--output', '-o', type=str, help='Выходной файл для сохранения результата')
    parser.add_argument('--method', '-m', type=str, choices=['markov', 'lstm'], default='markov',
                        help='Метод генерации текста (markov или lstm)')
    parser.add_argument('--length', '-l', type=int, default=100, help='Максимальная длина генерируемого текста')
    parser.add_argument('--train', '-t', action='store_true', help='Обучить модель LSTM перед генерацией')
    parser.add_argument('--epochs', '-e', type=int, default=5, help='Количество эпох для обучения LSTM модели')
    
    # Если нет аргументов, показываем справку
    if len(sys.argv) == 1:
        print_help()
        return
        
    args = parser.parse_args()
    
    # Проверяем наличие входного файла
    if not args.input:
        print("Ошибка: Не указан входной файл. Используйте опцию --input или -i.")
        return
        
    if not os.path.exists(args.input):
        print(f"Ошибка: Файл {args.input} не найден.")
        return
        
    # Читаем входной текст
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            input_text = f.read()
    except Exception as e:
        print(f"Ошибка при чтении файла: {str(e)}")
        return
        
    if not input_text:
        print("Ошибка: Входной файл пуст.")
        return
        
    print(f"Метод генерации: {args.method}")
    print(f"Максимальная длина: {args.length}")
    
    # Генерация текста
    if args.method == 'markov':
        print("Генерация текста с использованием цепей Маркова...")
        generated_text = generate_text_markov_chain(input_text, args.length)
    else:  # lstm
        if args.train:
            print(f"Обучение модели LSTM ({args.epochs} эпох)...")
            model, tokenizer = train_lstm_model([input_text], max_length=args.length, epochs=args.epochs)
            if model is None or tokenizer is None:
                print("Ошибка: Не удалось обучить модель LSTM.")
                return
            print("Модель LSTM успешно обучена.")
            
        print("Генерация текста с использованием LSTM...")
        generated_text = generate_text_lstm(input_text[:100], args.length)  # Используем первые 100 символов как затравку
    
    print("\nСгенерированный текст:")
    print("-" * 50)
    print(generated_text)
    print("-" * 50)
    
    # Сохраняем результат в файл, если указан выходной файл
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(generated_text)
            print(f"Результат сохранен в файл: {args.output}")
        except Exception as e:
            print(f"Ошибка при сохранении результата: {str(e)}")
    
    print("\nПояснение:")
    print("Генерация текста - это процесс создания нового текста на основе существующего.")
    print("В данной программе реализованы два метода генерации текста:")
    print("1. Цепи Маркова - простой статистический метод, который анализирует")
    print("   вероятности появления слов после других слов в исходном тексте.")
    print("2. LSTM (Long Short-Term Memory) - нейронная сеть, способная учитывать")
    print("   более сложные зависимости в тексте и генерировать более связные тексты.")
    print("\nДля использования LSTM необходимо сначала обучить модель на исходном тексте.")
    print("Чем больше исходный текст и чем больше эпох обучения, тем лучше будет результат.")
    print("Однако это требует больше времени и вычислительных ресурсов.")
    
if __name__ == "__main__":
    main() 