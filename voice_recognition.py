import speech_recognition as sr

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Говорите...")
        audio = recognizer.listen(source)
    
    try:
        text = recognizer.recognize_google(audio, language="ru-RU")
        return text
    except sr.UnknownValueError:
        return "Речь не распознана"
    except sr.RequestError:
        return "Ошибка сервиса распознавания речи"
