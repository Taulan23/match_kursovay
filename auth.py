from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QPushButton, QMessageBox, QComboBox)
from PyQt5.QtCore import pyqtSignal
import json
import os

class User:
    def __init__(self, username, password, role):
        self.username = username
        self.password = password
        self.role = role

class AuthManager:
    def __init__(self):
        self.users = {}
        self.current_user = None
        self.load_users()

    def load_users(self):
        if os.path.exists('users.json'):
            with open('users.json', 'r') as f:
                data = json.load(f)
                for username, user_data in data.items():
                    self.users[username] = User(username, user_data['password'], user_data['role'])
        else:
            self.users['admin'] = User('admin', 'admin123', 'Администратор')
            
    def save_users(self):
        data = {username: {'password': user.password, 'role': user.role} for username, user in self.users.items()}
        with open('users.json', 'w') as f:
            json.dump(data, f)

    def register_user(self, username, password, role):
        if username in self.users:
            return False
        self.users[username] = User(username, password, role)
        self.save_users()
        return True

    def login(self, username, password):
        user = self.users.get(username)
        if user and user.password == password:
            self.current_user = user
            return True
        return False

    def logout(self):
        self.current_user = None

    def get_current_user(self):
        return self.current_user

class LoginDialog(QDialog):
    login_successful = pyqtSignal(str, str)

    def __init__(self, auth_manager):
        super().__init__()
        self.auth_manager = auth_manager
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("Вход в систему")
        layout = QVBoxLayout()

        self.username_input = QLineEdit()
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)

        layout.addWidget(QLabel("Имя пользователя:"))
        layout.addWidget(self.username_input)
        layout.addWidget(QLabel("Пароль:"))
        layout.addWidget(self.password_input)

        login_button = QPushButton("Войти")
        login_button.clicked.connect(self.login)
        layout.addWidget(login_button)

        register_button = QPushButton("Зарегистрироваться")
        register_button.clicked.connect(self.open_registration)
        layout.addWidget(register_button)

        guest_button = QPushButton("Войти как гость")
        guest_button.clicked.connect(self.login_as_guest)
        layout.addWidget(guest_button)

        self.setLayout(layout)

    def login(self):
        username = self.username_input.text()
        password = self.password_input.text()
        if self.auth_manager.login(username, password):
            user = self.auth_manager.get_current_user()
            self.login_successful.emit(user.username, user.role)
            self.accept()
        else:
            QMessageBox.warning(self, "Ошибка", "Неверное имя пользователя или пароль")

    def open_registration(self):
        dialog = RegistrationDialog(self.auth_manager)
        if dialog.exec_() == QDialog.Accepted:
            QMessageBox.information(self, "Успех", "Регистрация прошла успешно")

    def login_as_guest(self):
        self.login_successful.emit("Гость", "Гость")
        self.accept()

class RegistrationDialog(QDialog):
    def __init__(self, auth_manager):
        super().__init__()
        self.auth_manager = auth_manager
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("Регистрация")
        layout = QVBoxLayout()

        self.username_input = QLineEdit()
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        self.role_combo = QComboBox()
        self.role_combo.addItems(["Пользователь", "Администратор"])

        layout.addWidget(QLabel("Имя пользователя:"))
        layout.addWidget(self.username_input)
        layout.addWidget(QLabel("Пароль:"))
        layout.addWidget(self.password_input)
        layout.addWidget(QLabel("Роль:"))
        layout.addWidget(self.role_combo)

        register_button = QPushButton("Зарегистрироваться")
        register_button.clicked.connect(self.register)
        layout.addWidget(register_button)

        self.setLayout(layout)

    def register(self):
        username = self.username_input.text()
        password = self.password_input.text()
        role = self.role_combo.currentText()
        if self.auth_manager.register_user(username, password, role):
            self.accept()
        else:
            QMessageBox.warning(self, "Ошибка", "Пользователь с таким именем уже существует")