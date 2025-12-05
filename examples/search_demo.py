#!/usr/bin/env python3
"""
Пример использования поисковой системы.
Запуск: python examples/search_demo.py
"""

print("=" * 60)
print("Демонстрация Landmark Search System")
print("=" * 60)

print("\n📁 Структура проекта:")
print("landmark-search-system/")
print("├── landmark_system.py     # Основной скрипт")
print("├── config.py             # Настройки")
print("├── requirements.txt      # Зависимости")
print("├── data/                 # Данные")
print("└── examples/             # Эта демонстрация")

print("\n🚀 Примеры команд для запуска:")
print("\n1. Поиск по изображению:")
print('   python landmark_system.py search_image --image "тестовое_изображение.jpg"')
print('\n2. Поиск по тексту:')
print('   python landmark_system.py search_text --query "храм" --top_k 10')
print('\n3. Обработка данных:')
print('   python landmark_system.py process')
print('\n4. Построение системы:')
print('   python landmark_system.py build --clean')

print("\n" + "=" * 60)
print("Готово! Проект настроен и готов к использованию.")
print("=" * 60)
