# 🏛️ Landmark Search System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CLIP](https://img.shields.io/badge/CLIP-OpenAI-red.svg)](https://openai.com/research/clip)

Система поиска достопримечательностей по изображениям и текстовым запросам с использованием нейросетей (CLIP). Поддерживает города: Екатеринбург, Нижний Новгород, Владимир, Ярославль.

## 🚀 Быстрый старт

## Установка
### 1. Клонируйте репозиторий
```bash
git clone https://github.com/Andrewkea/landmark-search-system.git
cd landmark-search-system
```
### 2. Установите зависимости
```bash
pip install -r requirements.txt
```

# 📥 Скачивание данных

Данные слишком велики для GitHub. Скачайте их отдельно:

##  Google Drive
1. Перейдите по ссылке: [Google Drive](https://drive.google.com/drive/folders/1gXHmgrFyyMLp1BxYYWW-lyYHs9K0xMt1?usp=sharing)
2. Скачайте архив `data.zip` (~500 МБ)
3. Распакуйте в папку `data/`

### Пример:
```text
data/
├── EKB_images.csv
├── EKB_places.csv
├── NN_images.csv
└── NN_places.csv
```

# 🛠️ Использование
1. Обработка данных
```bash
python landmark_system.py process
```
Система автоматически создаст:

- combined_dataset.csv (обработанные данные)

- decoded_images/ (изображения из base64)

2. Построение системы поиска
```bash
python landmark_system.py build --clean
```
Система автоматически создаст:

- landmark_system.pk1 (поисковая система)

3. Поиск по изображению
```bash
python landmark_system.py search_image --image path/to/your/image.jpg
```
4. Поиск по тексту
```bash
python landmark_system.py search_text --query "Эрмитаж"
```
Система выдаст сообщение:
```text
============================================================
TOP-5 IMAGES FOR: 'Эрмитаж'
Search method: CLIP
============================================================

1. Авиамеханический колледж (score: 0.3147)
   Category: architecture,historic_architecture,interesting_places,other_buildings_and_structures
   City: Владимир
   Path: C:\Users\424\Desktop\data\data\decoded_images\Vladimir\Vladimir_72_Авиамеханический колледж.jpg

2. Владимир (score: 0.3141)
   Category: historic,monuments_and_memorials,interesting_places,monuments
   City: Владимир
   Path: C:\Users\424\Desktop\data\data\decoded_images\Vladimir\Vladimir_1764_Владимир.jpg

3. Владимир (score: 0.3141)
   Category: historic,monuments_and_memorials,interesting_places,monuments
   City: Владимир
   Path: C:\Users\424\Desktop\data\data\decoded_images\Vladimir\Vladimir_1764_Владимир.jpg

4. Владимир (score: 0.3141)
   Category: historic,monuments_and_memorials,interesting_places,monuments
   City: Владимир
   Path: C:\Users\424\Desktop\data\data\decoded_images\Vladimir\Vladimir_1764_Владимир.jpg

5. Дворец вице-губернатора (score: 0.3105)
   Category: architecture,historic_architecture,interesting_places,other_buildings_and_structures
   City: Нижний Новгород
   Path: C:\Users\424\Desktop\data\data\decoded_images\NN\NN_489_Дворец вице-губернатора.jpg
```
### 📊 Поддерживаемые города по умолчанию
- EKB - Екатеринбург

- NN - Нижний Новгород

- Vladimir - Владимир

- Yaroslavl - Ярославль

### ⚙️ Параметры командной строки

- --data_dir - путь к директории с данными (по умолчанию: ./data)

- --top_k - количество результатов (по умолчанию: 5)

- --clean - очистка данных перед построением системы

# 📊 Пример вывода
Топ-5 названий:
1. Динамо (score: 4.0291)
2. Центральный стадион (score: 2.2179)
3. Владимирский академический областной драматический театр (score: 2.1981)
4. Театр музыкальной комедии (score: 0.7478)
5. улица Чернышевского (score: 0.7374)

Топ-5 категорий:
1. sport,architecture,historic_architecture,interesting_places,stadiums,other_buildings_and_structures (score: 6.2470)
2. cultural,theatres_and_entertainments,interesting_places,other_theatres (score: 2.1981)
3. cultural,museums,interesting_places,art_galleries (score: 1.4583)
4. cultural,theatres_and_entertainments,interesting_places,music_venues (score: 0.7478)
5. cultural,urban_environment,interesting_places,squares (score: 0.7374)

# 🏗️ Структура проекта
```text
landmark-search-system/
├── landmark_system.py     # Основной скрипт
├── requirements.txt       # Зависимости
├── README.md             # Документация
├── data/                 # Директория с данными
│   ├── decoded_images/   # Декодированные изображения
│   └── *.csv            # Исходные CSV файлы
└── *.pkl                # Сохраненные модели
```

# ⚠️ Примечания

- Для работы с GPU установите соответствующую версию PyTorch

- При первом запуске будет загружена модель CLIP (~400 МБ)

- Рекомендуется не менее 4 ГБ оперативной памяти

# 📄 Лицензия
MIT License

# 👤 Автор
Гераськин Андрей

GitHub: @Andrewkea

# 🙏 Благодарности
- OpenAI за модель CLIP

- Авторам датасетов достопримечательностей

- Сообществу open source

# ⭐ Если проект был полезен, поставьте звезду на GitHub!



















