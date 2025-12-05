#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
landmark_system.py
Система обработки данных достопримечательностей
"""

import os
import sys
import io
import argparse
import base64
import pickle
import math
import hashlib
import re
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter, defaultdict
import json

import numpy as np
import pandas as pd
from PIL import Image, ImageFile, UnidentifiedImageError
ImageFile.LOAD_TRUNCATED_IMAGES = True

import cv2
import imagehash
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# Конфигурация путей
# -------------------------
DATA_DIR = r"C:\Users\424\Desktop\data\data"
CITIES = {
    'EKB': 'Екатеринбург',
    'NN': 'Нижний Новгород', 
    'Vladimir': 'Владимир',
    'Yaroslavl': 'Ярославль'
}

# -------------------------
# Utility functions
# -------------------------
def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def read_image_from_base64(b64str: str) -> Optional[Image.Image]:
    """Чтение изображения из base64 строки с надежной обработкой ошибок."""
    try:
        # Проверка на пустую строку
        if pd.isna(b64str) or not b64str or str(b64str).strip() == '':
            return None
        
        b64str = str(b64str).strip()
        
        # Удаляем возможные префиксы типа "data:image/jpeg;base64,"
        if ',' in b64str:
            b64str = b64str.split(',')[1].strip()
        
        # Очищаем строку от пробелов и переводов строк
        b64str = ''.join(b64str.split())
        
        # Проверяем минимальную длину
        if len(b64str) < 20:
            return None
        
        # Добавляем padding если нужно
        # Base64 должен иметь длину, кратную 4
        pad_length = len(b64str) % 4
        if pad_length != 0:
            # Добавляем недостающие символы '='
            b64str += '=' * (4 - pad_length)
        
        # Проверяем, что строка содержит только допустимые символы base64
        if not re.match(r'^[A-Za-z0-9+/]*={0,2}$', b64str):
            # Попробуем удалить недопустимые символы
            b64str = re.sub(r'[^A-Za-z0-9+/=]', '', b64str)
            if len(b64str) < 20:
                return None
        
        # Пробуем декодировать
        try:
            data = base64.b64decode(b64str, validate=True)
        except base64.binascii.Error:
            # Если не удалось с validate=True, пробуем без валидации
            try:
                data = base64.b64decode(b64str, validate=False)
            except:
                return None
        
        # Проверяем минимальный размер данных
        if len(data) < 100:  # Минимум 100 байт для изображения
            return None
        
        # Пробуем открыть как изображение
        return Image.open(io.BytesIO(data)).convert('RGB')
        
    except Exception:
        # Любая ошибка - возвращаем None
        return None

def read_image_from_path(path: str) -> Optional[Image.Image]:
    """Чтение изображения из файла."""
    try:
        if pd.isna(path) or not path or not os.path.exists(path):
            return None
        return Image.open(path).convert('RGB')
    except Exception:
        return None

def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    if pil_img is None:
        raise ValueError("pil_img is None")
    arr = np.array(pil_img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def image_hash(ph: Image.Image) -> Optional[str]:
    try:
        return str(imagehash.phash(ph))
    except Exception:
        return None

def normalize_vec(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

# -------------------------
# Dataset Cleaner
# -------------------------
class EnhancedDatasetCleaner:
    """
    Автоматическая очистка изображений по эвристикам + OpenCV.
    """
    def __init__(self, thresholds: Dict[str, Any] = None, debug: bool = False):
        self.thresholds = thresholds or {
            'saturation': 15,
            'laplacian': 50,
            'brightness_low': 30,
            'brightness_high': 220,
            'min_resolution': 150,
            'min_aspect_ratio': 0.3,
            'max_aspect_ratio': 3.0,
            'edge_density_max': 0.6,
            'colorfulness': 8,
            'entropy': 4.5,
            'contrast': 25,
            'noise_threshold': 30,
            'min_unique_colors': 100,
            'max_file_size_kb': 50,
        }
        self.debug = debug

    def analyze_image(self, pil_img: Image.Image, file_path: str = None) -> Dict[str, Any]:
        """
        Анализ изображения на качество.
        """
        result = {'is_bad': False, 'reasons': [], 'metrics': {}}
        
        if pil_img is None:
            result['is_bad'] = True
            result['reasons'].append('cannot_read')
            return result

        try:
            # Проверка размера файла если указан путь
            if file_path and os.path.exists(file_path):
                file_size_kb = os.path.getsize(file_path) / 1024
                result['metrics']['file_size_kb'] = float(file_size_kb)
                if file_size_kb < self.thresholds['max_file_size_kb']:
                    result['reasons'].append('file_too_small')
            
            # convert
            cv = pil_to_cv2(pil_img)
            h, w = cv.shape[:2]
            result['metrics']['width'] = int(w)
            result['metrics']['height'] = int(h)
            result['metrics']['total_pixels'] = int(w * h)

            # resolution
            if min(h, w) < self.thresholds['min_resolution']:
                result['is_bad'] = True
                result['reasons'].append('low_resolution')

            # aspect ratio
            ar = w / (h + 1e-9)
            result['metrics']['aspect_ratio'] = float(ar)
            if ar < self.thresholds['min_aspect_ratio'] or ar > self.thresholds['max_aspect_ratio']:
                result['reasons'].append('weird_aspect_ratio')

            # brightness
            gray = cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)
            mean_b = float(np.mean(gray))
            result['metrics']['brightness'] = mean_b
            if mean_b < self.thresholds['brightness_low']:
                result['is_bad'] = True
                result['reasons'].append('too_dark')
            if mean_b > self.thresholds['brightness_high']:
                result['reasons'].append('too_bright')

            # blur (Laplacian var)
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            lap_var = float(lap.var())
            result['metrics']['laplacian'] = lap_var
            if lap_var < self.thresholds['laplacian']:
                result['is_bad'] = True
                result['reasons'].append('blurry')

            # colorfulness
            (B, G, R) = cv2.split(cv.astype('float'))
            rg = np.abs(R - G)
            yb = np.abs(0.5 * (R + G) - B)
            std_rg = np.std(rg)
            std_yb = np.std(yb)
            mean_rg = np.mean(rg)
            mean_yb = np.mean(yb)
            colorfulness = np.sqrt(std_rg ** 2 + std_yb ** 2) + 0.3 * np.sqrt(mean_rg ** 2 + mean_yb ** 2)
            result['metrics']['colorfulness'] = float(colorfulness)
            if colorfulness < self.thresholds['colorfulness']:
                result['reasons'].append('low_colorfulness')

            # saturation via HSV
            hsv = cv2.cvtColor(cv, cv2.COLOR_BGR2HSV)
            sat = float(np.mean(hsv[:, :, 1]))
            result['metrics']['saturation'] = sat
            if sat < self.thresholds['saturation']:
                result['reasons'].append('low_saturation')

            # entropy
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
            probs = hist / (hist.sum() + 1e-9)
            entropy = -np.sum([p * math.log2(p) for p in probs if p > 0])
            result['metrics']['entropy'] = float(entropy)
            if entropy < self.thresholds['entropy']:
                result['reasons'].append('low_entropy')

            # contrast (std)
            contrast = float(np.std(gray))
            result['metrics']['contrast'] = contrast
            if contrast < self.thresholds['contrast']:
                result['reasons'].append('low_contrast')

            # edges density
            edges = cv2.Canny(gray, 100, 200)
            edge_density = float(np.mean(edges > 0))
            result['metrics']['edge_density'] = edge_density
            if edge_density > self.thresholds['edge_density_max']:
                result['reasons'].append('possible_collage')

            # unique colors
            uniq = np.unique(cv.reshape(-1, 3), axis=0).shape[0]
            result['metrics']['unique_colors'] = int(uniq)
            if uniq < self.thresholds['min_unique_colors']:
                result['reasons'].append('few_colors')

            # watermark detection
            _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            small_cc = [c for c in cnts if 10 < cv2.contourArea(c) < (w * h) * 0.02]
            if len(small_cc) > 8:
                result['reasons'].append('possible_watermark')

            # grayscale detection
            is_grayscale = False
            if sat < (self.thresholds['saturation'] * 0.8):
                is_grayscale = True
            color_diff = np.mean(np.abs(R - G)) + np.mean(np.abs(G - B)) + np.mean(np.abs(B - R))
            if color_diff < 10:
                is_grayscale = True
            
            if is_grayscale:
                result['reasons'].append('grayscale')
                result['is_bad'] = True

            # noise estimation
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            hp = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            noise_est = float(np.std(hp))
            result['metrics']['noise'] = noise_est
            if noise_est > self.thresholds['noise_threshold']:
                result['reasons'].append('high_noise')

            # Hard rejection reasons
            hard_reasons = {'cannot_read', 'low_resolution', 'blurry', 'too_dark', 'grayscale'}
            if any(r in hard_reasons for r in result['reasons']):
                result['is_bad'] = True

        except Exception as e:
            result['is_bad'] = True
            result['reasons'].append(f'analysis_error: {str(e)[:50]}')

        if self.debug and result['is_bad']:
            print(f"Bad image analysis: {result['reasons']}")

        return result

# -------------------------
# Data Processor
# -------------------------
class DataProcessor:
    """
    Обработчик данных: декодирует изображения, объединяет с метаданными.
    """
    
    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, 'decoded_images')
        ensure_dir(self.images_dir)
        
    def process_city(self, city_code: str, city_name: str = None) -> pd.DataFrame:
        """
        Обрабатывает данные для одного города.
        Возвращает DataFrame с объединенными данными.
        """
        if city_name is None:
            city_name = CITIES.get(city_code, city_code)
        
        print(f"\n{'='*60}")
        print(f"Processing city: {city_name} ({city_code})")
        print('='*60)
        
        # Создаем папку для изображений города
        city_images_dir = os.path.join(self.images_dir, city_code)
        ensure_dir(city_images_dir)
        
        # Загружаем CSV файлы
        images_csv = os.path.join(self.data_dir, f"{city_code}_images.csv")
        places_csv = os.path.join(self.data_dir, f"{city_code}_places.csv")
        
        if not os.path.exists(images_csv):
            print(f"Warning: Images CSV not found: {images_csv}")
            return pd.DataFrame()
        
        # Читаем данные
        try:
            df_images = pd.read_csv(images_csv)
            print(f"Loaded images CSV: {len(df_images)} rows")
            
            # Проверяем наличие нужных колонок
            if 'name' not in df_images.columns or 'image' not in df_images.columns:
                print(f"Error: Required columns 'name' or 'image' not found in {images_csv}")
                print(f"Available columns: {list(df_images.columns)}")
                return pd.DataFrame()
            
            # Декодируем и сохраняем изображения
            print("Decoding images...")
            image_paths = []
            valid_indices = []
            failed_count = 0
            
            for idx, row in df_images.iterrows():
                if idx % 100 == 0:
                    print(f"  Processed {idx}/{len(df_images)} images... (failed: {failed_count})")
                
                img_name = row['name']
                if pd.isna(img_name):
                    img_name = f"image_{idx}"
                
                # Создаем безопасное имя файла
                safe_name = "".join(c for c in str(img_name) if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_name = safe_name[:100]
                img_filename = f"{city_code}_{idx}_{safe_name}.jpg"
                img_path = os.path.join(city_images_dir, img_filename)
                
                # Декодируем и сохраняем
                pil_img = read_image_from_base64(row['image'])
                if pil_img is not None:
                    try:
                        pil_img.save(img_path, 'JPEG', quality=90)
                        image_paths.append(img_path)
                        valid_indices.append(idx)
                    except Exception as e:
                        failed_count += 1
                        if failed_count % 100 == 0:
                            print(f"  Failed to save image {idx}: {e}")
                else:
                    failed_count += 1
            
            print(f"Successfully decoded {len(image_paths)} images, failed: {failed_count}")
            
            if not image_paths:
                print(f"Warning: No images successfully decoded for {city_code}")
                return pd.DataFrame()
            
            # Обновляем DataFrame с только успешно декодированными изображениями
            df_images = df_images.iloc[valid_indices].copy()
            df_images['image_path'] = image_paths
            df_images['city'] = city_name
            df_images['city_code'] = city_code
            
            # Загружаем places данные если есть
            if os.path.exists(places_csv):
                try:
                    df_places = pd.read_csv(places_csv)
                    print(f"Loaded places CSV: {len(df_places)} rows")
                    
                    # Объединяем данные
                    merged_df = self._merge_data(df_images, df_places)
                    return merged_df
                except Exception as e:
                    print(f"Error loading places CSV: {e}")
                    # Возвращаем только данные об изображениях
                    return self._add_empty_place_columns(df_images)
            else:
                print(f"Places CSV not found: {places_csv}")
                # Возвращаем только данные об изображениях
                return self._add_empty_place_columns(df_images)
                
        except Exception as e:
            print(f"Error processing city {city_code}: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def _add_empty_place_columns(self, df_images: pd.DataFrame) -> pd.DataFrame:
        """Добавляет пустые колонки places к DataFrame images."""
        df = df_images.copy()
        place_columns = ['XID', 'Name', 'Kind', 'City', 'OSM', 'WikiData', 'Rate', 'Lon', 'Lat']
        for col in place_columns:
            if col not in df.columns:
                df[col] = None
        return df
    
    def _merge_data(self, df_images: pd.DataFrame, df_places: pd.DataFrame) -> pd.DataFrame:
        """
        Объединяет данные images и places.
        """
        # Создаем копии
        df1 = df_images.copy()
        df2 = df_places.copy()
        
        # Стандартизируем названия колонок
        if 'name' in df1.columns:
            df1['name_lower'] = df1['name'].fillna('').astype(str).str.lower().str.strip()
        else:
            df1['name_lower'] = ''
        
        if 'Name' in df2.columns:
            df2['Name_lower'] = df2['Name'].fillna('').astype(str).str.lower().str.strip()
        else:
            df2['Name_lower'] = ''
        
        # Пробуем объединить по названию
        try:
            merged = pd.merge(df1, df2, left_on='name_lower', right_on='Name_lower', how='left')
        except Exception as e:
            print(f"Error merging data: {e}")
            # Если объединение не удалось, возвращаем images с пустыми place колонками
            merged = self._add_empty_place_columns(df1)
        
        # Удаляем вспомогательные колонки
        if 'name_lower' in merged.columns:
            merged = merged.drop('name_lower', axis=1)
        if 'Name_lower' in merged.columns:
            merged = merged.drop('Name_lower', axis=1)
        
        print(f"Merged data: {len(merged)} rows")
        
        # Считаем совпадения
        if 'XID' in merged.columns:
            matched = merged['XID'].notna().sum()
            print(f"Matched places: {matched} ({matched/len(merged)*100:.1f}%)")
        
        return merged
    
    def process_all_cities(self) -> pd.DataFrame:
        """
        Обрабатывает все города и создает единый датасет.
        """
        all_data = []
        
        for city_code in CITIES.keys():
            city_data = self.process_city(city_code)
            if not city_data.empty:
                all_data.append(city_data)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"\n{'='*60}")
            print(f"COMBINED DATASET SUMMARY:")
            print('='*60)
            print(f"Total images: {len(combined_df)}")
            
            # Статистика по городам
            if 'city' in combined_df.columns:
                city_stats = combined_df['city'].value_counts()
                print("\nImages per city:")
                for city, count in city_stats.items():
                    print(f"  {city}: {count}")
            
            # Статистика по наличию place информации
            if 'XID' in combined_df.columns:
                has_info = combined_df['XID'].notna().sum()
                print(f"\nImages with place info: {has_info} ({has_info/len(combined_df)*100:.1f}%)")
                print(f"Images without place info: {len(combined_df) - has_info}")
            
            # Сохраняем объединенный датасет
            output_path = os.path.join(self.data_dir, 'combined_dataset.csv')
            combined_df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"\nSaved combined dataset to: {output_path}")
            
            # Сохраняем статистику
            self._save_statistics(combined_df)
            
            return combined_df
        else:
            print("No data processed!")
            return pd.DataFrame()
    
    def _save_statistics(self, df: pd.DataFrame):
        """Сохраняет статистику по датасету."""
        stats = {}
        
        # Преобразуем все значения в сериализуемые типы
        stats['total_images'] = int(len(df))
        
        if 'city' in df.columns:
            city_counts = df['city'].value_counts()
            stats['cities'] = {str(city): int(count) for city, count in city_counts.items()}
        else:
            stats['cities'] = {}
        
        if 'XID' in df.columns:
            stats['has_place_info'] = int(df['XID'].notna().sum())
            stats['missing_place_info'] = int(df['XID'].isna().sum())
        else:
            stats['has_place_info'] = 0
            stats['missing_place_info'] = int(len(df))
        
        if 'name' in df.columns:
            stats['unique_names'] = int(df['name'].nunique())
        else:
            stats['unique_names'] = 0
        
        if 'Kind' in df.columns:
            stats['unique_categories'] = int(df['Kind'].nunique())
        else:
            stats['unique_categories'] = 0
        
        stats_path = os.path.join(self.data_dir, 'dataset_statistics.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"Statistics saved to: {stats_path}")

# -------------------------
# Landmark Search System
# -------------------------
class EnhancedLandmarkSearchSystem:
    """
    Система поиска достопримечательностей с использованием CLIP.
    """
    def __init__(self, device: str = None, debug: bool = False):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.debug = debug
        
        # Модель будет загружена позже
        self.clip_model = None
        self.processor = None
        self._nn_index = None
        
        # Данные
        self.df = None
        self.img_embeddings = None
        self.name_embeddings = None
        self.category_embeddings = None
        self.ids = None
        self.names = None
        self.categories = None
        self.city_names = None
        
        # TF-IDF для текстового поиска
        self.vectorizer = TfidfVectorizer(max_features=2000, stop_words=None)
        self._vectorizer_fitted = False
    
    def load_model(self):
        """Загрузка модели CLIP."""
        if self.clip_model is None:
            print(f"Loading CLIP model on device: {self.device}")
            try:
                from transformers import CLIPProcessor, CLIPModel
                self.clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
                self.processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
                self.clip_model.to(self.device)
                self.clip_model.eval()
                print("CLIP model loaded successfully")
            except ImportError:
                raise ImportError("Please install transformers: pip install transformers")
            except Exception as e:
                raise RuntimeError(f"Failed to load CLIP model: {e}")
    
    @torch.no_grad()
    def encode_image(self, pil_image: Image.Image) -> Optional[np.ndarray]:
        """Кодирование изображения в эмбеддинг."""
        if pil_image is None:
            return None
        
        self.load_model()
        
        try:
            inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
            img_features = self.clip_model.get_image_features(**inputs)
            vec = img_features.cpu().numpy().astype(np.float32).squeeze()
            return normalize_vec(vec)
        except Exception as e:
            if self.debug:
                print(f"Error encoding image: {e}")
            return None
    
    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> np.ndarray:
        """Кодирование текста в эмбеддинги."""
        if len(texts) == 0:
            return np.zeros((0, 512), dtype=np.float32)
        
        self.load_model()
        
        try:
            inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            txt_feats = self.clip_model.get_text_features(**inputs)
            vecs = txt_feats.cpu().numpy().astype(np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
            return vecs / norms
        except Exception as e:
            if self.debug:
                print(f"Error encoding text: {e}")
            return np.zeros((len(texts), 512), dtype=np.float32)
    
    def build_system(self, df: pd.DataFrame, save_path: str = 'landmark_system.pkl', 
                     clean_first: bool = True, max_items: int = None) -> str:
        """
        Построение системы поиска из датасета.
        """
        if df is None or df.empty:
            raise ValueError("DataFrame is empty")
        
        print(f"Building search system from {len(df)} images...")
        
        # Очистка данных если нужно
        if clean_first:
            print("Cleaning images...")
            cleaner = EnhancedDatasetCleaner(debug=False)
            good_indices = []
            
            for idx, row in df.iterrows():
                if 'image_path' in row and pd.notna(row['image_path']):
                    pil_img = read_image_from_path(row['image_path'])
                    analysis = cleaner.analyze_image(pil_img, row['image_path'])
                    if not analysis['is_bad']:
                        good_indices.append(idx)
                else:
                    # Пропускаем изображения без пути
                    continue
            
            if good_indices:
                df = df.iloc[good_indices].reset_index(drop=True)
                print(f"After cleaning: {len(df)} good images")
            else:
                print("Warning: No good images after cleaning!")
        
        if max_items is not None and max_items < len(df):
            df = df.head(max_items)
        
        # Кодирование изображений
        print("Encoding images...")
        image_vecs = []
        ids = []
        names = []
        categories = []
        city_names = []
        
        self.load_model()
        embedding_dim = self.clip_model.config.projection_dim
        
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                print(f"  Encoded {idx}/{len(df)} images...")
            
            img_path = row.get('image_path', None)
            pil_img = read_image_from_path(img_path) if img_path else None
            
            if pil_img is not None:
                emb = self.encode_image(pil_img)
                if emb is not None and emb.shape[0] == embedding_dim:
                    image_vecs.append(emb.astype(np.float32))
                    ids.append(idx)
                    
                    # Собираем метаданные
                    name = row.get('name', row.get('Name', ''))
                    category = row.get('Kind', row.get('category', ''))
                    city = row.get('city', row.get('City', ''))
                    
                    names.append(str(name) if pd.notna(name) else '')
                    categories.append(str(category) if pd.notna(category) else '')
                    city_names.append(str(city) if pd.notna(city) else '')
        
        if not image_vecs:
            raise ValueError("No valid images found for encoding")
        
        # Сохраняем данные
        self.df = df
        self.img_embeddings = np.vstack(image_vecs).astype(np.float32)
        self.ids = ids
        self.names = names
        self.categories = categories
        self.city_names = city_names
        
        # Кодируем названия и категории
        print("Encoding names and categories...")
        self.name_embeddings = self.encode_text(self.names)
        self.category_embeddings = self.encode_text(self.categories)
        
        # Обучение TF-IDF
        try:
            self.vectorizer.fit(self.names)
            self._vectorizer_fitted = True
        except Exception as e:
            print(f"Warning: TF-IDF training failed: {e}")
        
        # Создание индекса поиска
        print("Building search index...")
        self._nn_index = NearestNeighbors(
            n_neighbors=min(50, len(self.ids)),
            metric='cosine',
            algorithm='auto'
        )
        self._nn_index.fit(self.img_embeddings)
        
        # Сохранение системы
        data_to_save = {
            'df': self.df,
            'img_embeddings': self.img_embeddings,
            'name_embeddings': self.name_embeddings,
            'category_embeddings': self.category_embeddings,
            'ids': self.ids,
            'names': self.names,
            'categories': self.categories,
            'city_names': self.city_names
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(data_to_save, f)
        
        print(f"System saved to: {save_path}")
        print(f"Total embeddings: {len(self.ids)}")
        
        return save_path
    
    def load_system(self, path: str = 'landmark_system.pkl'):
        """Загрузка сохраненной системы."""
        print(f"Loading system from {path}...")
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.df = data['df']
        self.img_embeddings = data['img_embeddings']
        self.name_embeddings = data['name_embeddings']
        self.category_embeddings = data['category_embeddings']
        self.ids = data['ids']
        self.names = data['names']
        self.categories = data['categories']
        self.city_names = data.get('city_names', [])
        
        # Восстановление индекса поиска
        self._nn_index = NearestNeighbors(
            n_neighbors=min(50, len(self.ids)),
            metric='cosine',
            algorithm='auto'
        )
        self._nn_index.fit(self.img_embeddings)
        
        # Восстановление TF-IDF
        try:
            self.vectorizer.fit(self.names)
            self._vectorizer_fitted = True
        except Exception:
            print("Warning: TF-IDF vectorizer training failed")
        
        print(f"System loaded: {len(self.ids)} items")
    
    def search_by_image(self, query_image: Image.Image, top_k: int = 5) -> Dict[str, Any]:
        """
        Поиск по изображению (Задание 2).
        Возвращает топ-5 названий и топ-5 категорий.
        """
        if query_image is None:
            return {'error': 'query_image is None'}
        
        print("Encoding query image...")
        q_emb = self.encode_image(query_image)
        if q_emb is None:
            return {'error': 'cannot encode query image'}
        
        if self._nn_index is None:
            return {'error': 'system not initialized'}
        
        # Поиск похожих изображений
        k = min(top_k * 3, len(self.ids))
        dists, neigh_idx = self._nn_index.kneighbors([q_emb], n_neighbors=k)
        sims = 1 - dists[0]
        neigh_idx = neigh_idx[0]
        
        # Агрегация названий и категорий
        name_scores = defaultdict(float)
        category_scores = defaultdict(float)
        
        for i, score in zip(neigh_idx[:k], sims[:k]):
            name = self.names[i] if i < len(self.names) else ''
            cat = self.categories[i] if i < len(self.categories) else ''
            
            if name:
                name_scores[name] += float(score)
            if cat:
                category_scores[cat] += float(score)
        
        # Топ-5 названий
        top_names = sorted(name_scores.items(), key=lambda x: -x[1])[:top_k]
        
        # Топ-5 категорий
        top_categories = sorted(category_scores.items(), key=lambda x: -x[1])[:top_k]
        
        return {
            'top_names': top_names,
            'top_categories': top_categories,
            'query_embedding': q_emb.tolist() if q_emb is not None else None
        }
    
    def search_by_text(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Поиск по тексту (Задание 3).
        Возвращает топ-5 изображений.
        """
        if not query or not query.strip():
            return {'error': 'empty query'}
        
        print(f"Searching for: '{query}'")
        
        # Пробуем CLIP поиск
        q_vec = self.encode_text([query])[0]
        
        if not np.all(q_vec == 0):
            # Поиск по сходству с изображениями
            sims = cosine_similarity([q_vec], self.img_embeddings)[0]
            order = np.argsort(-sims)[:top_k]
            
            results = []
            for idx in order:
                row_idx = self.ids[idx]
                row = self.df.iloc[row_idx] if row_idx < len(self.df) else {}
                
                results.append({
                    'image_path': row.get('image_path', ''),
                    'name': self.names[idx] if idx < len(self.names) else '',
                    'category': self.categories[idx] if idx < len(self.categories) else '',
                    'city': self.city_names[idx] if idx < len(self.city_names) else '',
                    'score': float(sims[idx])
                })
            
            return {'results': results, 'method': 'CLIP'}
        else:
            # Fallback на TF-IDF
            if not self._vectorizer_fitted:
                return {'error': 'search system not ready'}
            
            try:
                qv = self.vectorizer.transform([query])
                names_mat = self.vectorizer.transform(self.names)
                sims = (qv @ names_mat.T).toarray().ravel()
                order = np.argsort(-sims)[:top_k]
                
                results = []
                for idx in order:
                    row_idx = self.ids[idx]
                    row = self.df.iloc[row_idx] if row_idx < len(self.df) else {}
                    
                    results.append({
                        'image_path': row.get('image_path', ''),
                        'name': self.names[idx] if idx < len(self.names) else '',
                        'category': self.categories[idx] if idx < len(self.categories) else '',
                        'city': self.city_names[idx] if idx < len(self.city_names) else '',
                        'score': float(sims[idx])
                    })
                
                return {'results': results, 'method': 'TF-IDF'}
                
            except Exception as e:
                return {'error': f'TF-IDF search failed: {str(e)}'}

# -------------------------
# Main CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Landmark Search System")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Команда обработки данных
    process_parser = subparsers.add_parser('process', help='Process all city data')
    process_parser.add_argument('--output_csv', type=str, default='combined_dataset.csv',
                               help='Output CSV path')
    
    # Команда очистки
    clean_parser = subparsers.add_parser('clean', help='Clean images')
    clean_parser.add_argument('--dataset_csv', type=str, default='combined_dataset.csv',
                             help='Dataset CSV path')
    clean_parser.add_argument('--good_dir', type=str, default='cleaned_good',
                             help='Directory for good images')
    clean_parser.add_argument('--bad_dir', type=str, default='cleaned_bad',
                             help='Directory for bad images')
    
    # Команда построения системы
    build_parser = subparsers.add_parser('build', help='Build search system')
    build_parser.add_argument('--dataset_csv', type=str, default='combined_dataset.csv',
                             help='Dataset CSV path')
    build_parser.add_argument('--output_model', type=str, default='landmark_system.pkl',
                             help='Output model path')
    build_parser.add_argument('--clean', action='store_true',
                             help='Clean data before building')
    build_parser.add_argument('--max_items', type=int, default=None,
                             help='Maximum items to process')
    
    # Команда поиска по изображению
    search_img_parser = subparsers.add_parser('search_image', help='Search by image')
    search_img_parser.add_argument('--model', type=str, default='landmark_system.pkl',
                                  help='Model path')
    search_img_parser.add_argument('--image', type=str, required=True,
                                  help='Query image path')
    search_img_parser.add_argument('--top_k', type=int, default=5,
                                  help='Number of results')
    
    # Команда поиска по тексту
    search_text_parser = subparsers.add_parser('search_text', help='Search by text')
    search_text_parser.add_argument('--model', type=str, default='landmark_system.pkl',
                                   help='Model path')
    search_text_parser.add_argument('--query', type=str, required=True,
                                   help='Search query')
    search_text_parser.add_argument('--top_k', type=int, default=5,
                                   help='Number of results')
    
    args = parser.parse_args()
    
    if args.command == 'process':
        print("Processing all city data...")
        processor = DataProcessor(DATA_DIR)
        combined_df = processor.process_all_cities()
        
        if not combined_df.empty:
            output_path = os.path.join(DATA_DIR, args.output_csv)
            combined_df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"\nCombined dataset saved to: {output_path}")
            print(f"Total images: {len(combined_df)}")
    
    elif args.command == 'clean':
        print("Cleaning images...")
        
        dataset_path = os.path.join(DATA_DIR, args.dataset_csv)
        if not os.path.exists(dataset_path):
            print(f"Error: Dataset not found at {dataset_path}")
            return
        
        df = pd.read_csv(dataset_path, encoding='utf-8')
        
        if 'image_path' not in df.columns:
            print("Error: 'image_path' column not found in dataset")
            return
        
        # Создаем папки для очищенных изображений
        good_dir = os.path.join(DATA_DIR, args.good_dir)
        bad_dir = os.path.join(DATA_DIR, args.bad_dir)
        ensure_dir(good_dir)
        ensure_dir(bad_dir)
        
        # Запускаем очистку
        cleaner = EnhancedDatasetCleaner(debug=True)
        
        good_images = []
        bad_images = []
        
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                print(f"Processed {idx}/{len(df)} images...")
            
            img_path = row['image_path']
            if pd.isna(img_path) or not os.path.exists(img_path):
                bad_images.append(img_path)
                continue
            
            pil_img = read_image_from_path(img_path)
            analysis = cleaner.analyze_image(pil_img, img_path)
            
            if analysis['is_bad']:
                bad_images.append(img_path)
            else:
                good_images.append(img_path)
        
        print(f"\nCleaning results:")
        print(f"Good images: {len(good_images)}")
        print(f"Bad images: {len(bad_images)}")
        
        # Сохраняем результаты очистки
        clean_df = df.copy()
        clean_df['clean_status'] = clean_df['image_path'].apply(
            lambda x: 'good' if x in good_images else 'bad'
        )
        
        clean_output = os.path.join(DATA_DIR, 'cleaned_dataset.csv')
        clean_df.to_csv(clean_output, index=False, encoding='utf-8')
        print(f"\nCleaned dataset saved to: {clean_output}")
    
    elif args.command == 'build':
        print("Building search system...")
        
        dataset_path = os.path.join(DATA_DIR, args.dataset_csv)
        if not os.path.exists(dataset_path):
            print(f"Error: Dataset not found at {dataset_path}")
            return
        
        df = pd.read_csv(dataset_path, encoding='utf-8')
        
        if df.empty:
            print("Error: Dataset is empty")
            return
        
        # Строим систему поиска
        system = EnhancedLandmarkSearchSystem()
        model_path = os.path.join(DATA_DIR, args.output_model)
        
        try:
            system.build_system(
                df=df,
                save_path=model_path,
                clean_first=args.clean,
                max_items=args.max_items
            )
            print(f"\nSearch system built successfully!")
            print(f"Model saved to: {model_path}")
        except Exception as e:
            print(f"Error building system: {e}")
            import traceback
            traceback.print_exc()
    
    elif args.command == 'search_image':
        print("Searching by image...")
        
        model_path = os.path.join(DATA_DIR, args.model)
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            return
        
        # Загружаем систему
        system = EnhancedLandmarkSearchSystem()
        system.load_system(model_path)
        
        # Загружаем изображение запроса
        query_img = read_image_from_path(args.image)
        if query_img is None:
            print(f"Error: Cannot load query image from {args.image}")
            return
        
        # Выполняем поиск
        result = system.search_by_image(query_img, top_k=args.top_k)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
            return
        
        # Выводим результаты
        print("\n" + "="*60)
        print("TOP-5 NAMES:")
        print("="*60)
        for i, (name, score) in enumerate(result['top_names'], 1):
            print(f"{i}. {name} (score: {score:.4f})")
        
        print("\n" + "="*60)
        print("TOP-5 CATEGORIES:")
        print("="*60)
        for i, (category, score) in enumerate(result['top_categories'], 1):
            print(f"{i}. {category} (score: {score:.4f})")
    
    elif args.command == 'search_text':
        print("Searching by text...")
        
        model_path = os.path.join(DATA_DIR, args.model)
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            return
        
        # Загружаем систему
        system = EnhancedLandmarkSearchSystem()
        system.load_system(model_path)
        
        # Выполняем поиск
        result = system.search_by_text(args.query, top_k=args.top_k)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
            return
        
        # Выводим результаты
        print("\n" + "="*60)
        print(f"TOP-{args.top_k} IMAGES FOR: '{args.query}'")
        print(f"Search method: {result.get('method', 'unknown')}")
        print("="*60)
        
        for i, res in enumerate(result.get('results', []), 1):
            print(f"\n{i}. {res['name']} (score: {res['score']:.4f})")
            print(f"   Category: {res['category']}")
            print(f"   City: {res['city']}")
            print(f"   Path: {res['image_path']}")
    
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  1. python landmark_system.py process")
        print("  2. python landmark_system.py clean")
        print("  3. python landmark_system.py build --clean")
        print("  4. python landmark_system.py search_image --image query.jpg")
        print("  5. python landmark_system.py search_text --query 'Эрмитаж'")


if __name__ == '__main__':
    main()