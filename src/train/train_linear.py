"""
Linear Regression modelini eğitmek için script
"""
import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Ana dizini ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.helpers import save_model, evaluate_model

def train_linear_model(data_path, output_path, target='average_score'):
    """
    Linear Regression modelini eğitir ve kaydeder
    
    Parameters:
    -----------
    data_path : str
        İşlenmiş veri yolu
    output_path : str
        Eğitilmiş modelin kaydedileceği yol
    target : str
        Hedef değişken ('math_score', 'reading_score', 'writing_score', 'average_score')
    """
    # İşlenmiş veriyi yükle
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Hedef değişkeni seç
    if target == 'math_score':
        y_train = data['y_math_train']
        y_test = data['y_math_test']
    elif target == 'reading_score':
        y_train = data['y_reading_train']
        y_test = data['y_reading_test']
    elif target == 'writing_score':
        y_train = data['y_writing_train']
        y_test = data['y_writing_test']
    else:  # average_score
        y_train = data['y_avg_train']
        y_test = data['y_avg_test']
    
    # Eğitim verisini al (ölçeklendirilmiş veya normal)
    X_train = data['X_train_scaled']
    X_test = data['X_test_scaled']
    
    print(f"Linear Regression modeli eğitiliyor (hedef: {target})...")
    print(f"Eğitim veri boyutu: {X_train.shape}")
    
    # Modeli oluştur ve eğit
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Model performansını değerlendir
    metrics, y_pred = evaluate_model(model, X_test, y_test)
    
    print(f"Model performansı:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    
    # Modeli kaydet
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    
    save_model(model, output_path)
    
    # Model bilgilerini ve performans metriklerini kaydet
    model_info = {
        'model_type': 'LinearRegression',
        'target': target,
        'metrics': metrics,
        'feature_importance': dict(zip(data['feature_cols'], model.coef_)),
        'intercept': model.intercept_
    }
    
    info_path = output_path.replace('.pkl', '_info.pkl')
    with open(info_path, 'wb') as f:
        pickle.dump(model_info, f)
    
    print(f"Model bilgileri kaydedildi: {info_path}")
    
    return model, metrics

if __name__ == "__main__":
    data_path = "data/processed/processed_data.pkl"
    output_path = "models/linear_model.pkl"
    
    # Tüm hedef değişkenler için ayrı modeller eğit
    targets = ['math_score', 'reading_score', 'writing_score', 'average_score']
    
    for target in targets:
        model_path = output_path.replace('.pkl', f'_{target}.pkl')
        train_linear_model(data_path, model_path, target)