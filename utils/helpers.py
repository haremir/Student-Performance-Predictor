# utils/helpers.py
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def load_data(file_path):
    """
    CSV dosyasından veri yükler
    
    Parameters:
    -----------
    file_path : str
        Veri dosyasının yolu
        
    Returns:
    --------
    pandas.DataFrame
        Yüklenen veri çerçevesi
    """
    return pd.read_csv(file_path)

def save_processed_data(data_dict, output_path):
    """
    İşlenmiş veriyi pickle formatında kaydeder
    
    Parameters:
    -----------
    data_dict : dict
        Kaydedilecek veri sözlüğü
    output_path : str
        Kaydedilecek dosya yolu
    """
    with open(output_path, 'wb') as f:
        pickle.dump(data_dict, f)

def load_processed_data(file_path):
    """
    İşlenmiş veriyi pickle formatından yükler
    
    Parameters:
    -----------
    file_path : str
        Yüklenecek dosya yolu
        
    Returns:
    --------
    dict
        Yüklenen veri sözlüğü
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_model(model, model_path, model_info=None):
    """
    Eğitilmiş modeli kaydeder
    
    Parameters:
    -----------
    model : object
        Kaydedilecek model nesnesi
    model_path : str
        Model dosya yolu
    model_info : dict, optional
        Model hakkında ek bilgiler
    """
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    if model_info:
        info_path = model_path.replace('.pkl', '_info.pkl')
        with open(info_path, 'wb') as f:
            pickle.dump(model_info, f)

def load_model(model_path):
    """
    Kaydedilmiş modeli yükler
    
    Parameters:
    -----------
    model_path : str
        Model dosya yolu
        
    Returns:
    --------
    object
        Yüklenen model nesnesi
    """
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def evaluate_model(model, X_test, y_test):
    """
    Modeli test verisi üzerinde değerlendirir
    
    Parameters:
    -----------
    model : object
        Değerlendirilecek model
    X_test : array-like
        Test özellikleri
    y_test : array-like
        Test etiketleri
        
    Returns:
    --------
    tuple
        (metrics_dict, y_pred) - Metrikler sözlüğü ve tahminler
    """
    # Tahminleri yap
    y_pred = model.predict(X_test)
    
    # Metrikleri hesapla
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    
    return metrics, y_pred

def plot_predictions(y_true, y_pred, title):
    """
    Gerçek değerler vs tahminleri görselleştirir
    
    Parameters:
    -----------
    y_true : array-like
        Gerçek değerler
    y_pred : array-like
        Tahmin edilen değerler
    title : str
        Grafik başlığı
        
    Returns:
    --------
    matplotlib.figure.Figure
        Oluşturulan grafik nesnesi
    """
    plt.figure(figsize=(10, 6))
    
    # Gerçek vs tahmin grafiği
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Mükemmel tahmin çizgisi
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title(title)
    plt.xlabel('Gerçek Değerler')
    plt.ylabel('Tahmin Edilen Değerler')
    plt.grid(True)
    
    return plt

def get_feature_importance(model, feature_names):
    """
    Modelden özellik önemlerini çıkarır
    
    Parameters:
    -----------
    model : object
        Eğitilmiş model
    feature_names : list
        Özellik isimleri
        
    Returns:
    --------
    dict
        Özellik-önem çiftleri
    """
    # Model türüne göre özellik önemini al
    if hasattr(model, 'coef_'):
        # Linear regression için
        importances = np.abs(model.coef_)
    elif hasattr(model, 'feature_importances_'):
        # Tree-based modeller için
        importances = model.feature_importances_
    else:
        return {}
    
    # Özellik-önem sözlüğü oluştur
    feature_importance = dict(zip(feature_names, importances))
    return feature_importance