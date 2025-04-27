"""
Veri önişleme işlemlerini içeren script
"""
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(input_file, output_dir, test_size=0.2, random_state=42):
    """
    Veri setini yükler, temizler, dönüştürür ve kaydeder
    """
    # Veriyi yükle
    print(f"Veri yükleniyor: {input_file}")
    data = pd.read_csv(input_file)
    
    print(f"Orijinal veri şekli: {data.shape}")
    
    # Eksik değerleri kontrol et
    missing_values = data.isnull().sum()
    print("Eksik değerler:")
    print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "Eksik değer yok")
    
    # Kategorik değişkenleri encode et
    categorical_cols = ['gender', 'race/ethnicity', 'parental level of education', 
                        'lunch', 'test preparation course']
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data[col + '_encoded'] = le.fit_transform(data[col])
        label_encoders[col] = le
    
    # Hedef değişkenleri belirle - 3 farklı score için ortalama alalım
    data['average_score'] = (data['math score'] + data['reading score'] + data['writing score']) / 3
    
    # Özellik ve hedef değişkenleri ayır
    feature_cols = [col + '_encoded' for col in categorical_cols]
    X = data[feature_cols]
    
    # 3 farklı hedef değişken için veri setleri oluştur
    y_math = data['math score']
    y_reading = data['reading score']
    y_writing = data['writing score']
    y_avg = data['average_score']
    
    # Veriyi eğitim ve test setlerine böl
    X_train, X_test, y_math_train, y_math_test = train_test_split(
        X, y_math, test_size=test_size, random_state=random_state)
    
    _, _, y_reading_train, y_reading_test = train_test_split(
        X, y_reading, test_size=test_size, random_state=random_state)
    
    _, _, y_writing_train, y_writing_test = train_test_split(
        X, y_writing, test_size=test_size, random_state=random_state)
    
    _, _, y_avg_train, y_avg_test = train_test_split(
        X, y_avg, test_size=test_size, random_state=random_state)
    
    # Özellikleri ölçeklendir
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # İşlenmiş verileri kaydet
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Eğitim ve test verilerini kaydet
    processed_data = {
        'X_train': X_train,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_math_train': y_math_train,
        'y_math_test': y_math_test,
        'y_reading_train': y_reading_train,
        'y_reading_test': y_reading_test,
        'y_writing_train': y_writing_train,
        'y_writing_test': y_writing_test,
        'y_avg_train': y_avg_train,
        'y_avg_test': y_avg_test,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_cols': feature_cols
    }
    
    # Pickle dosyası olarak kaydet
    import pickle
    output_file = os.path.join(output_dir, 'processed_data.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(processed_data, f)
    
    print(f"İşlenmiş veriler kaydedildi: {output_file}")
    
    # İşlenmiş veriyi CSV olarak da kaydet (görüntülemek için)
    processed_df = X.copy()
    processed_df['math_score'] = y_math
    processed_df['reading_score'] = y_reading
    processed_df['writing_score'] = y_writing
    processed_df['average_score'] = y_avg
    
    csv_output = os.path.join(output_dir, 'processed_data.csv')
    processed_df.to_csv(csv_output, index=False)
    
    print(f"İşlenmiş veriler CSV olarak kaydedildi: {csv_output}")
    
    return processed_data

if __name__ == "__main__":
    input_file = r"C:\Users\PC\Desktop\student_performance\data\raw\StudentsPerformance.csv"
    output_dir = r"C:\Users\PC\Desktop\student_performance\data\processed"
    preprocess_data(input_file, output_dir)