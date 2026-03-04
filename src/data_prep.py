import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def load_and_prep_data(train_path="../data/raw/train.csv", test_path="../data/raw/test.csv", use_orig_data=True):
    """
    Kaggle Telco Customer Churn verisini yükler, orijinal IBM verisiyle birleştirir
    ve ileri seviye Feature Engineering (Sihirli Özellikler) uygular.
    """
    print("Veriler yükleniyor...")
    train = pd.read_csv(train_path, index_col='id')
    test = pd.read_csv(test_path, index_col='id')
    
    # 1. Orijinal Veriyi Ekleme (Data Augmentation)
    if use_orig_data:
        print("Orijinal IBM verisi indiriliyor ve birleştiriliyor...")
        url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
        orig = pd.read_csv(url).rename(columns={'customerID': 'id'}).set_index('id')
        
        y_train = train['Churn'].map({'Yes': 1, 'No': 0})
        train.drop('Churn', axis=1, inplace=True)
        y_orig = orig['Churn'].map({'Yes': 1, 'No': 0})
        orig.drop('Churn', axis=1, inplace=True)
        
        X_train_full = pd.concat([train, orig])
        y_train_full = pd.concat([y_train, y_orig])
    else:
        y_train_full = train['Churn'].map({'Yes': 1, 'No': 0})
        X_train_full = train.drop('Churn', axis=1)

    # İşlemleri tek seferde yapmak için test ile birleştir
    df_all = pd.concat([X_train_full, test], keys=['train', 'test'])
    
    # 2. Veri Temizleme (Sıfırıncı Ay Düzeltmesi)
    df_all['TotalCharges'] = pd.to_numeric(df_all['TotalCharges'], errors='coerce')
    df_all.loc[df_all['tenure'] == 0, 'TotalCharges'] = df_all['MonthlyCharges']
    df_all['TotalCharges'].fillna(0, inplace=True)
    
    # 3. Sihirli Özellikler (Magic Features)
    print("Feature Engineering (Sihirli Özellikler) uygulanıyor...")
    df_all['TotalCharges_Diff'] = df_all['TotalCharges'] - (df_all['MonthlyCharges'] * df_all['tenure'])
    
    contract_map = {'Month-to-month': 1, 'One year': 12, 'Two year': 24}
    df_all['Contract_Months'] = df_all['Contract'].map(contract_map)
    df_all['Tenure_Contract_Ratio'] = df_all['tenure'] / df_all['Contract_Months']
    
    ek_hizmetler = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df_all['Total_Extra_Services'] = (df_all[ek_hizmetler] == 'Yes').sum(axis=1)
    
    df_all['Auto_Payment'] = df_all['PaymentMethod'].isin(['Bank transfer (automatic)', 'Credit card (automatic)']).astype(int)
    df_all['Has_Family'] = ((df_all['Partner'] == 'Yes') | (df_all['Dependents'] == 'Yes')).astype(int)
    
    # 4. Kategorik Değişkenleri Kodlama (Sadece sayısal ve kategorik olarak bırakıyoruz, modelde Target Encoding yapılacak)
    cat_cols = df_all.select_dtypes(include=['object']).columns.tolist()
    
    # Veriyi tekrar Train ve Test olarak ayır
    X_train_final = df_all.xs('train')
    X_test_final = df_all.xs('test')
    
    print(f"Veri hazırlığı tamamlandı! Toplam Sütun: {X_train_final.shape[1]}")
    return X_train_final, y_train_full, X_test_final, cat_cols



# Örnek kullanım 
#from src.data_prep import load_and_prep_data
#X_train, y_train, X_test, cat_cols = load_and_prep_data()