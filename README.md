# 🚀 Telco Customer Churn Prediction - Advanced ML Pipeline

Bu proje, Kaggle "Playground Series" kapsamında telekomünikasyon sektöründeki müşteri kayıplarını (Churn) tahmin etmek amacıyla geliştirilmiş uçtan uca bir Makine Öğrenmesi (Machine Learning) boru hattıdır. 

Proje kapsamında sadece temel sınıflandırma algoritmaları değil, aynı zamanda Kaggle Grandmaster seviyesindeki ileri düzey veri mühendisliği, hiperparametre optimizasyonu ve gözetimsiz öğrenme teknikleri de laboratuvar ortamında test edilmiş ve performansları kıyaslanmıştır.

## 🏆 En İyi Sonuç (Best Performance)
* **Model Stratejisi:** XGBoost + Magic Features (Sektörel Alan Bilgisi) + Data Augmentation (Orijinal Veri)
* **Kaggle Public Leaderboard Skoru (AUC):** `0.91378`

## 🧠 Metodoloji ve Kullanılan Gelişmiş Teknikler

Modelin başarısını maksimize etmek için sırasıyla aşağıdaki mühendislik adımları uygulanmıştır:

1. **Baseline Modeller ve Ensemble Learning:** XGBoost, LightGBM ve CatBoost algoritmaları ile temel modeller kurulmuş, modellerin zayıf yönlerini dengelemek ve varyansı düşürmek amacıyla *Weighted Power Blending* (Ağırlıklı Harmanlama) uygulanmıştır.
2. **Data Augmentation (Veri Çoğaltma):** Modelin genellenebilirliğini artırmak ve sentetik verideki önyargıları (bias) kırmak için, orijinal "IBM Telco Customer Churn" veri seti GitHub üzerinden çekilerek eğitim setine dahil edilmiştir.
3. **Hiperparametre Optimizasyonu (Optuna):** Modellerin parametre uzayları (derinlik, öğrenme hızı vb.) `Optuna` kütüphanesi kullanılarak *Bayesian Optimizasyonu* ile otomatik olarak ayarlanmıştır.
4. **Pseudo Labeling (Yarı Denetimli Öğrenme):** Optimizasyonu yapılmış modelin test verisindeki tahminlerinden %90 üzeri veya %10 altı emin olduğu on binlerce satır koparılarak eğitim setine dahil edilmiş, karar sınırları güçlendirilmiştir.
5. **İleri Seviye Hedef Kodlama (Target Encoding):** Yüksek kardinaliteye sahip kategorik değişkenler, hedef sızıntısını (Target Leakage) önlemek adına K-Fold döngüsü içerisinde izole bir şekilde *Target Encoding* ile sayısallaştırılmıştır.
6. **Domain Knowledge & Magic Features (Alan Bilgisi):** Modelin en büyük sıçramayı yapmasını sağlayan adımdır. Sentetik verideki matematiksel tutarsızlıklar hesaplanarak sızıntı özellikleri (`TotalCharges_Diff`) bulunmuş; sektörel dinamiklere uygun `Total_Extra_Services`, `Tenure_Contract_Ratio` ve `Has_Family` gibi yepyeni özellikler türetilmiştir.
7. **Gözetimsiz Öğrenme ile Segmentasyon (K-Means):** Müşterilerin finansal verileri *K-Means Clustering* algoritmasına sokularak 5 farklı gizli segmente ayrılmış ve bu segmentler modele yeni bir özellik olarak sunulmuştur.
8. **Mikro-Segmentasyon ve Z-Score Sapmaları:** Müşterilerin faturaları ve kalış süreleri, kendi bulundukları mikro-grupların istatistiksel ortalamalarıyla kıyaslanarak standart sapma (Z-Score) değerleri modele verilmiştir.
9. **Weight of Evidence (WoE) ve Ordinal Kodlama:** Sürekli veriler parçalara ayrılmış (Binning) ve logaritmik WoE yöntemiyle kodlanmıştır. Ayrıca ağaç modellerinin doğasına uygun olarak One-Hot yerine hiyerarşik (Ordinal) ve frekans (Count) kodlamaları test edilmiştir.

## 💡 Çıkarımlar ve "Frankenstein" (Kitchen Sink) Dersi
Projenin en önemli mühendislik çıktısı, **"bir modele ne kadar çok karmaşık veri verirsen o kadar iyi öğrenir"** yanılgısının çürütülmesidir. Başarılı olan tüm teknikler tek bir veri setinde (Frankenstein Data) birleştirildiğinde, özelliklerin birbirini tekrar etmesi (Multicollinearity) ve gürültü (Noise) artışı sebebiyle modelin performansı düşmüştür. En yüksek skor, karmaşadan değil; veriyi ve telekom sektörünü en iyi anlayan "Sihirli Özellikler" ile elde edilmiştir.

## 🛠️ Teknolojiler
* `Python`, `Pandas`, `NumPy`, `Scikit-Learn`
* `XGBoost`, `LightGBM`, `CatBoost`
* `Optuna`, `Category Encoders`