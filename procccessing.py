# 📌 GEREKLİ KÜTÜPHANELER
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# 📌 NLTK VERİLERİNİ İNDİR (ilk seferde gerekir)
nltk.download('stopwords')
nltk.download('wordnet')

# 📌 1. DOSYAYI OKU
# ADMISSIONS.csv içinden 'diagnosis' sütununu al
df = pd.read_csv("ADMISSIONS.csv")
diagnoses = df['diagnosis'].dropna().astype(str)

# 📌 2. ARAÇLARI HAZIRLA
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# 📌 3. ÖN İŞLEME: Temizleme, Küçük Harfe Çevirme, Tokenizasyon, Stop Word, Lemma, Stem
processed_texts = []

for text in diagnoses:
    # Lowercasing
    text = text.lower()
    
    # HTML ve özel karakter temizliği
    text = re.sub(r'<[^>]+>', '', text)               # HTML etiketlerini temizle
    text = re.sub(r'[^a-z\s]', '', text)              # Noktalama ve özel karakter temizliği
    
    # Tokenization (split ile)
    tokens = text.split()
    
    # Stop word removal
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Stemming
    stemmed = [stemmer.stem(word) for word in lemmatized]
    
    # Temiz metni birleştir
    final_text = " ".join(stemmed)
    processed_texts.append(final_text)

# 📌 4. YENİ DATAFRAME OLUŞTUR
output_df = pd.DataFrame({
    "original_text": diagnoses.values,
    "cleaned_text": processed_texts
})

# 📌 5. CSV DOSYASINA KAYDET
output_df.to_csv("cleaned_diagnosis_texts.csv", index=False)

print("Temizlenmiş veriler 'cleaned_diagnosis_texts.csv' olarak kaydedildi.")
