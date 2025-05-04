# 📌 Gerekli Kütüphaneler
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# 📌 NLTK Kaynakları (ilk çalıştırmada indirilmesi gerekir)
nltk.download('stopwords')
nltk.download('wordnet')

# 📌 ADIM 1: Veriyi Yükle
df = pd.read_csv("ADMISSIONS.csv")
diagnoses = df['diagnosis'].dropna().astype(str)

# 📌 Araçlar
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

lemmatized_texts = []
stemmed_texts = []

# 📌 ADIM 2: Ön İşleme + Lemmatization + Stemming
for text in diagnoses:
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]

    lemmas = [lemmatizer.lemmatize(t) for t in tokens]
    stems = [stemmer.stem(t) for t in tokens]

    lemmatized_texts.append(" ".join(lemmas))
    stemmed_texts.append(" ".join(stems))

# 📌 ADIM 3: DataFrame ve CSV Kaydetme
df_lemma = pd.DataFrame({
    "original_text": diagnoses.values,
    "cleaned_text": lemmatized_texts
})
df_stem = pd.DataFrame({
    "original_text": diagnoses.values,
    "cleaned_text": stemmed_texts
})

df_lemma.to_csv("lemmatized_diagnosis.csv", index=False)
df_stem.to_csv("stemmed_diagnosis.csv", index=False)

# 📌 ADIM 4: Zipf Yasası Grafiği Fonksiyonu
def plot_zipf(text_list, title):
    all_words = " ".join(text_list).split()
    freq = Counter(all_words)
    sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    ranks = np.log(range(1, len(sorted_freq) + 1))
    frequencies = np.log([f[1] for f in sorted_freq])

    plt.figure(figsize=(8, 5))
    plt.plot(ranks, frequencies, marker='o', linestyle='none', markersize=2)
    plt.title(f"Zipf Grafiği - {title}")
    plt.xlabel("Log(Sıralama)")
    plt.ylabel("Log(Frekans)")
    plt.grid(True)
    plt.show()

# 📌 ADIM 5: Zipf Grafiğini Çiz
plot_zipf(lemmatized_texts, "Lemmatization")
plot_zipf(stemmed_texts, "Stemming")

# 📌 ADIM 6: Veri Boyutu Karşılaştırma
print("Orijinal toplam kelime:", sum(len(t.split()) for t in diagnoses))
print("Lemmatized toplam kelime:", sum(len(t.split()) for t in lemmatized_texts))
print("Stemmed toplam kelime:", sum(len(t.split()) for t in stemmed_texts))
