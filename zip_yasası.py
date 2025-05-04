# 📌 Gerekli Kütüphaneleri Yükle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import Counter

def zipf_plot_from_file(file_path, text_column, title, output_filename):
    # CSV dosyasını oku
    df = pd.read_csv(file_path)

    # İlgili metin sütununu al, boşları çıkar
    texts = df[text_column].dropna().astype(str)

    # Tüm metinleri birleştir
    full_text = " ".join(texts)

    # Küçük harfe çevir, noktalama ve özel karakterleri sil
    clean_text = re.sub(r'[^a-zA-Z\s]', '', full_text.lower())

    # Kelimelere ayır
    words = clean_text.split()

    # Kelime frekanslarını hesapla
    word_freq = Counter(words)
    sorted_freqs = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

    # Sıralama ve frekansların log'larını al
    ranks = np.log(range(1, len(sorted_freqs) + 1))
    frequencies = np.log([freq for word, freq in sorted_freqs])

    # Grafik çiz
    plt.figure(figsize=(10, 6))
    plt.plot(ranks, frequencies, marker='o', linestyle='none', markersize=3)
    plt.xlabel("Log(Sıralama)")
    plt.ylabel("Log(Frekans)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.show()

# 📌 Lemmatized için grafik çiz
zipf_plot_from_file(
    file_path="lemmatized_diagnosis.csv",
    text_column="cleaned_text",
    title="Zipf Yasası - Lemmatized Metinler",
    output_filename="zipf_lemmatized_scatter.png"
)

# 📌 Stemmed için grafik çiz
zipf_plot_from_file(
    file_path="stemmed_diagnosis.csv",
    text_column="cleaned_text",
    title="Zipf Yasası - Stemmed Metinler",
    output_filename="zipf_stemmed_scatter.png"
)
