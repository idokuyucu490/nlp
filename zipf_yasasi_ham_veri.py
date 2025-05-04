import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import Counter

# 1. CSV'den veri oku
df = pd.read_csv("lemmatized_diagnosis.csv")
texts = df["original_text"].dropna().astype(str)

# 2. Temizleme
full_text = " ".join(texts)
clean_text = re.sub(r'[^a-zA-Z\s]', '', full_text.lower())
words = clean_text.split()

# 3. Kelime frekanslarını hesapla
word_freq = Counter(words)
sorted_freqs = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

# 🔁 Olası max kelime sayısını belirle
actual_n = len(sorted_freqs)
top_n = min(200, actual_n)  # Eğer 200'den az varsa ona göre ayarla

top_words = sorted_freqs[:top_n]

# 4. Log-log veri
ranks = np.log(np.arange(1, top_n + 1))
frequencies = np.log([freq for _, freq in top_words])

# 5. Grafik çizimi
plt.figure(figsize=(10, 6))
plt.plot(ranks, frequencies, marker='o', linestyle='-', markersize=4, color='darkgreen')
plt.xlabel("Log(Sıralama)")
plt.ylabel("Log(Frekans)")
plt.title("Zipf Yasası - Ham Veri (original_text)")
plt.grid(True)
plt.tight_layout()
plt.savefig("zipf_lineer_fixed.png")
plt.show()
