import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. CSV dosyalarını oku
df_lemma = pd.read_csv("lemmatized_diagnosis.csv")
df_stem = pd.read_csv("stemmed_diagnosis.csv")

# 2. Doğru sütunlardan metinleri al
text_lemma = df_lemma['cleaned_text'].dropna().astype(str)
text_stem = df_stem['cleaned_text'].dropna().astype(str)

# 3. TF-IDF vektörleştiricileri oluştur
vectorizer_lemma = TfidfVectorizer()
vectorizer_stem = TfidfVectorizer()

# 4. Vektör matrislerini üret
tfidf_lemma = vectorizer_lemma.fit_transform(text_lemma)
tfidf_stem = vectorizer_stem.fit_transform(text_stem)

# 5. DataFrame'lere dönüştür
tfidf_lemma_df = pd.DataFrame(tfidf_lemma.toarray(), columns=vectorizer_lemma.get_feature_names_out())
tfidf_stem_df = pd.DataFrame(tfidf_stem.toarray(), columns=vectorizer_stem.get_feature_names_out())

# 6. CSV'ye kaydet
tfidf_lemma_df.to_csv("tfidf_lemmatized.csv", index=False)
tfidf_stem_df.to_csv("tfidf_stemmed.csv", index=False)

print("✅ TF-IDF dosyaları oluşturuldu.")
