import pandas as pd
from gensim.models import Word2Vec
import os
import time

# 1. Verileri yükle
df_lemma = pd.read_csv("lemmatized_diagnosis.csv")
df_stem = pd.read_csv("stemmed_diagnosis.csv")

# 2. Tokenize et
tokenized_lemma = df_lemma['cleaned_text'].dropna().astype(str).apply(lambda x: x.split()).tolist()
tokenized_stem = df_stem['cleaned_text'].dropna().astype(str).apply(lambda x: x.split()).tolist()

# 3. Parametre seti
parameters = [
    {'model_type': 'cbow', 'window': 2, 'vector_size': 100},
    {'model_type': 'skipgram', 'window': 2, 'vector_size': 100},
    {'model_type': 'cbow', 'window': 4, 'vector_size': 100},
    {'model_type': 'skipgram', 'window': 4, 'vector_size': 100},
    {'model_type': 'cbow', 'window': 2, 'vector_size': 300},
    {'model_type': 'skipgram', 'window': 2, 'vector_size': 300},
    {'model_type': 'cbow', 'window': 4, 'vector_size': 300},
    {'model_type': 'skipgram', 'window': 4, 'vector_size': 300}
]

# 4. Eğitim ve CSV kaydetme fonksiyonu
def train_and_export(data, label_prefix):
    for params in parameters:
        sg = 0 if params['model_type'] == 'cbow' else 1
        model_name = f"word2vec_{label_prefix}_{params['model_type']}_win{params['window']}_dim{params['vector_size']}"
        model_file = model_name + ".model"
        csv_file = model_name + ".csv"

        print(f"\n🔧 Eğitim başlatılıyor: {model_file}")
        start_time = time.time()

        model = Word2Vec(
            sentences=data,
            vector_size=params['vector_size'],
            window=params['window'],
            sg=sg,
            min_count=1,
            workers=1,
            epochs=10
        )

        model.save(model_file)
        end_time = time.time()
        duration = round(end_time - start_time, 2)
        size_mb = round(os.path.getsize(model_file) / (1024 * 1024), 2)

        print(f"✅ Model kaydedildi: {model_file}")
        print(f"⏱ Süre: {duration} saniye, 💾 Boyut: {size_mb} MB")

        # 🔽 Vektörleri DataFrame olarak dışa aktar
        words = model.wv.index_to_key
        vectors = [model.wv[word] for word in words]
        df_vectors = pd.DataFrame(vectors, index=words)
        df_vectors.to_csv(csv_file)

        print(f"📄 Vektör CSV oluşturuldu: {csv_file}")

        # Örnek: pain kelimesine en yakın 5 kelime
        example_word = "pain"
        if example_word in model.wv:
            print(f"🔍 '{example_word}' için en benzer 5 kelime:")
            print(model.wv.most_similar(example_word, topn=5))
        else:
            print(f"⚠️ '{example_word}' kelimesi modelde bulunamadı.")

# 5. Lemmatized ve Stemmed için modelleri eğit ve CSV'ye aktar
train_and_export(tokenized_lemma, "lemmatized")
train_and_export(tokenized_stem, "stemmed")
