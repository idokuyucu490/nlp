# 🧠 Medical NLP Diagnosis Vectors with Word2Vec & TF-IDF

Bu proje, sağlık alanında yer alan **hastalık tanıları (diagnosis)** metinlerini kullanarak **doğal dil işleme (NLP)** teknikleriyle vektörleştirme (TF-IDF ve Word2Vec) işlemlerini gerçekleştirmeyi amaçlamaktadır.

## 📁 Veri Seti

Projenin temelinde, [MIMIC-III (ADMISSIONS.csv, PRESCRIPTIONS.csv)](https://physionet.org/content/mimiciii/1.4/) veri setinden alınan tanı metinleri (`DIAGNOSIS`) yer almaktadır. Bu metinler hem **ham**, hem de **ön işlenmiş (cleaned, lemmatized, stemmed)** versiyonlarıyla kullanılmıştır.

---

## 🎯 Amaç

- Metinleri temizleyerek ön işleme süreçlerini uygulamak (tokenization, stopword removal, stemming, lemmatization)
- Temizlenmiş veri üzerinden:
  - TF-IDF vektörleri oluşturmak
  - Farklı parametrelerle 16 farklı Word2Vec modeli eğitmek
- Zipf Yasası analizini gerçekleştirmek
- Sonuçları yorumlayarak en etkili model yapılandırmasını tespit etmek

---

## 📌 Kurulum adimlari

Aşağıdaki komutlarla tüm bağımlılıkları kurabilirsiniz:

```bash
pip install pandas numpy matplotlib scikit-learn gensim nltk spacy
python -m nltk.downloader stopwords punkt
python -m spacy download en_core_web_sm

# 🔹 1. Veriyi Temizle
$ python temizle.py
📄 ADMISSIONS.csv dosyası yüklendi.
✅ Temizlenmiş veriler oluşturuldu:
   → cleaned_diagnosis.csv
   → lemmatized_diagnosis.csv
   → stemmed_diagnosis.csv

# 🔹 2. TF-IDF Vektörleri Oluştur
$ python tfidf_olustur.py
✅ TF-IDF başarıyla oluşturuldu:
   → tfidf_lemmatized.csv
   → tfidf_stemmed.csv

# 🔹 3. Word2Vec Model Eğitimi
$ python word2vec_training.py
🔁 16 farklı model eğitiliyor...
✅ Kaydedildi: word2vec_lemmatized_skipgram_win4_dim300.model
🕒 Eğitim süresi: 10.34 saniye
💾 Model boyutu: 8.57 MB
🔍 En benzer kelimeler: ['pain', 'ache', 'discomfort', 'fever', 'burning']
...

# 🔹 4. Zipf Yasası Log-Log Grafiği
$ python zipf_yasasi_ham_veri.py
📊 Grafik oluşturuluyor...
✅ Zipf yasası grafiği başarıyla gösterildi.
🧠 Grafik yorum: Zipf yasasına uygun log-log düz çizgi elde edildi.

 
