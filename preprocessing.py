import re
import json
import string
import requests
from io import StringIO

import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# === 1. Kamus dari GitHub KBBA ===
def load_kbba_dict():
    url = 'https://raw.githubusercontent.com/insomniagung/kamus_kbba/main/kbba.txt'
    response = requests.get(url)
    kbba_data = pd.read_csv(StringIO(response.text), delimiter='\t', names=['slang', 'formal'])
    return dict(zip(kbba_data['slang'], kbba_data['formal']))

# === 2. Kamus slang custom lokal ===
def load_custom_slang():
    with open("resources/slang_dict.json", "r", encoding="utf-8") as f:
        return json.load(f)

# === 3. Whitelist lokal (do_not_normalize) ===
def load_whitelist():
    with open("resources/do_not_normalize.txt", "r", encoding="utf-8") as f:
        return set(line.strip().lower() for line in f if line.strip())

# Gabungkan semua dalam satu fungsi load
def build_slang_dict():
    slang_kbba = load_kbba_dict()
    slang_custom = load_custom_slang()
    slang_kbba.update(slang_custom)  # Prioritaskan custom
    return slang_kbba

# Inisialisasi global
slang_dict = build_slang_dict()
do_not_normalize = load_whitelist()

# Load stopwords
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Stopword dari Sastrawi
factory = StopWordRemoverFactory()
stopwords_sastrawi = set(factory.get_stop_words())

# Stopword tambahan dari file custom
with open("resources/stopwords_custom.txt", "r", encoding="utf-8") as f:
    stopwords_custom = set(line.strip() for line in f.readlines())

# Gabungkan dua-duanya
stopwords_all = stopwords_sastrawi.union(stopwords_custom)

# Init stemmer
stemmer = StemmerFactory().create_stemmer()
stem_cache = {}

# === PREPROCESSING FUNCTIONS ===
# Fungsi untuk menghapus emoji
def remove_emoji(text):
    emoji_pattern = re.compile("[" 
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# Fungsi untuk membersihkan simbol khusus (opsional)
def remove_symbols(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# Fungsi utama untuk cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)         # mention
    text = re.sub(r'#[A-Za-z0-9_]+', '', text)         # hashtag
    text = re.sub(r'\bRT\b[\s]*', '', text)            # retweet
    text = re.sub(r'http\S+', '', text)                # URL
    text = re.sub(r'\d+', '', text)                    # angka
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)  # tanda baca
    text = text.strip()                                # hapus spasi di awal/akhir
    text = remove_emoji(text)                          # hapus emoji
    text = remove_symbols(text)                        # hapus simbol sisa (jika ada)
    text = re.sub(r"\s+", " ", text)                   # spasi berlebih
    return text.strip()

def normalize_slang(text):
    text = str(text).lower()
    words = text.split()
    normalized = [
        word if word in do_not_normalize else slang_dict.get(word, word)
        for word in words
    ]
    return ' '.join(normalized)

def stemming_text(text):
    words = text.split()
    stemmed = []
    for word in words:
        if word in stem_cache:
            stemmed.append(stem_cache[word])
        else:
            s = stemmer.stem(word)
            stem_cache[word] = s
            stemmed.append(s)
    return ' '.join(stemmed)

def remove_stopwords(text):
    words = text.split()
    return ' '.join([w for w in words if w not in stopwords_all])

# Load daftar akun yang ingin dikecualikan
with open("resources/accounts_to_exclude.txt", "r", encoding="utf-8") as f:
    accounts_to_exclude = set(line.strip().lower() for line in f if line.strip())

# Fungsi untuk filter akun (di level DataFrame, bukan per teks)
def filter_accounts(df, username_column="username"):
    if username_column in df.columns:
        df = df[~df[username_column].apply(lambda x: isinstance(x, str) and x.lower() in accounts_to_exclude)]
    return df

def full_preprocess(text):
    text = clean_text(text)
    text = normalize_slang(text)
    text = stemming_text(text)
    text = remove_stopwords(text)
    return text
