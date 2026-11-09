"""Streamlit dashboard for the existing Banknote Authentication assignment.

Read-only view: it never modifies main.py, the dataset or saved experiment charts.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'data_banknote_authentication.txt'
RESULTS = ROOT / 'results'
FEATURES = ['variance', 'skewness', 'curtosis', 'entropy']
COLUMNS = FEATURES + ['class']

st.set_page_config(page_title='Banknote ML Dashboard', page_icon='📊', layout='wide')
st.markdown('''<style>
.block-container {max-width: 1180px; padding-top: 1.6rem;}
[data-testid="stMetric"] {background: rgba(128,128,128,.065); border:1px solid rgba(128,128,128,.13); border-radius:13px; padding:14px;}
.hero-eyebrow {letter-spacing:.13em; font-size:.75rem; font-weight:750; color:#16a085;}
.hero-note {color: #888; font-size:.94rem;}
</style>''', unsafe_allow_html=True)

@st.cache_data

def get_data() -> pd.DataFrame:
    if not DATA.is_file():
        raise FileNotFoundError(f'Dataset tidak ditemukan: {DATA}. Jalankan python download_dataset.py dahulu.')
    df = pd.read_csv(DATA, header=None, names=COLUMNS)
    if df.shape != (1372, 5) or df[FEATURES].isna().any().any() or set(df['class'].unique()) != {0, 1}:
        raise ValueError('Dataset tidak sesuai dengan struktur UCI Banknote Authentication.')
    return df

@st.cache_data
def run_experiment(clean_duplicates: bool) -> dict:
    df = get_data()
    if clean_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
    X, y = df[FEATURES], df['class']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.20, random_state=42, stratify=y
    )
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return dict(
        rows=len(df), train_rows=len(X_train), test_rows=len(X_test),
        train_accuracy=model.score(X_train, y_train), test_accuracy=accuracy_score(y_test, pred),
        precision=precision_score(y_test, pred, pos_label=1, zero_division=0),
        recall=recall_score(y_test, pred, pos_label=1, zero_division=0),
        f1=f1_score(y_test, pred, pos_label=1, zero_division=0),
        matrix=confusion_matrix(y_test, pred, labels=[0, 1]),
    )

try:
    df = get_data()
    baseline = run_experiment(False)
    dedup = run_experiment(True)
except (FileNotFoundError, ValueError, OSError) as error:
    st.error(str(error))
    st.stop()

st.markdown('<div class="hero-eyebrow">MACHINE LEARNING / CLASSIFICATION</div>', unsafe_allow_html=True)
st.title('Banknote Authentication Dashboard')
st.markdown('Hasil klasifikasi **Decision Tree** dari dataset UCI Banknote Authentication. Berjalan lokal tanpa GPU.')
st.info('**Catatan label:** Sumber yang dipakai tidak menetapkan pemetaan 0/1 secara cukup jelas. Dashboard menggunakan istilah **kelas 0** dan **kelas 1**, bukan mengklaim langsung uang asli atau palsu.', icon='ℹ️')

page_overview, page_evaluation, page_visualization, page_dataset = st.tabs([
    '📌 Ringkasan', '🎯 Evaluasi model', '📈 Visualisasi', '🗂️ Dataset'
])

with page_overview:
    st.subheader('Performa model baseline')
    a, b, c, d = st.columns(4)
    a.metric('Akurasi testing', f"{baseline['test_accuracy']:.2%}")
    b.metric('Akurasi training', f"{baseline['train_accuracy']:.2%}")
    c.metric('F1-Score kelas 1', f"{baseline['f1']:.2%}")
    d.metric('Jumlah data', f"{len(df):,}".replace(',', '.'))
    st.caption('Model baseline: DecisionTreeClassifier(max_depth=4, random_state=42), split stratified 80:20.')
    st.subheader('Cara kerja program')
    st.write('**1.** Membaca 4 fitur numerik → **2.** Membagi data latih/uji → **3.** Melatih Decision Tree → **4.** Mengukur prediksi pada data uji.')
    st.subheader('Perbandingan eksperimen')
    comparison = pd.DataFrame([
        {'Eksperimen':'Baseline (dengan duplikat)', 'Data':baseline['rows'], 'Training':f"{baseline['train_accuracy']:.2%}", 'Testing':f"{baseline['test_accuracy']:.2%}"},
        {'Eksperimen':'Tanpa duplikat', 'Data':dedup['rows'], 'Training':f"{dedup['train_accuracy']:.2%}", 'Testing':f"{dedup['test_accuracy']:.2%}"},
    ])
    st.dataframe(comparison, hide_index=True, use_container_width=True)
    st.caption('Eksperimen tanpa duplikat merupakan pembanding terpisah, bukan pengganti baseline. Perbedaan performa juga dipengaruhi perubahan pembagian data.')

with page_evaluation:
    selection = st.radio('Pilih eksperimen', ['Baseline', 'Tanpa duplikat'], horizontal=True)
    r = baseline if selection == 'Baseline' else dedup
    x, y, z = st.columns(3)
    x.metric('Precision kelas 1', f"{r['precision']:.2%}")
    y.metric('Recall kelas 1', f"{r['recall']:.2%}")
    z.metric('F1-Score kelas 1', f"{r['f1']:.2%}")
    st.subheader('Confusion Matrix')
    matrix = r['matrix']
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.imshow(matrix, cmap='Blues')
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(matrix[row, col]), ha='center', va='center', fontsize=16)
    ax.set(xticks=[0,1], yticks=[0,1], xticklabels=['0','1'], yticklabels=['0','1'], xlabel='Prediksi', ylabel='Aktual')
    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)
    tn, fp, fn, tp = matrix.ravel()
    st.write(f'**Interpretasi:** {tn} data kelas 0 dan {tp} data kelas 1 diprediksi benar. Ada {fp} data kelas 0 diprediksi sebagai 1, dan {fn} data kelas 1 diprediksi sebagai 0.')

with page_visualization:
    st.subheader('Grafik dari eksperimen Python')
    st.caption('Grafik berikut berasal dari folder results/ yang dihasilkan oleh `python main.py`.')
    visuals = [
        ('class_distribution.png', 'Distribusi kelas'),
        ('confusion_matrix.png', 'Confusion Matrix baseline'),
        ('decision_tree.png', 'Pohon keputusan (max_depth=4)'),
        ('training_testing_accuracy.png', 'Akurasi training vs testing'),
    ]
    for filename, title in visuals:
        st.markdown(f'**{title}**')
        path = RESULTS / filename
        if path.is_file():
            st.image(str(path), use_container_width=True)
        else:
            st.warning(f'{filename} belum tersedia. Jalankan `python main.py` di terminal terlebih dahulu.')

with page_dataset:
    st.subheader('Informasi dataset')
    l, m, n = st.columns(3)
    l.metric('Jumlah observasi', len(df))
    m.metric('Fitur numerik', len(FEATURES))
    n.metric('Baris duplikat', int(df.duplicated().sum()))
    st.dataframe(df.head(20), use_container_width=True)
    st.subheader('Distribusi kelas')
    st.bar_chart(df['class'].value_counts().sort_index().rename(index={0:'Kelas 0',1:'Kelas 1'}))
    st.write('**Fitur:** variance, skewness, curtosis, entropy. Semua berasal dari ekstraksi citra dan bukan input foto langsung.')
    st.markdown('Sumber: [UCI Banknote Authentication Dataset](https://archive.ics.uci.edu/dataset/267/banknote+authentication)')
    st.warning('Pada split baseline terdapat 9 baris data uji yang identik dengan observasi di data latih. Hal ini berpotensi membuat evaluasi terlalu optimistis.')

st.divider()
st.caption('Dashboard tambahan untuk melihat hasil tugas. `main.py` tetap menjadi program utama Machine Learning.')
