# Banknote Authentication dengan Decision Tree

Proyek tugas Machine Learning ini melakukan klasifikasi biner pada **UCI Banknote Authentication Dataset** menggunakan `DecisionTreeClassifier`. Program berjalan di CPU dan menghasilkan evaluasi serta visualisasi yang dapat dijelaskan dalam presentasi akademik.

## 1. Latar belakang dan masalah

Autentikasi uang kertas dapat dipelajari sebagai klasifikasi berdasarkan karakteristik numerik hasil pengolahan citra. Tujuan proyek adalah membedakan dua kelas, mengukur performa pada data yang tidak dipakai saat pelatihan, dan membahas keterbatasannya secara transparan.

## 2. Dataset dan fitur

Sumber resmi: [UCI Machine Learning Repository — Banknote Authentication](https://archive.ics.uci.edu/dataset/267/banknote+authentication).

Dataset berisi 1.372 observasi dengan fitur `variance`, `skewness`, `curtosis`, dan `entropy`, serta target `class`. File asli tidak memiliki header. Halaman UCI menjelaskan fitur hasil ekstraksi citra spesimen uang kertas, tetapi pemetaan label 0 dan 1 ke istilah asli atau palsu tidak cukup tegas untuk dijadikan klaim dalam laporan ini. Karena itu, proyek hanya menyebut **kelas 0** dan **kelas 1**.

## 3. Alasan memilih Decision Tree

Decision Tree membentuk aturan bercabang berdasarkan nilai fitur. Algoritma ini mudah dipahami dan divisualisasikan, mendukung klasifikasi biner, dan tidak membutuhkan standardisasi. Baseline memakai `max_depth=4` dan `random_state=42` agar hasil dapat direproduksi dan pohon tetap terbaca.

## 4. Prapemrosesan

Program memeriksa ukuran, lima baris pertama, tipe kolom, nilai kosong, duplikasi, distribusi kelas, dan validitas label. Tidak ada nilai kosong dan terdapat 24 baris duplikat. Baseline mempertahankan semua baris untuk menjaga hasil Phase 1–3, sedangkan Phase 4 menyediakan perbandingan terpisah setelah duplikat dihapus sebelum split. Fitur tidak diskalakan karena Decision Tree tidak memerlukannya.

## 5. Pelatihan dan pemisahan data

Baseline dibagi secara stratified menjadi 1.097 data latih dan 275 data uji. Model hanya melakukan `fit` pada data latih; data uji hanya digunakan untuk evaluasi. Audit menemukan 9 baris uji mempunyai observasi identik pada data latih, mewakili 7 nilai unik dan 19 pasangan kecocokan lintas split. Ini merupakan potensi leakage karena sebagian data uji tidak sepenuhnya independen.

## 6. Accuracy dan Confusion Matrix baseline

- Akurasi pelatihan: **97,27%**
- Akurasi pengujian: **97,09%**
- Confusion Matrix: **`[[145, 8], [0, 122]]`**

Baris matriks adalah label aktual dan kolom adalah prediksi dalam urutan `[0, 1]`. Jadi, 145 data kelas 0 dan 122 data kelas 1 benar; 8 data kelas 0 diprediksi sebagai kelas 1; tidak ada kelas 1 yang diprediksi sebagai kelas 0.

## 7. Precision, recall, dan F1-score

Untuk kelas positif 1, precision **93,85%**, recall **100%**, dan F1-score **96,83%**. Dataset awal terdiri dari 762 kelas 0 dan 610 kelas 1, sehingga accuracy tetap perlu dibaca bersama matriks dan metrik per kelas.

## 8. Overfitting dan perbandingan model

Model terbatas memperoleh akurasi latih 97,27% dan uji 97,09%. Pohon tanpa batas memperoleh 100% dan 99,27%. Jadi, pembatasan kedalaman **tidak meningkatkan akurasi uji** pada eksperimen ini; manfaatnya adalah model lebih sederhana dan mudah dijelaskan. Akurasi latih 100% menunjukkan kapasitas menghafal yang lebih tinggi, tetapi akurasi uji yang tinggi berarti satu split ini belum cukup untuk menyatakan overfitting yang merugikan generalisasi. Diperlukan validasi silang atau data eksternal untuk kesimpulan lebih kuat.

## 9. Audit, hasil pembanding, dan keterbatasan

Setelah 24 duplikat dihapus sebelum split, tersisa 1.348 observasi: 1.078 latih dan 270 uji. Model kedalaman empat memperoleh akurasi latih **97,68%**, akurasi uji **97,78%**, Confusion Matrix **`[[142, 6], [0, 122]]`**, precision **95,31%**, recall **100%**, dan F1-score **97,60%**. Ini adalah pembanding, bukan pengganti baseline. Perubahan tidak dapat dianggap hanya akibat duplikasi karena komposisi split juga berubah.

Keterbatasan lain adalah satu split, satu dataset, satu algoritma utama, dan empat fitur. Hasil belum membuktikan performa pada populasi uang kertas atau proses akuisisi lain.

## 10. Kesimpulan

Decision Tree kedalaman empat memberikan performa baseline tinggi dan mudah diinterpretasikan. Perhitungan metrik konsisten dengan Confusion Matrix. Namun, laporan harus menyebut potensi leakage duplikasi, tidak menyatakan bahwa pruning meningkatkan akurasi, dan tidak mengklaim arti label tanpa sumber yang tegas.

## Struktur dan eksekusi

```text
banknote-authentication/
|-- data/data_banknote_authentication.txt
|-- results/
|   |-- analysis_id.md
|   |-- class_distribution.png
|   |-- confusion_matrix.png
|   |-- decision_tree.png
|   `-- training_testing_accuracy.png
|-- download_dataset.py
|-- main.py
|-- README.md
`-- requirements.txt
```

Gunakan Python 3.11 atau 3.12 di Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python download_dataset.py
python main.py
```

Setelah dataset tersimpan di `data`, program dapat dijalankan offline. Grafik disimpan di `results`, dan analisis aktual dibuat otomatis sebagai `results/analysis_id.md`.
