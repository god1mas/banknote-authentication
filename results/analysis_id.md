# Analisis Akademik Banknote Authentication

## 1. Latar belakang dan rumusan masalah

Autentikasi uang kertas dapat dirumuskan sebagai klasifikasi biner berdasarkan karakteristik numerik hasil ekstraksi citra. Tujuan program ini adalah mempelajari pola untuk membedakan dua label pada data dan mengevaluasi kemampuan generalisasinya pada data uji.

## 2. Dataset dan fitur

Program memakai UCI Banknote Authentication Dataset yang berisi 1372 sampel. Empat fiturnya adalah variance, skewness, curtosis, dan entropy, sedangkan target memakai label 0 dan 1. Dataset memuat 762 sampel kelas 0 dan 610 sampel kelas 1. Halaman UCI menjelaskan data dan proses ekstraksi fiturnya, tetapi tidak mendefinisikan pemetaan label 0/1 secara cukup tegas untuk dijadikan klaim kelas asli atau palsu dalam laporan ini. Karena itu, interpretasi tetap memakai istilah kelas 0 dan kelas 1.

## 3. Alasan memilih Decision Tree

Decision Tree membagi data berulang kali berdasarkan nilai fitur untuk membentuk aturan keputusan. Algoritma ini dipilih karena mudah dipahami dan divisualisasikan, dapat menangani fitur numerik tanpa standardisasi, serta sesuai untuk klasifikasi biner. Batas `max_depth=4` digunakan untuk menjaga model tetap sederhana.

## 4. Prapemrosesan

File tanpa header diberi lima nama kolom yang sesuai, lalu diperiksa ukuran, tipe data, nilai kosong, duplikasi, dan label targetnya. Terdapat 24 baris duplikat dan tidak ada nilai kosong. Baseline mempertahankan duplikasi agar hasil Phase 1–3 tetap dapat direproduksi. Data tidak diskalakan karena Decision Tree tidak memerlukannya.

## 5. Pelatihan model

Baseline membagi data 80% untuk pelatihan dan 20% untuk pengujian dengan `random_state=42` dan `stratify=y`. Model hanya melakukan `fit` pada 1.097 data pelatihan; 275 data pengujian hanya dipakai untuk evaluasi. Dengan demikian tidak ada pemakaian target pengujian dalam pelatihan atau pemilihan parameter model utama.

## 6. Accuracy dan Confusion Matrix

Akurasi pelatihan adalah 0.9727 (97.27%) dan akurasi pengujian adalah 0.9709 (97.09%). Confusion Matrix `[[145, 8], [0, 122]]` berarti 145 data kelas 0 dan 122 data kelas 1 diprediksi benar, sedangkan 8 data kelas 0 diprediksi sebagai kelas 1 dan 0 data kelas 1 diprediksi sebagai kelas 0.

## 7. Precision, recall, dan F1-score

Untuk kelas positif 1, precision sebesar 0.9385 menunjukkan proporsi prediksi kelas 1 yang benar. Recall sebesar 1.0000 menunjukkan proporsi data kelas 1 yang berhasil ditemukan, dan F1-score sebesar 0.9683 merangkum keseimbangan precision dan recall.

## 8. Overfitting dan perbandingan model

Distribusi kelas relatif seimbang (rasio kelas terbesar terhadap terkecil 1.25). Oleh karena itu, akurasi cukup informatif tetapi tidak boleh digunakan sendirian; Confusion Matrix, precision, recall, dan F1-score tetap dibutuhkan untuk melihat kesalahan per kelas.

Selisih akurasi pelatihan dan pengujian model terbatas adalah 0.0017. Pohon tanpa batas memperoleh akurasi pelatihan 1.0000 dan pengujian 0.9927. Pada pembagian ini, pohon tanpa batas justru memiliki akurasi uji lebih tinggi daripada model `max_depth=4`; jadi tidak benar jika dikatakan bahwa pembatasan kedalaman meningkatkan akurasi. Nilai pelatihan 100% pada pohon tanpa batas menunjukkan kapasitas menghafal data yang lebih besar, tetapi satu train-test split belum cukup untuk menyimpulkan overfitting secara umum. Perbandingan ini bersifat ilustratif dan hasil uji tidak digunakan untuk memilih ulang model utama.

## 9. Audit duplikasi, hasil, dan keterbatasan

Audit baseline menemukan 9 baris uji yang mempunyai observasi identik di data latih, mencakup 7 nilai unik dan 19 pasangan kecocokan lintas split. Kondisi ini merupakan potensi kebocoran karena sebagian data uji tidak sepenuhnya independen dan dapat membuat estimasi performa terlalu optimistis, walaupun tidak membuktikan besarnya pengaruh pada setiap prediksi.

Sebagai analisis terpisah, duplikasi dihapus sebelum pembagian sehingga tersisa 1348 baris. Dengan konfigurasi yang sama, perbandingan ini menghasilkan akurasi pelatihan 0.9768 (97.68%), akurasi pengujian 0.9778 (97.78%), Confusion Matrix `[[142, 6], [0, 122]]`, precision 0.9531, recall 1.0000, dan F1-score 0.9760. Hasil ini tidak menggantikan baseline karena ukuran dan komposisi split berubah.

Keterbatasan lain adalah evaluasi hanya memakai satu pembagian train-test, satu algoritma, dan empat fitur yang tersedia. Hasil belum membuktikan performa pada populasi lain dan arti label tidak boleh dinyatakan sebagai asli/palsu tanpa bukti otoritatif yang lebih tegas.

## 10. Kesimpulan

Baseline Decision Tree berkedalaman maksimum 4 menghasilkan akurasi uji 97.09% dan F1-score kelas 1 sebesar 0.9683. Model mudah dijelaskan dan memberikan hasil tinggi pada pembagian ini, tetapi interpretasi harus menyertakan potensi kebocoran duplikasi dan keterbatasan evaluasi. Perbandingan deduplikasi mendukung transparansi audit tanpa mengubah hasil baseline Phase 1–3.
