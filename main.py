"""Complete Decision Tree workflow for UCI Banknote Authentication."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

from download_dataset import download_dataset


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
COLUMNS = ["variance", "skewness", "curtosis", "entropy", "class"]
FEATURES = COLUMNS[:-1]


def load_and_validate_data() -> pd.DataFrame:
    """Load the headerless source file and enforce known UCI dataset properties."""
    data_path = download_dataset()
    data = pd.read_csv(data_path, header=None, names=COLUMNS)

    print("\n=== PHASE 1: SETUP & PREPROCESSING ===")
    print(f"Dataset shape: {data.shape}")
    print("\nFirst five rows:")
    print(data.head().to_string(index=False))
    print("\nColumn names and data types:")
    print(data.dtypes.to_string())
    print("\nMissing values:")
    print(data.isna().sum().to_string())
    print(f"\nDuplicate rows: {data.duplicated().sum()}")
    print("\nClass distribution:")
    print(data["class"].value_counts().sort_index().to_string())

    if data.shape != (1372, 5):
        raise ValueError(f"Expected dataset shape (1372, 5), found {data.shape}.")
    if data[FEATURES].isna().any().any():
        raise ValueError("Feature columns contain missing values.")
    if not all(pd.api.types.is_numeric_dtype(data[col]) for col in FEATURES):
        raise TypeError("All feature columns must be numerical.")
    if set(data["class"].unique()) != {0, 1}:
        raise ValueError("Target must contain exactly labels 0 and 1.")

    print("\nPhase 1 data integrity: PASS")
    return data


def save_visualizations(data, model, matrix, train_accuracy, test_accuracy):
    """Save all required charts using lightweight Matplotlib figures."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    counts = data["class"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([str(x) for x in counts.index], counts.values, color=["#4472C4", "#ED7D31"])
    ax.bar_label(bars)
    ax.set(title="Class Distribution", xlabel="Class", ylabel="Number of Samples")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "class_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, matrix[row, col], ha="center", va="center", color="black")
    ax.set(
        title="Confusion Matrix",
        xlabel="Predicted Label",
        ylabel="Actual Label",
        xticks=[0, 1],
        yticks=[0, 1],
    )
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(16, 9))
    plot_tree(
        model,
        feature_names=FEATURES,
        class_names=["0", "1"],
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax,
    )
    ax.set_title("Decision Tree (max_depth=4)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "decision_tree.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Training", "Testing"], [train_accuracy, test_accuracy], color=["#70AD47", "#5B9BD5"])
    ax.bar_label(bars, fmt="%.4f")
    ax.set(title="Training vs Testing Accuracy", ylabel="Accuracy", ylim=(0, 1.08))
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "training_testing_accuracy.png", dpi=150)
    plt.close(fig)


def audit_and_compare_duplicates(data, train_indices, test_indices):
    """Audit exact cross-split duplicates and run a separate deduplicated comparison."""
    train_rows = data.loc[train_indices].reset_index(names="train_index")
    test_rows = data.loc[test_indices].reset_index(names="test_index")
    overlap = train_rows.merge(test_rows, on=COLUMNS, how="inner")
    audit = {
        "pair_matches": len(overlap),
        "unique_values": len(overlap[COLUMNS].drop_duplicates()),
        "train_rows": overlap["train_index"].nunique(),
        "test_rows": overlap["test_index"].nunique(),
    }

    clean_data = data.drop_duplicates().reset_index(drop=True)
    X_clean = clean_data[FEATURES]
    y_clean = clean_data["class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.20, random_state=42, stratify=y_clean
    )
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    matrix = confusion_matrix(y_test, test_predictions, labels=[0, 1])
    comparison = {
        "rows": len(clean_data),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "train_accuracy": accuracy_score(y_train, train_predictions),
        "test_accuracy": accuracy_score(y_test, test_predictions),
        "matrix": matrix,
        "precision": precision_score(y_test, test_predictions, pos_label=1, zero_division=0),
        "recall": recall_score(y_test, test_predictions, pos_label=1, zero_division=0),
        "f1": f1_score(y_test, test_predictions, pos_label=1, zero_division=0),
    }

    return audit, comparison


def write_analysis(
    data, train_accuracy, test_accuracy, matrix, precision, recall, f1,
    unrestricted, duplicate_audit, deduplicated,
):
    """Write an Indonesian academic summary using only computed values."""
    counts = data["class"].value_counts().sort_index()
    gap = train_accuracy - test_accuracy
    tn, fp, fn, tp = matrix.ravel()
    imbalance_ratio = counts.max() / counts.min()
    balance_text = (
        "relatif seimbang" if imbalance_ratio < 1.5 else "menunjukkan ketidakseimbangan yang perlu diperhatikan"
    )
    analysis = f"""# Analisis Akademik Banknote Authentication

## 1. Latar belakang dan rumusan masalah

Autentikasi uang kertas dapat dirumuskan sebagai klasifikasi biner berdasarkan karakteristik numerik hasil ekstraksi citra. Tujuan program ini adalah mempelajari pola untuk membedakan dua label pada data dan mengevaluasi kemampuan generalisasinya pada data uji.

## 2. Dataset dan fitur

Program memakai UCI Banknote Authentication Dataset yang berisi {len(data)} sampel. Empat fiturnya adalah variance, skewness, curtosis, dan entropy, sedangkan target memakai label 0 dan 1. Dataset memuat {counts[0]} sampel kelas 0 dan {counts[1]} sampel kelas 1. Halaman UCI menjelaskan data dan proses ekstraksi fiturnya, tetapi tidak mendefinisikan pemetaan label 0/1 secara cukup tegas untuk dijadikan klaim kelas asli atau palsu dalam laporan ini. Karena itu, interpretasi tetap memakai istilah kelas 0 dan kelas 1.

## 3. Alasan memilih Decision Tree

Decision Tree membagi data berulang kali berdasarkan nilai fitur untuk membentuk aturan keputusan. Algoritma ini dipilih karena mudah dipahami dan divisualisasikan, dapat menangani fitur numerik tanpa standardisasi, serta sesuai untuk klasifikasi biner. Batas `max_depth=4` digunakan untuk menjaga model tetap sederhana.

## 4. Prapemrosesan

File tanpa header diberi lima nama kolom yang sesuai, lalu diperiksa ukuran, tipe data, nilai kosong, duplikasi, dan label targetnya. Terdapat {data.duplicated().sum()} baris duplikat dan tidak ada nilai kosong. Baseline mempertahankan duplikasi agar hasil Phase 1–3 tetap dapat direproduksi. Data tidak diskalakan karena Decision Tree tidak memerlukannya.

## 5. Pelatihan model

Baseline membagi data 80% untuk pelatihan dan 20% untuk pengujian dengan `random_state=42` dan `stratify=y`. Model hanya melakukan `fit` pada 1.097 data pelatihan; 275 data pengujian hanya dipakai untuk evaluasi. Dengan demikian tidak ada pemakaian target pengujian dalam pelatihan atau pemilihan parameter model utama.

## 6. Accuracy dan Confusion Matrix

Akurasi pelatihan adalah {train_accuracy:.4f} ({train_accuracy:.2%}) dan akurasi pengujian adalah {test_accuracy:.4f} ({test_accuracy:.2%}). Confusion Matrix `[[{tn}, {fp}], [{fn}, {tp}]]` berarti {tn} data kelas 0 dan {tp} data kelas 1 diprediksi benar, sedangkan {fp} data kelas 0 diprediksi sebagai kelas 1 dan {fn} data kelas 1 diprediksi sebagai kelas 0.

## 7. Precision, recall, dan F1-score

Untuk kelas positif 1, precision sebesar {precision:.4f} menunjukkan proporsi prediksi kelas 1 yang benar. Recall sebesar {recall:.4f} menunjukkan proporsi data kelas 1 yang berhasil ditemukan, dan F1-score sebesar {f1:.4f} merangkum keseimbangan precision dan recall.

## 8. Overfitting dan perbandingan model

Distribusi kelas {balance_text} (rasio kelas terbesar terhadap terkecil {imbalance_ratio:.2f}). Oleh karena itu, akurasi cukup informatif tetapi tidak boleh digunakan sendirian; Confusion Matrix, precision, recall, dan F1-score tetap dibutuhkan untuk melihat kesalahan per kelas.

Selisih akurasi pelatihan dan pengujian model terbatas adalah {gap:.4f}. Pohon tanpa batas memperoleh akurasi pelatihan {unrestricted[0]:.4f} dan pengujian {unrestricted[1]:.4f}. Pada pembagian ini, pohon tanpa batas justru memiliki akurasi uji lebih tinggi daripada model `max_depth=4`; jadi tidak benar jika dikatakan bahwa pembatasan kedalaman meningkatkan akurasi. Nilai pelatihan 100% pada pohon tanpa batas menunjukkan kapasitas menghafal data yang lebih besar, tetapi satu train-test split belum cukup untuk menyimpulkan overfitting secara umum. Perbandingan ini bersifat ilustratif dan hasil uji tidak digunakan untuk memilih ulang model utama.

## 9. Audit duplikasi, hasil, dan keterbatasan

Audit baseline menemukan {duplicate_audit['test_rows']} baris uji yang mempunyai observasi identik di data latih, mencakup {duplicate_audit['unique_values']} nilai unik dan {duplicate_audit['pair_matches']} pasangan kecocokan lintas split. Kondisi ini merupakan potensi kebocoran karena sebagian data uji tidak sepenuhnya independen dan dapat membuat estimasi performa terlalu optimistis, walaupun tidak membuktikan besarnya pengaruh pada setiap prediksi.

Sebagai analisis terpisah, duplikasi dihapus sebelum pembagian sehingga tersisa {deduplicated['rows']} baris. Dengan konfigurasi yang sama, perbandingan ini menghasilkan akurasi pelatihan {deduplicated['train_accuracy']:.4f} ({deduplicated['train_accuracy']:.2%}), akurasi pengujian {deduplicated['test_accuracy']:.4f} ({deduplicated['test_accuracy']:.2%}), Confusion Matrix `{deduplicated['matrix'].tolist()}`, precision {deduplicated['precision']:.4f}, recall {deduplicated['recall']:.4f}, dan F1-score {deduplicated['f1']:.4f}. Hasil ini tidak menggantikan baseline karena ukuran dan komposisi split berubah.

Keterbatasan lain adalah evaluasi hanya memakai satu pembagian train-test, satu algoritma, dan empat fitur yang tersedia. Hasil belum membuktikan performa pada populasi lain dan arti label tidak boleh dinyatakan sebagai asli/palsu tanpa bukti otoritatif yang lebih tegas.

## 10. Kesimpulan

Baseline Decision Tree berkedalaman maksimum 4 menghasilkan akurasi uji {test_accuracy:.2%} dan F1-score kelas 1 sebesar {f1:.4f}. Model mudah dijelaskan dan memberikan hasil tinggi pada pembagian ini, tetapi interpretasi harus menyertakan potensi kebocoran duplikasi dan keterbatasan evaluasi. Perbandingan deduplikasi mendukung transparansi audit tanpa mengubah hasil baseline Phase 1–3.
"""
    (RESULTS_DIR / "analysis_id.md").write_text(analysis, encoding="utf-8")


def main():
    data = load_and_validate_data()
    X = data[FEATURES]
    y = data["class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    if len(X_train) != 1097 or len(X_test) != 275:
        raise AssertionError("Unexpected train-test split sizes.")
    print(f"Train shape: {X_train.shape}; Test shape: {X_test.shape}")

    print("\n=== PHASE 2: TRAINING & EVALUATION ===")
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)
    matrix = confusion_matrix(y_test, test_predictions, labels=[0, 1])
    precision = precision_score(y_test, test_predictions, pos_label=1, zero_division=0)
    recall = recall_score(y_test, test_predictions, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, test_predictions, pos_label=1, zero_division=0)

    print(f"Training accuracy: {train_accuracy:.6f}")
    print(f"Testing accuracy:  {test_accuracy:.6f}")
    print(f"Precision (class 1): {precision:.6f}")
    print(f"Recall (class 1):    {recall:.6f}")
    print(f"F1-score (class 1):  {f1:.6f}")
    print("\nConfusion Matrix (rows=actual, columns=predicted):")
    print(matrix)
    print("\nClassification Report:")
    print(classification_report(y_test, test_predictions, labels=[0, 1], digits=4, zero_division=0))

    unrestricted_model = DecisionTreeClassifier(random_state=42)
    unrestricted_model.fit(X_train, y_train)
    unrestricted = (
        unrestricted_model.score(X_train, y_train),
        unrestricted_model.score(X_test, y_test),
    )
    print(f"Unrestricted tree accuracy (train/test): {unrestricted[0]:.6f} / {unrestricted[1]:.6f}")
    if matrix.shape != (2, 2) or len(test_predictions) != len(y_test):
        raise AssertionError("Evaluation output has an invalid shape.")
    tn, fp, fn, tp = matrix.ravel()
    if not np.isclose(test_accuracy, (tn + tp) / matrix.sum()):
        raise AssertionError("Accuracy is inconsistent with the confusion matrix.")
    if not np.isclose(precision, tp / (tp + fp)):
        raise AssertionError("Precision is inconsistent with the confusion matrix.")
    if not np.isclose(recall, tp / (tp + fn)):
        raise AssertionError("Recall is inconsistent with the confusion matrix.")
    print("Phase 2 model evaluation: PASS")

    duplicate_audit, deduplicated = audit_and_compare_duplicates(
        data, X_train.index, X_test.index
    )

    print("\n=== PHASE 3: VISUALIZATION & ANALYSIS ===")
    save_visualizations(data, model, matrix, train_accuracy, test_accuracy)
    write_analysis(
        data, train_accuracy, test_accuracy, matrix, precision, recall, f1,
        unrestricted, duplicate_audit, deduplicated,
    )
    expected = [
        RESULTS_DIR / "class_distribution.png",
        RESULTS_DIR / "confusion_matrix.png",
        RESULTS_DIR / "decision_tree.png",
        RESULTS_DIR / "training_testing_accuracy.png",
        RESULTS_DIR / "analysis_id.md",
    ]
    missing = [path.name for path in expected if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Missing or empty result files: {missing}")
    print("Generated files:")
    for path in expected:
        print(f"- {path.relative_to(ROOT)}")
    print("Phase 3 visualization and analysis: PASS")

    print("\n=== PHASE 4: FINAL AUDIT ===")
    print(f"Cross-split duplicate pair matches: {duplicate_audit['pair_matches']}")
    print(f"Unique duplicated values across splits: {duplicate_audit['unique_values']}")
    print(f"Affected training rows: {duplicate_audit['train_rows']}")
    print(f"Affected testing rows: {duplicate_audit['test_rows']}")
    print("Separate comparison after drop_duplicates() before splitting:")
    print(
        f"Rows: {deduplicated['rows']}; train/test: "
        f"{deduplicated['train_rows']}/{deduplicated['test_rows']}"
    )
    print(f"Training/testing accuracy: {deduplicated['train_accuracy']:.6f} / {deduplicated['test_accuracy']:.6f}")
    print(f"Precision/recall/F1 (class 1): {deduplicated['precision']:.6f} / {deduplicated['recall']:.6f} / {deduplicated['f1']:.6f}")
    print(f"Confusion Matrix: {deduplicated['matrix'].tolist()}")
    print("Phase 4 final audit: PASS")
    print("\nOVERALL STATUS: PASS")


if __name__ == "__main__":
    main()
