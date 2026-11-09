"""Download the original UCI Banknote Authentication dataset."""

from pathlib import Path
from urllib.request import urlopen


DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00267/data_banknote_authentication.txt"
)
DATA_PATH = Path(__file__).resolve().parent / "data" / "data_banknote_authentication.txt"


def download_dataset(force: bool = False) -> Path:
    """Download the dataset unless a local copy already exists."""
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists() and not force:
        print(f"Dataset sudah tersedia: {DATA_PATH}")
        return DATA_PATH

    print(f"Mengunduh dataset dari {DATA_URL}")
    try:
        with urlopen(DATA_URL, timeout=30) as response:
            content = response.read()
    except Exception as exc:
        raise RuntimeError(
            "Dataset gagal diunduh. Periksa koneksi internet lalu jalankan kembali."
        ) from exc

    DATA_PATH.write_bytes(content)
    print(f"Dataset tersimpan: {DATA_PATH}")
    return DATA_PATH


if __name__ == "__main__":
    download_dataset()
