"""Dataset download utilities for LAMBDA benchmarks."""
import os
import pandas as pd
import requests
from benchmarks.datasets.registry import DatasetInfo, ALL_DATASETS

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def download_uci_csv(uci_id: int, filename: str) -> str:
    """Download dataset from UCI ML Repository. Returns local CSV path."""
    save_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(save_path):
        print(f"  [skip] {filename} already exists")
        return save_path

    try:
        from ucimlrepo import fetch_ucirepo
        data = fetch_ucirepo(id=uci_id)
        df = data.data.original
        if df is not None and not df.empty:
            df.to_csv(save_path, index=False)
            print(f"  [OK] Downloaded {filename} ({len(df)} rows) from UCI #{uci_id}")
            return save_path
    except Exception as e:
        print(f"  [ucimlrepo failed] {e}, trying direct CSV...")

    # Fallback: direct URL
    urls = [
        f"https://archive.ics.uci.edu/static/markup/{uci_id}",
    ]
    print(f"  [WARN] Could not download {filename} automatically.")
    print(f"  Please download manually and save to: {save_path}")
    return save_path


def download_sklearn_dataset(name: str, filename: str) -> str:
    """Load dataset from sklearn.datasets. Returns local CSV path."""
    save_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(save_path):
        print(f"  [skip] {filename} already exists")
        return save_path

    from sklearn.datasets import fetch_openml
    try:
        if "breast" in name.lower():
            data = fetch_openml(name="wdbc", as_frame=True)
            df = data.frame
            df.to_csv(save_path, index=False)
        elif "wine" in name.lower():
            from sklearn.datasets import load_wine
            data = load_wine(as_frame=True)
            df = data.frame
            df['class'] = data.target
            df.to_csv(save_path, index=False)
        elif "iris" in name.lower():
            from sklearn.datasets import load_iris
            data = load_iris(as_frame=True)
            df = data.frame
            df.to_csv(save_path, index=False)
        else:
            data = fetch_openml(name=name, as_frame=True)
            df = data.frame
            df.to_csv(save_path, index=False)
        print(f"  [OK] Downloaded {filename} from sklearn ({len(df)} rows)")
    except Exception as e:
        print(f"  [ERROR] sklearn download failed for {name}: {e}")
    return save_path


def download_sms_spam(filename: str) -> str:
    """Download SMS Spam Collection dataset."""
    save_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(save_path):
        print(f"  [skip] {filename} already exists")
        return save_path

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    zip_path = os.path.join(DATA_DIR, "sms_spam.zip")
    try:
        r = requests.get(url, timeout=60)
        with open(zip_path, "wb") as f:
            f.write(r.content)
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as z:
            with z.open("SMSSpamCollection") as src:
                df = pd.read_csv(src, sep="\t", header=None, names=["label", "text"])
                df.to_csv(save_path, index=False)
        os.remove(zip_path)
        print(f"  [OK] Downloaded {filename} ({len(df)} rows)")
    except Exception as e:
        print(f"  [ERROR] SMS Spam download failed: {e}")
        print(f"  Please download manually and save to: {save_path}")
    return save_path


def download_dataset(ds: DatasetInfo) -> str:
    """Download a single dataset. Returns local CSV path."""
    ensure_data_dir()

    if ds.source == "torch" or ds.source == "synthetic":
        return ""  # No file needed - loaded programmatically

    if ds.uci_id is not None:
        return download_uci_csv(ds.uci_id, ds.filename)

    if "spam" in ds.name.lower():
        return download_sms_spam(ds.filename)

    if ds.source == "sklearn":
        return download_sklearn_dataset(ds.name, ds.filename)

    # Generic fallback
    print(f"  [WARN] No automatic download for {ds.name}. Using manual path.")
    return os.path.join(DATA_DIR, ds.filename)


def download_all():
    """Download all benchmark datasets."""
    ensure_data_dir()
    print("=" * 60)
    print("  Downloading LAMBDA Benchmark Datasets")
    print("=" * 60)

    for ds in ALL_DATASETS:
        if ds.source in ("torch", "synthetic"):
            print(f"  [skip] {ds.name} — loaded programmatically")
            continue
        print(f"\n  Downloading: {ds.name}...")
        path = download_dataset(ds)
        if path and os.path.exists(path):
            try:
                df = pd.read_csv(path)
                print(f"    ✓ Shape: {df.shape}")
            except Exception:
                pass

    print("\n" + "=" * 60)
    print("  Dataset download complete!")
    print("=" * 60)


if __name__ == "__main__":
    download_all()
