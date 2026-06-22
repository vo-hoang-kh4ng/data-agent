"""ScienceAgentBench data preparation.

SAB ships two pieces:
  - annotations (102 tasks) on HF: osunlp/ScienceAgentBench (verified split).
  - the full benchmark_verified.zip (datasets/ + gold_programs/ + eval_programs/
    + scoring_rubrics/), password-protected, hosted on Ohio State SharePoint:
    https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/.../benchmark_verified.zip
    password = "scienceagentbench".

`prepare()` materialises both into `data/sab/`:
    data/sab/annotations.jsonl   (from HF)
    data/sab/benchmark/...        (extracted from the zip)

If the zip is absent locally it tries the SharePoint URL; if that fails the user
can drop benchmark_verified.zip next to this module and re-run. Extraction of the
1.77GB ZipCrypto archive via Python is slow (~minutes) but done once (cached).
"""
import json
import os
import sys
import zipfile


def _repo_root() -> str:
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        if os.path.exists(os.path.join(d, "config.yaml")) or os.path.exists(os.path.join(d, ".env")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.getcwd()


REPO_ROOT = _repo_root()
DEFAULT_DATA_DIR = os.environ.get("TRIADIC_EVAL_DATA", os.path.join(REPO_ROOT, "data"))
ZIP_PASSWORD = b"scienceagentbench"
SHAREPOINT_URL = ("https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/"
                  "chen_8336_buckeyemail_osu_edu/IQB870QrmuqwS5Ck33cHpJfkAVt3LsMeariREIwP3AT7byA"
                  "?e=IpASb7&download=1")


def _download_zip(dest: str) -> None:
    import urllib.request
    import shutil
    import subprocess
    print(f"[sab] downloading benchmark_verified.zip from SharePoint to {dest} ...")
    # curl handles the SharePoint cookie/redirect chain more reliably than urllib.
    try:
        subprocess.run(
            ["curl", "-sL", SHAREPOINT_URL, "-c", "_cj.txt", "-b", "_cj.txt",
             "--max-time", "1200", "-o", dest], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        # Fallback to urllib (single redirect hop; may fail on cookie carry).
        req = urllib.request.Request(SHAREPOINT_URL, headers={"User-Agent": "curl/8"})
        with urllib.request.urlopen(req, timeout=1200) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)
    if os.path.getsize(dest) < 1_000_000:
        raise RuntimeError("download too small — SharePoint fetch failed; place benchmark_verified.zip manually")


def _extract(zip_path: str, out_root: str) -> None:
    """Extract benchmark/* from the password zip into out_root (so files land at
    out_root/datasets, out_root/gold_programs, ...).

    Resumable: skips files already extracted (so a restart after a stall continues).
    Logs progress every 50 files so a stall on a huge file is visible. Pure-Python
    ZipCrypto is slow on large files, so this can take many minutes.

    size_limit_mb: skip files larger than this (default 150). Such files are large
    binary scientific datasets (.npy/.npz/.geojson/.xtc) whose tasks need specialised
    deps (mne/mdtraj/geopandas) and cannot run in the local eval anyway; they are
    logged so the gap is visible. Set TRIADIC_SAB_SIZE_LIMIT_MB=0 to extract all."""
    import time
    size_limit = float(os.environ.get("TRIADIC_SAB_SIZE_LIMIT_MB", "150")) * 1024 * 1024
    print(f"[sab] extracting {zip_path} -> {out_root} (resumable; logs every 50 files; "
          f"size_limit={int(size_limit/1048576)}MB) ...")
    os.makedirs(out_root, exist_ok=True)
    n_done = n_skipped = n_warn = n_big = 0
    t0 = last = time.monotonic()
    with zipfile.ZipFile(zip_path) as z:
        infos = z.infolist()
        for idx, info in enumerate(infos):
            if info.is_dir():
                continue
            name = info.filename
            rel = name[len("benchmark/"):] if name.startswith("benchmark/") else name
            if not rel or rel.endswith(".DS_Store"):
                continue
            if size_limit > 0 and info.file_size > size_limit:
                n_big += 1
                if n_big <= 20:
                    print(f"  [skip-big] {rel} ({info.file_size/1048576:.0f}MB)")
                continue
            target = os.path.join(out_root, rel)
            # resume: skip files already written with the expected size
            if os.path.exists(target) and os.path.getsize(target) == info.file_size:
                n_skipped += 1
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            try:
                data = z.read(info.filename, pwd=ZIP_PASSWORD)
            except RuntimeError as e:
                n_warn += 1
                if n_warn <= 10:
                    print(f"  [warn] could not decrypt {name}: {e}")
                continue
            with open(target, "wb") as f:
                f.write(data)
            n_done += 1
            if n_done % 50 == 0:
                now = time.monotonic()
                rate = 50 / (now - last)
                last = now
                print(f"  [sab] {idx+1}/{len(infos)} files | wrote {n_done} | "
                      f"skipped {n_skipped} | big-skip {n_big} | {rate:.1f} files/s")
    print(f"[sab] extraction done. wrote={n_done} skipped={n_skipped} big-skipped={n_big} "
          f"warns={n_warn} in {time.monotonic()-t0:.0f}s")


def prepare(data_dir: str = "") -> str:
    data_dir = data_dir or os.path.join(DEFAULT_DATA_DIR, "sab")
    bench_root = os.path.join(data_dir, "benchmark")
    ann_path = os.path.join(data_dir, "annotations.jsonl")

    os.makedirs(data_dir, exist_ok=True)

    # 1) annotations from HF
    if not os.path.exists(ann_path):
        print("[sab] writing annotations from HF (osunlp/ScienceAgentBench verified) ...")
        from datasets import load_dataset
        ds = load_dataset("osunlp/ScienceAgentBench", split="verified")
        with open(ann_path, "w", encoding="utf-8") as f:
            for r in ds:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 2) benchmark data from the zip. Use eval_programs/gold_results/ (near the end of the
    # zip) as the completion sentinel so a PARTIAL extraction (datasets present, but
    # gold_programs/eval_programs not yet) correctly resumes.
    gold_results_dir = os.path.join(bench_root, "eval_programs", "gold_results")
    need_extract = (not os.path.isdir(bench_root)
                    or not os.path.isdir(os.path.join(bench_root, "datasets"))
                    or not os.path.isdir(gold_results_dir))
    if need_extract:
        zip_path = os.path.join(REPO_ROOT, "benchmark_verified.zip")
        if not os.path.exists(zip_path):
            zip_path = os.path.join(data_dir, "benchmark_verified.zip")
        if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 1_000_000:
            _download_zip(zip_path)
        _extract(zip_path, bench_root)
    else:
        print(f"[sab] benchmark already extracted at {bench_root}")

    return data_dir


if __name__ == "__main__":
    prepare()
