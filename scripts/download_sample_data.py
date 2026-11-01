"""
scripts/download_sample_data.py
===============================
Downloads a lightweight, representative sample of English (TESS) and
Tamil Speech Emotion datasets from public repositories for feasibility testing.

Datasets:
  - English: Toronto Emotional Speech Set (TESS) from Scholars Portal Dataverse
    (15 samples per emotion across 7 emotions = 105 files, ~8.5 MB)
  - Tamil: Tamil Speech Emotion Dataset from HuggingFace (Thanushs25)
    (15 samples per emotion across 5 emotions = 75 files, ~12 MB)

Total download size: ~20 MB (fast, minimal disk footprint, enables all splits).
"""

from __future__ import annotations

import os
import sys
import json
import time
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------

def download_file(url: str, dest_path: Path, max_retries: int = 3) -> bool:
    """Download a URL to dest_path with retries."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 1000:
        return True

    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req, timeout=20) as response, open(dest_path, "wb") as out_file:
                out_file.write(response.read())
            if dest_path.stat().st_size > 100:
                return True
        except Exception as exc:
            if attempt == max_retries:
                print(f"  [ERROR] Failed to download {url}: {exc}")
            time.sleep(1.0)

    return False


# ---------------------------------------------------------------------------
# English TESS Downloader (Scholars Portal Dataverse)
# ---------------------------------------------------------------------------

DATAVERSE_API = "https://dataverse.scholarsportal.info/api/datasets/:persistentId/?persistentId=doi:10.5683/SP2/E8H2MF"
DATAVERSE_FILE_URL = "https://dataverse.scholarsportal.info/api/access/datafile/{file_id}"

TESS_EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "ps", "sad"]


def download_english_tess(target_dir: Path, samples_per_emotion: int = 15) -> int:
    """Download representative TESS samples from Dataverse."""
    print("\n" + "=" * 65)
    print("  [1/2] Downloading English TESS Sample Dataset")
    print(f"  Target: {target_dir}")
    print(f"  Samples per emotion: {samples_per_emotion} across {len(TESS_EMOTIONS)} emotions")
    print("=" * 65)

    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fetch metadata (try local cache first for instant catalog)
    cache_path = Path(r"C:\Users\srira\.gemini\antigravity-ide\brain\b6fbc3c8-33e6-423d-ae8e-e3a4869b3dc3\.system_generated\steps\123\content.md")
    files = None
    if cache_path.exists():
        try:
            raw_text = cache_path.read_text(encoding="utf-8")
            json_start = raw_text.find('{"status":')
            if json_start != -1:
                catalog = json.loads(raw_text[json_start:])
                files = catalog["data"]["latestVersion"]["files"]
                print(f"  Loaded catalog from local cache: {len(files)} files.")
        except Exception:
            files = None

    if not files:
        print("  Fetching dataset file catalog from Dataverse API...")
        try:
            req = urllib.request.Request(
                DATAVERSE_API,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req, timeout=30) as res:
                catalog = json.loads(res.read().decode("utf-8"))
            files = catalog["data"]["latestVersion"]["files"]
        except Exception as exc:
            print(f"  [WARN] Dataverse API failed: {exc}. Trying fallback...")
            return 0

    print(f"  Catalog retrieved: {len(files)} total files available.")

    # 2. Select samples per emotion
    # Filenames: OAF_word_emotion.wav or YAF_word_emotion.wav
    selected: dict[str, list[dict]] = {emo: [] for emo in TESS_EMOTIONS}

    for item in files:
        label = item["label"]  # e.g. OAF_back_angry.wav
        parts = label.replace(".wav", "").split("_")
        if len(parts) >= 3:
            emo = parts[-1].lower()
            if emo in selected and len(selected[emo]) < samples_per_emotion:
                selected[emo].append(item)

    download_tasks = []
    for emo, items in selected.items():
        for item in items:
            label = item["label"]
            file_id = item["dataFile"]["id"]
            # Speaker is parts[0] (e.g. OAF or YAF)
            spk = label.split("_")[0]
            folder_name = f"{spk}_{emo}"
            dest = target_dir / folder_name / label
            url = DATAVERSE_FILE_URL.format(file_id=file_id)
            download_tasks.append((url, dest))

    print(f"  Downloading {len(download_tasks)} TESS audio files...")

    success_count = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(download_file, url, dest): dest for url, dest in download_tasks}
        for fut in as_completed(futures):
            if fut.result():
                success_count += 1
            if success_count % 20 == 0:
                print(f"    Downloaded {success_count}/{len(download_tasks)} files...", flush=True)

    print(f"  [OK] English TESS: {success_count} files ready in {target_dir}")
    return success_count


# ---------------------------------------------------------------------------
# Tamil Speech Emotion Downloader (Hugging Face)
# ---------------------------------------------------------------------------

HF_BASE_URL = "https://huggingface.co/datasets/Thanushs25/tamil-audio-emotion-classification/resolve/main/final_data/{emotion}/{filename}"
TAMIL_EMOTIONS = ["angry", "fear", "happy", "neutral", "sad"]


def download_tamil_data(target_dir: Path, samples_per_emotion: int = 15) -> int:
    """Download representative Tamil emotion samples from Hugging Face."""
    print("\n" + "=" * 65)
    print("  [2/2] Downloading Tamil Emotion Sample Dataset")
    print(f"  Target: {target_dir}")
    print(f"  Samples per emotion: {samples_per_emotion} across {len(TAMIL_EMOTIONS)} emotions")
    print("=" * 65)

    target_dir.mkdir(parents=True, exist_ok=True)

    download_tasks = []
    for emo in TAMIL_EMOTIONS:
        emo_dir = target_dir / emo
        for i in range(samples_per_emotion):
            fname = f"audio_{i}.wav"
            url = HF_BASE_URL.format(emotion=emo, filename=fname)
            dest = emo_dir / fname
            download_tasks.append((url, dest))

    print(f"  Downloading {len(download_tasks)} Tamil audio files...")

    success_count = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(download_file, url, dest): dest for url, dest in download_tasks}
        for fut in as_completed(futures):
            if fut.result():
                success_count += 1
            if success_count % 20 == 0:
                print(f"    Downloaded {success_count}/{len(download_tasks)} files...", flush=True)

    print(f"  [OK] Tamil Dataset: {success_count} files ready in {target_dir}")
    return success_count


# ---------------------------------------------------------------------------
# Verification & Summary
# ---------------------------------------------------------------------------

def verify_audio_files(data_dir: Path) -> tuple[int, int]:
    """Verify that downloaded files can be parsed as audio."""
    import soundfile as sf

    valid = 0
    invalid = 0
    for wav_file in data_dir.rglob("*.wav"):
        try:
            info = sf.info(str(wav_file))
            if info.frames > 0 and info.samplerate > 0:
                valid += 1
            else:
                invalid += 1
        except Exception:
            invalid += 1
    return valid, invalid


def main():
    start = time.time()
    tess_dir = PROJECT_ROOT / "data" / "english" / "TESS"
    tamil_dir = PROJECT_ROOT / "data" / "tamil"

    n_eng = download_english_tess(tess_dir, samples_per_emotion=15)
    n_tam = download_tamil_data(tamil_dir, samples_per_emotion=15)

    print("\n" + "=" * 65)
    print("  DOWNLOAD & INTEGRITY SUMMARY")
    print("=" * 65)

    v_eng, inv_eng = verify_audio_files(tess_dir)
    v_tam, inv_tam = verify_audio_files(tamil_dir)

    print(f"  English TESS  : {v_eng} valid audio files ({inv_eng} invalid)")
    print(f"  Tamil Dataset : {v_tam} valid audio files ({inv_tam} invalid)")
    print(f"  Total Time    : {time.time() - start:.1f}s")
    print("=" * 65 + "\n")

    if v_eng > 0 and v_tam > 0:
        print("[SUCCESS] Datasets are downloaded and verified ready for training!\n")
    else:
        print("[WARNING] Some files failed verification. Please check network.\n")


if __name__ == "__main__":
    main()
