"""
prepare_266_dataset.py
สคริปต์สกัดชุดข้อมูล 266 คลาสจาก fruits-360_100x100.zip
พร้อมเชื่อมโยงกับทุเรียน (Durian) รวมเป็น 267 ชนิด
"""

import os
import sys
import shutil
import zipfile
import re
from pathlib import Path
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ZIP_FILE = Path("dataset/fruits-360_100x100.zip")
DATASET_DIR = Path("dataset")
SAMPLES_PER_CLASS = 25


def sanitize_folder_name(name: str) -> str:
    # แปลงชื่อโฟลเดอร์ให้ปลอดภัย (ตัวพิมพ์เล็ก แทนที่ช่องว่างด้วย _)
    clean = re.sub(r"[^\w\s-]", "", name).strip()
    return clean.replace(" ", "_").lower()


def extract_all_classes():
    if not ZIP_FILE.exists():
        print(f"❌ ไม่พบไฟล์ zip: {ZIP_FILE}")
        return

    print("=" * 65)
    print(" เริ่มต้นแตกไฟล์ชุดข้อมูล 266 คลาสจาก fruits-360_100x100.zip")
    print(f" โควตาต่อคลาส: {SAMPLES_PER_CLASS} รูป")
    print("=" * 65)

    # 1. สำรองทุเรียน (Durian) ไว้ชั่วคราว
    durian_backup = Path("temp_durian_backup")
    durian_src = DATASET_DIR / "durian"
    if durian_src.exists():
        if durian_backup.exists():
            shutil.rmtree(durian_backup)
        shutil.copytree(durian_src, durian_backup)
        print(" สำรองข้อมูลคลาส 'durian' เรียบร้อย")

    # 2. ล้างโฟลเดอร์เก่าใน dataset (ยกเว้นไฟล์ zip)
    for item in DATASET_DIR.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            shutil.rmtree(item, ignore_errors=True)

    # 3. กู้คืน durian
    if durian_backup.exists():
        durian_dst = DATASET_DIR / "durian"
        shutil.copytree(durian_backup, durian_dst)
        shutil.rmtree(durian_backup, ignore_errors=True)
        # ปรับให้ durian มี 25 รูปเท่ากับคลาสอื่น
        durian_imgs = list(durian_dst.glob("*.jpg"))
        for extra_img in durian_imgs[SAMPLES_PER_CLASS:]:
            extra_img.unlink()
        print(f" คลาส 'durian': {len(list(durian_dst.glob('*.jpg')))} รูป")

    # 4. สกัดภาพจาก zip file
    with zipfile.ZipFile(ZIP_FILE, "r") as zf:
        # จัดกลุ่มไฟล์ตามคลาสใน Training
        class_files = defaultdict(list)
        for name in zf.namelist():
            if not name.lower().endswith((".jpg", ".png", ".jpeg")):
                continue
            parts = Path(name).parts
            if len(parts) >= 4 and parts[0] == "fruits-360" and parts[1] == "Training":
                raw_cls = parts[2]
                if len(class_files[raw_cls]) < SAMPLES_PER_CLASS:
                    class_files[raw_cls].append(name)

        print(f"\n กำลังสกัด {len(class_files)} คลาสจาก zip file...")
        for raw_cls, files in class_files.items():
            folder_name = sanitize_folder_name(raw_cls)
            target_dir = DATASET_DIR / folder_name
            target_dir.mkdir(parents=True, exist_ok=True)

            for idx, zip_entry in enumerate(files, 1):
                dest_file = target_dir / f"{folder_name}_{idx:03d}.jpg"
                data = zf.read(zip_entry)
                with open(dest_file, "wb") as f:
                    f.write(data)

    all_dirs = [d for d in DATASET_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")]
    print("\n" + "=" * 65)
    print(f" สกัดเสร็จสิ้นสมบูรณ์!")
    print(f" จำนวนคลาสผลไม้ทั้งหมด: {len(all_dirs)} คลาส")
    print(f" จำนวนรูปภาพทั้งหมด: {sum(len(list(d.glob('*.jpg'))) for d in all_dirs)} รูป")
    print("=" * 65)


if __name__ == "__main__":
    extract_all_classes()
