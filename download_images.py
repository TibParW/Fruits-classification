"""
download_images.py
สคริปต์สำหรับดาวน์โหลดชุดข้อมูลภาพผลไม้จากอินเทอร์เน็ตอัตโนมัติ
จัดโฟลเดอร์ตามชื่อผลไม้: dataset/apple, dataset/banana, dataset/orange
พร้อมระบบตรวจสอบความถูกต้องของไฟล์ภาพ (ลบภาพที่เสียหรือเปิดไม่ได้อัตโนมัติ)
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from PIL import Image
from bing_image_downloader import downloader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# การจับคู่ระหว่างโฟลเดอร์ปลายทาง และคำค้นหาเพื่อให้ได้ภาพผลไม้ที่ชัดเจนที่สุด
DEFAULT_TARGETS = {
    "apple": "fresh red apple fruit",
    "banana": "ripe yellow banana fruit",
    "orange": "fresh orange fruit",
}


def clean_corrupted_images(folder_path: Path):
    """ตรวจสอบภาพในโฟลเดอร์ และลบภาพที่ไม่สมบูรณ์ออก"""
    valid_count = 0
    corrupt_count = 0
    
    for file_path in folder_path.glob("*"):
        if not file_path.is_file():
            continue
        try:
            with Image.open(file_path) as img:
                img.verify()  # ตรวจสอบความเสียหายของไฟล์
            # เปิดใหม่อีกครั้งเพื่อทดสอบการแปลงเป็น RGB
            with Image.open(file_path) as img:
                img.convert("RGB")
            valid_count += 1
        except Exception:
            corrupt_count += 1
            try:
                os.remove(file_path)
            except OSError:
                pass

    print(f"[{folder_path.name}] รูปภาพใช้งานได้: {valid_count} รูป (ลบรูปเสีย {corrupt_count} รูป)")
    return valid_count


def download_fruits(output_dir="dataset", limit=50):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    temp_dir = output_path / "_temp_bing"

    print("=" * 60)
    print(" กำลังเริ่มดาวน์โหลด Dataset ผลไม้")
    print(f" จำนวนเป้าหมายต่อคลาส: {limit} รูป")
    print(f" บันทึกไว้ที่โฟลเดอร์: {output_path.resolve()}")
    print("=" * 60)

    for class_name, search_query in DEFAULT_TARGETS.items():
        class_folder = output_path / class_name
        class_folder.mkdir(parents=True, exist_ok=True)

        print(f"\n>>> ดาวน์โหลด {class_name.upper()} (ค้นหา: '{search_query}')")
        
        # ดาวน์โหลดลงโฟลเดอร์ชั่วคราว
        downloader.download(
            search_query,
            limit=limit,
            output_dir=str(temp_dir),
            adult_filter_off=True,
            force_replace=False,
            timeout=30,
            verbose=False,
        )

        # ย้ายรูปภาพไปยัง dataset/<class_name>/
        downloaded_subfolder = temp_dir / search_query
        if downloaded_subfolder.exists():
            existing_count = len(list(class_folder.glob("*")))
            for i, img_file in enumerate(downloaded_subfolder.glob("*"), start=existing_count + 1):
                ext = img_file.suffix.lower()
                if ext not in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
                    ext = ".jpg"
                dest_file = class_folder / f"{class_name}_{i:03d}{ext}"
                shutil.move(str(img_file), str(dest_file))

        # ตรวจสอบและลบรูปที่เปิดไม่ได้
        clean_corrupted_images(class_folder)

    # ลบโฟลเดอร์ชั่วคราว
    if temp_dir.exists():
        shutil.rmtree(str(temp_dir), ignore_errors=True)

    print("\n" + "=" * 60)
    print(" ดาวน์โหลดและเตรียมข้อมูลเสร็จเรียบร้อยแล้ว!")
    for class_name in DEFAULT_TARGETS.keys():
        count = len(list((output_path / class_name).glob("*")))
        print(f" - {class_name}: {count} รูป")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ดาวน์โหลดภาพผลไม้สำหรับเทรนโมเดล")
    parser.add_argument("--limit", type=int, default=50, help="จำนวนรูปต่อคลาส (ค่าเริ่มต้น: 50)")
    parser.add_argument("--output", type=str, default="dataset", help="โฟลเดอร์เก็บข้อมูล (ค่าเริ่มต้น: dataset)")
    args = parser.parse_args()

    download_fruits(output_dir=args.output, limit=args.limit)
