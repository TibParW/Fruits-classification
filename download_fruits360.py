"""
download_fruits360.py
สคริปต์ดาวน์โหลดชุดข้อมูล Fruits-360 (Kaggle Dataset) โดยตรงจาก Official Repository
รวดเร็ว มีประสิทธิภาพ พร้อมคัดกรองความสมบูรณ์ของภาพอัตโนมัติ
"""

import os
import sys
import shutil
import urllib.request
import urllib.parse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# คัดเลือกผลไม้ยอดนิยมจาก Fruits-360 (Kaggle)
FRUITS_360_MAP = {
    "apple": "Apple Red 1",
    "banana": "Banana",
    "orange": "Orange",
    "lemon": "Lemon",
    "strawberry": "Strawberry",
    "watermelon": "Watermelon",
    "grape": "Grape Blue",
    "pineapple": "Pineapple",
    "mango": "Mango",
}

GITHUB_API_BASE = "https://api.github.com/repos/Horea94/Fruit-Images-Dataset/contents/Test"


def fetch_file(download_url, dest_path):
    try:
        req = urllib.request.Request(download_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
        with open(dest_path, "wb") as f:
            f.write(data)
        # ตรวจสอบความถูกต้องของภาพ
        with Image.open(dest_path) as img:
            img.verify()
        with Image.open(dest_path) as img:
            img.convert("RGB")
        return True
    except Exception:
        if dest_path.exists():
            try:
                dest_path.unlink()
            except OSError:
                pass
        return False


def download_fruits_360(output_dir="dataset", limit_per_class=60):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(" เริ่มต้นดาวน์โหลด Dataset: Fruits-360 (จาก Kaggle Official Repo)")
    print(f" เป้าหมาย: {len(FRUITS_360_MAP)} ชนิดผลไม้ | คลาสละ {limit_per_class} รูป")
    print(f" โฟลเดอร์ปลายทาง: {output_path.resolve()}")
    print("=" * 65)

    for class_key, repo_folder in FRUITS_360_MAP.items():
        class_dir = output_path / class_key
        # เคลียร์โฟลเดอร์เดิมเพื่อให้เป็น Fruits-360 ล้วนๆ
        if class_dir.exists():
            shutil.rmtree(class_dir)
        class_dir.mkdir(parents=True, exist_ok=True)

        encoded_folder = urllib.parse.quote(repo_folder)
        api_url = f"{GITHUB_API_BASE}/{encoded_folder}"
        print(f"\n>>> ดึงรายชื่อภาพ {class_key.upper()} (โฟลเดอร์ Kaggle: '{repo_folder}')...")

        try:
            req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                items = json.loads(resp.read().decode())
        except Exception as e:
            print(f" เกิดข้อผิดพลาดในการเชื่อมต่อ GitHub API สำหรับ {repo_folder}: {e}")
            continue

        # กรองเฉพาะไฟล์ภาพ
        image_items = [item for item in items if item.get("name", "").lower().endswith((".jpg", ".png", ".jpeg"))]
        image_items = image_items[:limit_per_class]

        print(f" กำลังดาวน์โหลด {len(image_items)} รูปภาพพร้อมกันด้วย Multi-threading...")
        success_count = 0
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_idx = {}
            for idx, item in enumerate(image_items, start=1):
                dest_file = class_dir / f"{class_key}_{idx:03d}.jpg"
                fut = executor.submit(fetch_file, item["download_url"], dest_file)
                future_to_idx[fut] = idx

            for fut in as_completed(future_to_idx):
                if fut.result():
                    success_count += 1

        print(f" [{class_key}] ดาวน์โหลดสำเร็จและตรวจสอบสมบูรณ์: {success_count} รูป")

    print("\n" + "=" * 65)
    print(" ดาวน์โหลดชุดข้อมูล Fruits-360 (Kaggle) เสร็จสิ้นสมบูรณ์!")
    for class_key in FRUITS_360_MAP.keys():
        count = len(list((output_path / class_key).glob("*.jpg")))
        print(f" - {class_key}: {count} รูป")
    print("=" * 65)


if __name__ == "__main__":
    download_fruits_360(limit_per_class=60)
