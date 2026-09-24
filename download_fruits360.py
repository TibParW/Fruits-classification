"""
download_fruits360.py
สคริปต์ดาวน์โหลดชุดข้อมูล Fruits-360 (Kaggle Dataset) และผลไม้ไทยยอดนิยม
ขยายคลาสผลไม้ครอบคลุมถึง 21 ชนิดผลไม้ ทั้งผลไม้ไทยยอดนิยมและผลไม้สากล
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

# รายชื่อผลไม้ 20 คลาสจาก Fruits-360 (Kaggle)
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
    "mangosteen": "Mangostan",
    "rambutan": "Rambutan",
    "dragonfruit": "Pitahaya Red",
    "papaya": "Papaya",
    "coconut": "Cocos",
    "guava": "Guava",
    "lychee": "Lychee",
    "salak": "Salak",
    "kiwi": "Kiwi",
    "avocado": "Avocado",
    "pomegranate": "Pomegranate",
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
        existing_imgs = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        if len(existing_imgs) >= limit_per_class:
            print(f" [{class_key}] มีภาพครบ {len(existing_imgs)} รูปแล้ว ข้ามการดาวน์โหลด")
            continue

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


def download_durian(output_dir="dataset", limit=60):
    """ดาวน์โหลดภาพทุเรียน (Durian) เพิ่มเติมจาก Bing Image Search (White Background Isolated)"""
    class_dir = Path(output_dir) / "durian"
    existing_imgs = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
    if len(existing_imgs) >= limit:
        print(f" [durian] มีภาพครบ {len(existing_imgs)} รูปแล้ว ข้ามการดาวน์โหลด")
        return

    class_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(output_dir) / "_temp_durian"
    print("\n>>> ดาวน์โหลดภาพ ทุเรียน (Durian 👑) จาก Search พร้อมคัดแยกพื้นหลังสะอาด...")

    try:
        from bing_image_downloader import downloader
        query = "durian fruit isolated white background"
        downloader.download(
            query,
            limit=limit + 20,
            output_dir=str(temp_dir),
            adult_filter_off=True,
            force_replace=False,
            timeout=10,
            verbose=False
        )

        downloaded_folder = temp_dir / query
        valid_count = 0
        if downloaded_folder.exists():
            for file_path in downloaded_folder.glob("*"):
                if not file_path.is_file():
                    continue
                try:
                    with Image.open(file_path) as img:
                        img.verify()
                    with Image.open(file_path) as img:
                        # Convert to RGB & resize standard
                        rgb_img = img.convert("RGB")
                    dest_file = class_dir / f"durian_{valid_count + 1:03d}.jpg"
                    rgb_img.save(dest_file, "JPEG", quality=92)
                    valid_count += 1
                    if valid_count >= limit:
                        break
                except Exception:
                    pass

        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

        print(f" [durian] ดาวน์โหลดภาพทุเรียนสำเร็จ: {valid_count} รูป")
    except Exception as e:
        print(f" เกิดข้อผิดพลาดในการดาวน์โหลดทุเรียน: {e}")


def main():
    download_fruits_360(limit_per_class=60)
    download_durian(limit=60)

    output_path = Path("dataset")
    print("\n" + "=" * 65)
    print(" สรุปจำนวนรูปภาพใน Dataset ทั้งหมด:")
    all_classes = sorted([d.name for d in output_path.iterdir() if d.is_dir() and not d.name.startswith("_")])
    total_imgs = 0
    for cls in all_classes:
        count = len(list((output_path / cls).glob("*.jpg")) + list((output_path / cls).glob("*.png")))
        total_imgs += count
        print(f" - {cls:<15}: {count} รูป")
    print(f"\n รวมทั้งหมด: {len(all_classes)} ชนิดผลไม้ | รวม {total_imgs} รูปภาพ")
    print("=" * 65)


if __name__ == "__main__":
    main()
