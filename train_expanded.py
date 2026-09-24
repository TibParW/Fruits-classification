"""
train_expanded.py
สคริปต์เทรนโมเดลจำแนกผลไม้ 267 คลาสแบบขยายขนาดชุดข้อมูล (Expanded Training)
- เพิ่มขนาดข้อมูลเป็น 20,000+ รูปภาพ (จากเดิม 6,675 รูปภาพ)
- เสริมระบบ Background Invariance Augmentation (ลดผลกระทบจากโต๊ะไม้/ฉากหลังรก)
- เสริมภาพผลไม้จริงและผลไม้ผ่าครึ่ง (เช่น แก้วมังกรเนื้อขาว, มะเฟือง, ทุเรียน) จาก Fruit-262
- บันทึกผลลัพธ์เป็น fruit_model.pkl
"""

import sys
import os
import io
import time
import random
import zipfile
from pathlib import Path
from collections import defaultdict

import joblib
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from train import extract_features, build_thai_labels

# กำหนดสุ่มเพื่อความแม่นยำซ้ำเดิมได้
random.seed(42)
np.random.seed(42)

F360_ZIP = Path("dataset/fruits-360_100x100.zip")
F262_ZIP = Path("dataset/Fruit-262.zip")
OUTPUT_MODEL = Path("fruit_model.pkl")

# ฟังก์ชันเปลี่ยนพื้นหลังเพื่อสร้าง Background Invariance
def apply_bg_augmentation(pil_img: Image.Image) -> Image.Image:
    """สุ่มเปลี่ยนฉากหลังสีขาวสตูดิโอให้เป็นสีธรรมชาติ (โต๊ะไม้/เทา/มืด) เพื่อสอนให้โมเดลไม่ยึดติดฉากหลัง"""
    arr = np.array(pil_img.convert("RGB"))
    # พิกเซลขาวฉากหลัง
    mask_bg = (arr[:, :, 0] > 240) & (arr[:, :, 1] > 240) & (arr[:, :, 2] > 240)
    if not np.any(mask_bg):
        return pil_img

    choice = random.random()
    arr_aug = arr.copy()
    if choice < 0.35:
        # โทนสีไม้ / โต๊ะไม้ (น้ำตาลส้ม)
        arr_aug[mask_bg] = [
            random.randint(140, 180),
            random.randint(100, 140),
            random.randint(60, 100),
        ]
    elif choice < 0.65:
        # โทนสีเทากลาง / ขาวนวล / เบจ
        arr_aug[mask_bg] = [
            random.randint(160, 210),
            random.randint(160, 210),
            random.randint(150, 200),
        ]
    elif choice < 0.85:
        # โทนสีเทาเข้ม / มืด
        arr_aug[mask_bg] = [random.randint(40, 80)] * 3
    else:
        # คงเดิม
        return pil_img

    return Image.fromarray(arr_aug)


def load_expanded_dataset(images_per_class=75):
    print(f"\n📦 เริ่มต้นอ่านและสกัดฟีเจอร์สำหรับ 267 คลาส (เป้าหมาย ~{images_per_class} รูป/คลาส)...")
    t0 = time.time()

    # รวบรวมรายชื่อคลาสจาก dataset directory ปัจจุบัน (เพื่อให้ตรง 267 คลาสเดิม 100%)
    dataset_path = Path("dataset")
    dirs = sorted([d.name for d in dataset_path.iterdir() if d.is_dir() and not d.name.startswith((".", "_"))])
    print(f"✅ ยืนยันจำนวนโฟลเดอร์คลาสเป้าหมาย: {len(dirs)} คลาส")

    # รวบรวมไฟล์ใน fruits-360 zip
    with zipfile.ZipFile(F360_ZIP, "r") as z360:
        zip_map = {}
        for n in z360.namelist():
            if n.startswith("fruits-360/Training/") and n.endswith(".jpg"):
                parts = n.split("/")
                if len(parts) >= 3 and parts[2]:
                    cls_key = parts[2].lower().replace(" ", "_")
                    if cls_key not in zip_map:
                        zip_map[cls_key] = []
                    zip_map[cls_key].append(n)

    X = []
    y = []
    total_imgs = 0

    # เปิด zip ทั้ง 2 เพื่อดึงข้อมูล
    with zipfile.ZipFile(F360_ZIP, "r") as z360, zipfile.ZipFile(F262_ZIP, "r") as z262:
        # ดึงไฟล์พิเศษจาก Fruit-262: dragonfruit, durian, carambola
        f262_dragon = [n for n in z262.namelist() if n.startswith("dragonfruit/") and n.endswith(".jpg")]
        f262_durian = [n for n in z262.namelist() if n.startswith("durian/") and n.endswith(".jpg")]
        f262_carambola = [n for n in z262.namelist() if n.startswith("carambola/") and n.endswith(".jpg")]

        for idx, cls_name in enumerate(dirs):
            cls_imgs = 0
            if cls_name == "durian":
                # ทุเรียน ใช้ภาพที่มีใน dataset/durian (ฉากหลังขาวมาตรฐาน) และทำ Data Augmentation ให้ครบจำนวน
                local_dir = dataset_path / "durian"
                dur_files = list(local_dir.glob("*.jpg")) + list(local_dir.glob("*.png"))
                for fn in dur_files:
                    try:
                        im = Image.open(fn).convert("RGB")
                        # ภาพต้นฉบับ
                        X.append(extract_features(im))
                        y.append("durian")
                        cls_imgs += 1
                        # ภาพพลิกแนวนอน (Horizontal Flip)
                        im_flip = im.transpose(Image.FLIP_LEFT_RIGHT)
                        X.append(extract_features(im_flip))
                        y.append("durian")
                        cls_imgs += 1
                        # ภาพเปลี่ยนฉากหลัง (Augmented Background)
                        im_aug = apply_bg_augmentation(im)
                        X.append(extract_features(im_aug))
                        y.append("durian")
                        cls_imgs += 1
                    except Exception:
                        pass
            elif cls_name in zip_map:
                available_files = zip_map[cls_name]
                sample_count = min(len(available_files), images_per_class)
                selected_files = available_files[:sample_count]

                # เสริมภาพแก้วมังกรจริงจาก Fruit-262 (ทั้งลูกและผ่าครึ่ง)
                if ("pitahaya" in cls_name or "dragon" in cls_name) and f262_dragon:
                    extra_dragon = f262_dragon[:40]
                    for fn in extra_dragon:
                        try:
                            im = Image.open(io.BytesIO(z262.read(fn))).convert("RGB").resize((100, 100))
                            X.append(extract_features(im))
                            y.append(cls_name)
                            cls_imgs += 1
                            # เพิ่มภาพสลับซ้ายขวา
                            im_flip = im.transpose(Image.FLIP_LEFT_RIGHT)
                            X.append(extract_features(im_flip))
                            y.append(cls_name)
                            cls_imgs += 1
                        except Exception:
                            pass

                if "carambola_1" in cls_name and f262_carambola:
                    # ใส่ภาพมะเฟืองจริงจาก Fruit-262 20 รูป
                    extra_caram = f262_carambola[:20]
                    for fn in extra_caram:
                        try:
                            im = Image.open(io.BytesIO(z262.read(fn))).convert("RGB").resize((100, 100))
                            X.append(extract_features(im))
                            y.append(cls_name)
                            cls_imgs += 1
                        except Exception:
                            pass

                for fn in selected_files:
                    try:
                        im = Image.open(io.BytesIO(z360.read(fn))).convert("RGB")
                        # ภาพต้นฉบับ
                        feat = extract_features(im)
                        X.append(feat)
                        y.append(cls_name)
                        cls_imgs += 1

                        # ภาพเพิ่มฉากหลัง (Augmented) ทุกๆ 3 ภาพ
                        if random.random() < 0.35:
                            im_aug = apply_bg_augmentation(im)
                            feat_aug = extract_features(im_aug)
                            X.append(feat_aug)
                            y.append(cls_name)
                            cls_imgs += 1
                    except Exception:
                        pass
            else:
                # กรณีโฟลเดอร์ที่มีอยู่ใน dataset/ อยู่แล้ว
                local_dir = dataset_path / cls_name
                local_files = list(local_dir.glob("*.jpg")) + list(local_dir.glob("*.png"))
                for fn in local_files:
                    try:
                        im = Image.open(fn).convert("RGB")
                        feat = extract_features(im)
                        X.append(feat)
                        y.append(cls_name)
                        cls_imgs += 1
                    except Exception:
                        pass

            total_imgs += cls_imgs
            if (idx + 1) % 50 == 0 or (idx + 1) == len(dirs):
                print(f"  ⚡ ประมวลผลเสร็จแล้ว {idx + 1}/{len(dirs)} คลาส (สะสม {total_imgs} รูปภาพ)...")

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    dt = time.time() - t0
    print(f"🎉 สกัดฟีเจอร์สำเร็จทั้งหมด {len(X)} ตัวอย่าง ในเวลา {dt:.1f} วินาที | ขนาด Matrix: {X.shape}")
    return X, y, dirs


def main():
    print("=" * 65)
    print("🚀 เริ่มต้นกระบวนการเทรนโมเดล 267 คลาส (เวอร์ชันชุดข้อมูลขยายใหญ่)")
    print("=" * 65)

    # 1. โหลดและสกัดฟีเจอร์ (~75 รูปต่อคลาส)
    X, y, class_names = load_expanded_dataset(images_per_class=75)

    # 2. แบ่งชุดข้อมูล Train 80% / Test 20%
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n📊 การแบ่งชุดข้อมูล:")
    print(f"   - ชุดฝึกสอน (Train Set 80%): {len(X_train)} รูปภาพ")
    print(f"   - ชุดทดสอบ (Test Set 20%):   {len(X_test)} รูปภาพ")

    # 3. สร้าง Pipeline (StandardScaler + Logistic Regression)
    print(f"\n⚙️ กำลังฝึกสอนโมเดล Logistic Regression บน {len(X_train)} ตัวอย่าง...")
    t_train = time.time()
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=150, C=1.0, random_state=42, solver="lbfgs"))
    ])
    pipeline.fit(X_train, y_train)
    print(f"✅ ฝึกสอนสำเร็จในเวลา {time.time() - t_train:.1f} วินาที!")

    # 4. ประเมินผลบน Test Set
    print("\n🔍 กำลังประเมินผลบน Test Set...")
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("=" * 65)
    print(f"🏆 ผลลัพธ์ความแม่นยำบน Test Set (จำนวน {len(y_test)} ภาพ): {accuracy * 100:.2f}%")
    print("=" * 65)

    # 5. สร้าง Thai Labels
    thai_labels = build_thai_labels(class_names)

    # 6. บันทึกโมเดล
    model_data = {
        "version": "1.3.0",
        "pipeline": pipeline,
        "classes": class_names,
        "thai_labels": thai_labels,
        "feature_mode": "combined",
        "img_size": (32, 32),
        "model_type": "Logistic Regression (Linear Classifier)",
        "accuracy": float(accuracy),
        "total_samples": len(X),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }

    joblib.dump(model_data, OUTPUT_MODEL, compress=3)
    file_size_mb = OUTPUT_MODEL.stat().st_size / (1024 * 1024)
    print(f"💾 บันทึกไฟล์โมเดลสำเร็จ: {OUTPUT_MODEL.resolve()} ({file_size_mb:.2f} MB)")

    # 7. บันทึก Classification Report
    report = classification_report(y_test, y_pred, zero_division=0)
    with open("classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Model: Logistic Regression (267 Classes - Expanded Dataset)\n")
        f.write(f"Total Samples: {len(X)} (Train: {len(X_train)}, Test: {len(X_test)})\n")
        f.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")
        f.write(report)
    print(f"📄 บันทึกรายงาน classification_report.txt เรียบร้อย")

    # 8. ทดสอบความแม่นยำบน User Samples และ 12 Built-in Samples ทันที
    print("\n🧪 กำลังทดสอบ Verification บนภาพจริง...")
    test_files = [
        ("temp_dragonfruit.png", "pitahaya_red_1", "แก้วมังกรผ่าครึ่ง"),
        ("temp_starfruit.png", "carambola_1", "มะเฟืองบนโต๊ะไม้"),
    ]
    for fn, expected, desc in test_files:
        if Path(fn).exists():
            im = Image.open(fn)
            feat = extract_features(im)
            probs = pipeline.predict_proba([feat])[0]
            top_cls = class_names[np.argmax(probs)]
            top_conf = np.max(probs) * 100
            print(f"   [{desc}] -> {top_cls} ({top_conf:.1f}%) | คาดหวัง: {expected}")


if __name__ == "__main__":
    main()
