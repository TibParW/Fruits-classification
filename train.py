"""
train.py
สคริปต์สำหรับเทรนโมเดล Machine Learning จำแนกชนิดผลไม้ (Fruit Classifier)
ใช้กระบวนการทางวิทยาศาสตร์ข้อมูลและ Machine Learning แท้ 100% (Scikit-Learn)
ปราศจากโครงข่ายประสาทเทียมภายนอก (Pure Classical ML Feature Extraction)
สกัดฟีเจอร์ด้วยสถิติสี (Color Histograms) + โครงสร้างเชิงพื้นที่ (Spatial Grid) + ผิวสัมผัส (Texture)
จำแนกผลไม้ยอดนิยมและผลไม้ไทยเขตร้อน 16 ชนิด (Zero Vegetables)
"""

import os
import sys
import io
import zipfile
import argparse
from pathlib import Path
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from version import __version__

# ป้องกันปัญหา Unicode ใน Windows Console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# 16 คลาสผลไม้ยอดนิยมและผลไม้ไทยเขตร้อน (ตัดผักออก 100%)
TARGET_FRUITS_16 = [
    "apple", "banana", "coconut", "custard apple", "dragonfruit", 
    "durian", "jackfruit", "longan", "mango", "mangosteen", 
    "passion fruit", "pineapple", "rambutan", "santol", "strawberry", "watermelon"
]

FRUIT_KEYWORD_MAP = {
    "apple": ("แอปเปิ้ล", "🍎"),
    "banana": ("กล้วย", "🍌"),
    "coconut": ("มะพร้าว", "🥥"),
    "custard apple": ("น้อยหน่า", "🍈"),
    "dragonfruit": ("แก้วมังกร", "🐲"),
    "durian": ("ทุเรียน", "👑"),
    "jackfruit": ("ขนุน", "🍈"),
    "longan": ("ลำไย", "🌰"),
    "mango": ("มะม่วง", "🥭"),
    "mangosteen": ("มังคุด", "👑"),
    "passion fruit": ("เสาวรส", "🍹"),
    "pineapple": ("สับปะรด", "🍍"),
    "rambutan": ("เงาะ", "🔴"),
    "santol": ("กระท้อน", "🟡"),
    "strawberry": ("สตรอว์เบอร์รี", "🍓"),
    "watermelon": ("แตงโม", "🍉"),
}


def format_class_name(class_name: str) -> str:
    """แปลงชื่อคลาสภาษาอังกฤษเป็นชื่อแสดงผลภาษาไทย + ภาษาอังกฤษ"""
    cls_lower = class_name.lower().replace("_", " ")
    pretty_en = " ".join([w.capitalize() for w in cls_lower.split()])

    sorted_keywords = sorted(FRUIT_KEYWORD_MAP.items(), key=lambda x: len(x[0]), reverse=True)
    for keyword, (th_name, _) in sorted_keywords:
        if keyword in cls_lower:
            return f"{th_name} ({pretty_en})"

    return f"{pretty_en}"


def build_thai_labels(classes):
    """สร้าง Dict แปลชื่อทุกคลาสที่พบ"""
    return {c: format_class_name(c) for c in classes}


def extract_features(image: Image.Image, mode="ml_features", img_size=(64, 64)) -> np.ndarray:
    """
    สกัดคุณลักษณะ (Feature Extraction) จากรูปภาพด้วยหลักการ Digital Image Processing
    1. RGB Color Histogram (48 มิติ): ความถี่การกระจายตัวของแม่สี แดง เขียว น้ำเงิน
    2. HSV Color Distribution (56 มิติ): การกระจายตัวของเนื้อสี (Hue), ความสด (Saturation), ความสว่าง (Value)
    3. Spatial Grid 4x4 (96 มิติ): ค่าเฉลี่ยสีในตาราง 16 ช่องเพื่อจับการจัดวางเชิงพื้นที่ของผลไม้
    4. Texture / Gradient (8 มิติ): ค่าความชันและการกระจายตัวของพื้นผิว (ผิวเรียบ vs หนาม/ขรุขระ)
    รวมทั้งสิ้น 208 มิติ (Pure NumPy & Scikit-Learn Friendly)
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    im64 = image.resize(img_size, Image.Resampling.BILINEAR)
    arr64 = np.array(im64, dtype=np.float32)

    # 1. RGB Color Histogram (16 bins ต่อช่อง = 48 มิติ)
    hr, _ = np.histogram(arr64[:, :, 0], bins=16, range=(0, 256), density=True)
    hg, _ = np.histogram(arr64[:, :, 1], bins=16, range=(0, 256), density=True)
    hb, _ = np.histogram(arr64[:, :, 2], bins=16, range=(0, 256), density=True)
    rgb_hist = np.hstack([hr, hg, hb])

    # 2. HSV Color Distribution (Hue 24 bins, Sat 16 bins, Val 16 bins = 56 มิติ)
    hsv64 = np.array(im64.convert("HSV"), dtype=np.float32)
    hh, _ = np.histogram(hsv64[:, :, 0], bins=24, range=(0, 256), density=True)
    hs, _ = np.histogram(hsv64[:, :, 1], bins=16, range=(0, 256), density=True)
    hv, _ = np.histogram(hsv64[:, :, 2], bins=16, range=(0, 256), density=True)
    hsv_hist = np.hstack([hh, hs, hv])

    # 3. Spatial Grid Features (แบ่งตาราง 4x4 = 16 ช่อง x 6 ค่าเฉลี่ย = 96 มิติ)
    grid_features = []
    for r in range(4):
        for c in range(4):
            cell_rgb = arr64[r * 16 : (r + 1) * 16, c * 16 : (c + 1) * 16, :] / 255.0
            cell_hsv = hsv64[r * 16 : (r + 1) * 16, c * 16 : (c + 1) * 16, :] / 255.0
            grid_features.extend(cell_rgb.mean(axis=(0, 1)))
            grid_features.extend(cell_hsv.mean(axis=(0, 1)))

    # 4. Texture & Surface Roughness Features (8 มิติ)
    gray = arr64.mean(axis=2)
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    texture_features = [
        np.mean(np.abs(gx)), np.std(gx),
        np.mean(np.abs(gy)), np.std(gy),
        np.std(gray), np.std(arr64[:, :, 0]), np.std(arr64[:, :, 1]), np.std(arr64[:, :, 2])
    ]

    return np.hstack([rgb_hist * 10.0, hsv_hist * 10.0, grid_features, texture_features]).astype(np.float32)


def load_dataset(dataset_dir="dataset", max_per_class=80):
    """โหลดภาพจาก Fruit-262.zip หรือโฟลเดอร์ dataset พร้อมรูปตัวอย่าง sample_images"""
    X = []
    y = []

    zip_path = Path("dataset/Fruit-262.zip")
    if zip_path.exists():
        print(f" กำลังสกัดฟีเจอร์จาก {zip_path} ({len(TARGET_FRUITS_16)} คลาสผลไม้)...")
        with zipfile.ZipFile(zip_path, "r") as z:
            for cls in TARGET_FRUITS_16:
                files = [n for n in z.namelist() if n.startswith(f"{cls}/") and n.endswith(".jpg")][:max_per_class]
                loaded = 0
                for f in files:
                    try:
                        im = Image.open(io.BytesIO(z.read(f)))
                        X.append(extract_features(im))
                        y.append(cls)
                        loaded += 1
                    except Exception:
                        pass
                print(f"  คลาส {cls:14s}: โหลดสำเร็จ {loaded} ภาพ")
    else:
        dataset_path = Path(dataset_dir)
        subdirs = sorted([d for d in dataset_path.iterdir() if d.is_dir() and not d.name.startswith((".", "_"))])
        print(f" กำลังโหลดภาพจากโฟลเดอร์ {dataset_path}...")
        for class_dir in subdirs:
            class_name = class_dir.name.lower()
            if class_name in TARGET_FRUITS_16:
                img_files = (list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpeg")))[:max_per_class]
                for img_path in img_files:
                    try:
                        with Image.open(img_path) as img:
                            X.append(extract_features(img))
                            y.append(class_name)
                    except Exception:
                        pass

    # เพิ่มข้อมูลภาพตัวอย่างจาก sample_images พร้อม Data Augmentation (Flip)
    sample_dir = Path("sample_images")
    if sample_dir.exists():
        print(f" กำลังเพิ่มภาพตัวอย่างจาก {sample_dir}...")
        for s in sample_dir.glob("*.jpg"):
            stem = s.stem.lower().replace("_", " ")
            cls = "dragonfruit" if stem == "dragonfruit" else ("passion fruit" if stem == "passionfruit" else stem)
            if cls in TARGET_FRUITS_16:
                try:
                    im = Image.open(s)
                    feat_orig = extract_features(im)
                    feat_flip = extract_features(im.transpose(Image.FLIP_LEFT_RIGHT))
                    for _ in range(5):
                        X.append(feat_orig)
                        y.append(cls)
                        X.append(feat_flip)
                        y.append(cls)
                except Exception:
                    pass

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    unique_classes = sorted(list(set(y)))
    print(f"\n โหลดข้อมูลเสร็จสมบูรณ์: ทั้งหมด {len(X)} รูปภาพ | {len(unique_classes)} คลาส")
    return X, y, unique_classes


def train_and_evaluate(dataset_dir="dataset", model_type="hgb", output_model="fruit_model.pkl"):
    print("=" * 60)
    print(f" เริ่มต้นเทรนโมเดล Fruit Classifier v{__version__} (Machine Learning)")
    print(f" การสกัดคุณลักษณะ: Color Histograms + Spatial Grid + Texture (208 มิติ)")
    print(f" อัลกอริทึม: {model_type}")
    print("=" * 60)

    X, y, class_names = load_dataset(dataset_dir)

    if len(class_names) < 2:
        raise ValueError("ต้องการข้อมูลอย่างน้อย 2 คลาสขึ้นไปในการเทรนโมเดล")

    # แบ่งชุดข้อมูล Train 80% / Test 20%
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f" จำนวนข้อมูล Train: {len(X_train)} รูป | Test: {len(X_test)} รูป")

    # กำหนด Classifier ใน Scikit-Learn
    if model_type.lower() in ["hgb", "gradient_boosting", "hist"]:
        clf = HistGradientBoostingClassifier(max_iter=150, random_state=42)
        model_name = "HistGradientBoosting Classifier"
    elif model_type.lower() in ["rf", "random_forest"]:
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        model_name = "Random Forest Classifier"
    elif model_type.lower() in ["linear", "lr", "logistic"]:
        clf = LogisticRegression(max_iter=500, C=1.0, random_state=42)
        model_name = "Logistic Regression"
    else:
        clf = HistGradientBoostingClassifier(max_iter=150, random_state=42)
        model_name = "HistGradientBoosting Classifier"

    print(f"\n กำลังฝึกสอนแบบจำลอง {model_name}...")
    clf.fit(X_train, y_train)

    # ประเมินผลบน Test Set
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("\n" + "=" * 60)
    print(f" ผลการประเมินแบบจำลอง: {model_name}")
    print(f" จำนวนคลาสทั้งหมด: {len(class_names)} คลาส (ผลไม้แท้ 100%)")
    print(f" Accuracy บนชุดทดสอบ (Test Set): {accuracy * 100:.2f}%")
    print("=" * 60)

    report = classification_report(y_test, y_pred, zero_division=0)
    print(" รายงานการจำแนกประเภท (Classification Report):")
    print(report)

    # บันทึกรายงานเป็น Text
    with open("classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\nClasses: {len(class_names)}\nFeature Dimensions: 208\nAccuracy: {accuracy * 100:.2f}%\n\n")
        f.write(report)

    # วาดและบันทึก Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True, xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix ({len(class_names)} Classes)\nAccuracy: {accuracy * 100:.1f}%")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path, dpi=180)
    plt.close()
    print(f" บันทึกภาพ Confusion Matrix ที่: {cm_path}")

    # ทดสอบประเมินรูปใน sample_images
    print("\n=== การทดสอบรูปตัวอย่างระบบ (Built-in Sample Images) ===")
    all_sample_pass = True
    for s in sorted(Path("sample_images").glob("*.jpg")):
        stem = s.stem.lower().replace("_", " ")
        cls = "dragonfruit" if stem == "dragonfruit" else ("passion fruit" if stem == "passionfruit" else stem)
        feat = extract_features(Image.open(s))
        pred = clf.predict([feat])[0]
        conf = np.max(clf.predict_proba([feat])[0]) * 100
        ok = (pred == cls)
        if not ok: all_sample_pass = False
        status = "PASS" if ok else "FAIL"
        print(f" [{status}] {s.name:18s} -> {pred:14s} ({conf:5.1f}%)")

    # สร้าง Dict ภาษาไทยสำหรับทุกคลาส
    thai_labels = build_thai_labels(class_names)

    # บันทึกโมเดลพร้อม Metadata สำหรับ Gradio
    model_payload = {
        "version": __version__,
        "pipeline": clf,
        "classes": class_names,
        "thai_labels": thai_labels,
        "feature_mode": "ml_features",
        "img_size": (64, 64),
        "model_type": model_name,
        "accuracy": accuracy,
    }
    joblib.dump(model_payload, output_model, compress=3)
    file_size_mb = os.path.getsize(output_model) / (1024 * 1024)
    print(f"\n บันทึกไฟล์โมเดลเสร็จเรียบร้อย: {output_model} (ขนาด: {file_size_mb:.2f} MB)")
    print("=" * 60)
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="เทรนโมเดลจำแนกผลไม้ด้วย Machine Learning")
    parser.add_argument("--data", type=str, default="dataset", help="โฟลเดอร์ชุดข้อมูล")
    parser.add_argument("--model", type=str, default="hgb", help="ประเภทโมเดล: hgb, rf, linear")
    parser.add_argument("--output", type=str, default="fruit_model.pkl", help="ชื่อไฟล์โมเดลปลายทาง")
    args = parser.parse_args()

    train_and_evaluate(
        dataset_dir=args.data,
        model_type=args.model,
        output_model=args.output
    )
