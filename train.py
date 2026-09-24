"""
train.py
สคริปต์สำหรับเทรนโมเดล Machine Learning จำแนกชนิดผลไม้ (Fruit Classifier)
รองรับทั้งโมเดล SVM และ Random Forest
พร้อมระบบสกัดฟีเจอร์:
  1. Flatten Array (พิกเซลภาพแปลงเป็นเวกเตอร์)
  2. Color Histogram (การกระจายตัวของค่าสี RGB)
  3. Combined (ดึงทั้งสีและโครงสร้างภาพรวมกัน เพื่อความแม่นยำสูงสุด)
"""

import os
import sys
import argparse
from pathlib import Path
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from version import __version__

# ป้องกันปัญหา UnicodeEncodeError ใน Windows Console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# การแปลชื่อคลาสเป็นภาษาไทยและ emoji เพื่อนำไปแสดงผล
THAI_LABELS = {
    "apple": "แอปเปิ้ล (Apple 🍎)",
    "banana": "กล้วย (Banana 🍌)",
    "orange": "ส้ม (Orange 🍊)",
    "lemon": "เลมอน/มะนาว (Lemon 🍋)",
    "strawberry": "สตรอว์เบอร์รี (Strawberry 🍓)",
    "watermelon": "แตงโม (Watermelon 🍉)",
    "grape": "องุ่น (Grape 🍇)",
    "mango": "มะม่วง (Mango 🥭)",
}


def extract_features(image: Image.Image, mode="combined", img_size=(64, 64)):
    """
    สกัด Feature จากรูปภาพ
    - mode='flatten': ย่อภาพเป็น 64x64 แล้วแปลงเป็น 1D Array (Normalized [0, 1])
    - mode='histogram': คำนวณ Color Histogram ทั้ง RGB และ HSV
    - mode='combined': รวมทั้งเวกเตอร์พิกเซล, RGB Histogram และ HSV Color Distribution
    """
    # ตรวจสอบและแปลงเป็น RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # ย่อภาพตามขนาดที่กำหนด
    resized_img = image.resize(img_size)
    img_array = np.array(resized_img, dtype=np.float32)

    # 1. Flatten Features
    flatten_feat = (img_array / 255.0).flatten()

    # 2. RGB Color Histogram (ช่องละ 16 bins)
    hist_r, _ = np.histogram(img_array[:, :, 0], bins=16, range=(0, 256), density=True)
    hist_g, _ = np.histogram(img_array[:, :, 1], bins=16, range=(0, 256), density=True)
    hist_b, _ = np.histogram(img_array[:, :, 2], bins=16, range=(0, 256), density=True)
    rgb_feat = np.hstack([hist_r, hist_g, hist_b])

    # 3. HSV Color Histogram (Hue 24 bins, Saturation 16 bins, Value 16 bins)
    hsv_img = image.convert("HSV")
    hsv_array = np.array(hsv_img, dtype=np.float32)
    hist_h, _ = np.histogram(hsv_array[:, :, 0], bins=24, range=(0, 256), density=True)
    hist_s, _ = np.histogram(hsv_array[:, :, 1], bins=16, range=(0, 256), density=True)
    hist_v, _ = np.histogram(hsv_array[:, :, 2], bins=16, range=(0, 256), density=True)
    hsv_feat = np.hstack([hist_h, hist_s, hist_v])

    if mode == "flatten":
        return flatten_feat
    elif mode == "histogram":
        return np.hstack([rgb_feat, hsv_feat])
    elif mode == "combined":
        return np.hstack([flatten_feat, rgb_feat, hsv_feat])
    else:
        raise ValueError(f"Unknown feature mode: {mode}")


def load_dataset(dataset_dir="dataset", mode="combined", img_size=(64, 64)):
    """โหลดภาพจากทุกโฟลเดอร์ใน dataset/ และสกัดฟีเจอร์"""
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์ dataset: {dataset_path.resolve()}")

    X = []
    y = []
    class_names = []

    # ค้นหาโฟลเดอร์ย่อย (แต่ละโฟลเดอร์คือ 1 คลาสผลไม้)
    subdirs = sorted([d for d in dataset_path.iterdir() if d.is_dir() and not d.name.startswith("_")])
    if not subdirs:
        raise ValueError(f"ไม่พบคลาสผลไม้ในโฟลเดอร์: {dataset_path.resolve()}")

    print("\n กำลังอ่านข้อมูลรูปภาพและสกัดฟีเจอร์...")
    for class_idx, class_dir in enumerate(subdirs):
        class_name = class_dir.name.lower()
        class_names.append(class_name)
        img_files = list(class_dir.glob("*.*"))
        valid_count = 0

        for img_path in img_files:
            try:
                with Image.open(img_path) as img:
                    feat = extract_features(img, mode=mode, img_size=img_size)
                    X.append(feat)
                    y.append(class_name)
                    valid_count += 1
            except Exception as e:
                print(f"[คำเตือน] ข้ามไฟล์ที่เปิดไม่ได้ {img_path.name}: {e}")

        print(f" - คลาส '{class_name}': อ่านสำเร็จ {valid_count} รูป")

    X = np.array(X)
    y = np.array(y)
    print(f"\n ขนาดของชุดข้อมูลทั้งหมด: X={X.shape}, y={y.shape}")
    return X, y, class_names


def train_and_evaluate(dataset_dir="dataset", feature_mode="combined", model_type="svm", output_model="fruit_model.pkl"):
    img_size = (64, 64)
    X, y, class_names = load_dataset(dataset_dir, mode=feature_mode, img_size=img_size)

    if len(class_names) < 2:
        raise ValueError("ต้องการข้อมูลอย่างน้อย 2 คลาสขึ้นไปในการเทรนโมเดล")

    # แบ่งชุดข้อมูล Train 80% / Test 20%
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f" จำนวนข้อมูล Train: {len(X_train)} รูป | Test: {len(X_test)} รูป")

    # สร้าง Pipeline: มาตรฐานข้อมูล (StandardScaler) + ตัวแบบ Machine Learning
    if model_type.lower() == "svm":
        try:
            from sklearn.calibration import CalibratedClassifierCV
            base_svc = SVC(kernel="rbf", C=1.0, random_state=42)
            clf = CalibratedClassifierCV(base_svc, ensemble=False)
        except Exception:
            clf = SVC(kernel="rbf", C=1.0, probability=True, random_state=42)
        model_name = "Support Vector Machine (SVM RBF)"
    elif model_type.lower() == "rf":
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        model_name = "Random Forest Classifier"
    else:
        raise ValueError(f"ไม่รู้จักประเภทโมเดล: {model_type} (เลือกได้: svm หรือ rf)")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", clf)
    ])

    print(f"\n กำลังเทรนโมเดล {model_name}...")
    pipeline.fit(X_train, y_train)

    # ประเมินผลบน Test Set
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("\n" + "=" * 60)
    print(f" ผลการทดสอบโมเดล: {model_name}")
    print(f" Accuracy: {accuracy * 100:.2f}%")
    print("=" * 60)
    
    report = classification_report(y_test, y_pred, target_names=class_names)
    print(" รายงานการจำแนกประเภท (Classification Report):")
    print(report)

    # บันทึกรายงานเป็นไฟล์ Text
    with open("classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\nFeature Mode: {feature_mode}\nAccuracy: {accuracy * 100:.2f}%\n\n")
        f.write(report)

    # วาดและบันทึก Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[c.capitalize() for c in class_names],
                yticklabels=[c.capitalize() for c in class_names])
    plt.title(f"Confusion Matrix ({model_name})\nAccuracy: {accuracy * 100:.1f}%")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f" บันทึกภาพ Confusion Matrix ที่: {cm_path}")

    # บันทึกโมเดลพร้อม Metadata สำหรับ Gradio
    model_payload = {
        "version": __version__,
        "pipeline": pipeline,
        "classes": class_names,
        "thai_labels": THAI_LABELS,
        "feature_mode": feature_mode,
        "img_size": img_size,
        "model_type": model_type,
        "accuracy": accuracy,
    }
    joblib.dump(model_payload, output_model)
    print(f" บันทึกไฟล์โมเดลเสร็จเรียบร้อย: {output_model}")
    print("=" * 60)
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="เทรนโมเดลจำแนกผลไม้")
    parser.add_argument("--data", type=str, default="dataset", help="โฟลเดอร์ชุดข้อมูล (ค่าเริ่มต้น: dataset)")
    parser.add_argument("--feature", type=str, default="combined", choices=["combined", "flatten", "histogram"],
                        help="ประเภทฟีเจอร์: combined (แนะนำ), flatten, histogram")
    parser.add_argument("--model", type=str, default="svm", choices=["svm", "rf"],
                        help="โมเดล: svm (ค่าเริ่มต้น) หรือ rf (Random Forest)")
    parser.add_argument("--output", type=str, default="fruit_model.pkl", help="ชื่อไฟล์โมเดลปลายทาง")
    args = parser.parse_args()

    train_and_evaluate(
        dataset_dir=args.data,
        feature_mode=args.feature,
        model_type=args.model,
        output_model=args.output
    )
