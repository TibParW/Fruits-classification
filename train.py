"""
train.py
สคริปต์สำหรับเทรนโมเดล Machine Learning จำแนกชนิดผลไม้ (Fruit Classifier)
รองรับการสเกลระดับ 267 ชนิดผลไม้ (Fruits-360 + ผลไม้ไทยยอดนิยม)
พร้อมระบบสกัดฟีเจอร์:
  1. Flatten Array (พิกเซลภาพแปลงเป็นเวกเตอร์ 32x32)
  2. Color Histogram (การกระจายตัวของค่าสี RGB)
  3. HSV Color Distribution (Hue, Saturation, Value)
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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from version import __version__

# ป้องกันปัญหา Unicode ใน Windows Console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# พจนานุกรมคำแปลภาษาไทยและ Emoji สำหรับผลไม้และพืชผล 267 คลาส
FRUIT_KEYWORD_MAP = {
    "apple": ("แอปเปิ้ล", "🍎"),
    "almond": ("อัลมอนด์", "🥜"),
    "apricot": ("แอปริคอต", "🍑"),
    "avocado": ("อะโวคาโด", "🥑"),
    "banana": ("กล้วย", "🍌"),
    "bean": ("ถั่ว", "🫘"),
    "beetroot": ("บีทรูท", "🟣"),
    "blackberry": ("แบล็กเบอร์รี", "🫐"),
    "blueberry": ("บลูเบอร์รี", "🫐"),
    "cabbage": ("กะหล่ำปลี", "🥬"),
    "cactus": ("ผลกระบองเพชร", "🌵"),
    "caju": ("มะม่วงหิมพานต์", "🥜"),
    "cantaloupe": ("แคนตาลูป", "🍈"),
    "carambola": ("มะเฟือง", "⭐"),
    "carrot": ("แครอท", "🥕"),
    "cauliflower": ("กะหล่ำดอก", "🥦"),
    "celery": ("ขึ้นฉ่าย", "🥬"),
    "cherimoya": ("น้อยโหน่ง", "🍈"),
    "cherry": ("เชอร์รี่", "🍒"),
    "chestnut": ("เกาลัด", "🌰"),
    "clementine": ("ส้มเคลเมนไทน์", "🍊"),
    "cocos": ("มะพร้าว", "🥥"),
    "coconut": ("มะพร้าว", "🥥"),
    "corn": ("ข้าวโพด", "🌽"),
    "cucumber": ("แตงกวา", "🥒"),
    "dates": ("อินทผลัม", "🌴"),
    "durian": ("ทุเรียน", "👑"),
    "eggplant": ("มะเขือยาว", "🍆"),
    "fig": ("มะเดื่อฝรั่ง", "🫐"),
    "garlic": ("กระเทียม", "🧄"),
    "ginger": ("ขิง", "🫚"),
    "gooseberry": ("กูสเบอร์รี", "🍈"),
    "granadilla": ("เสาวรสหวาน", "🍹"),
    "grape": ("องุ่น", "🍇"),
    "grapefruit": ("เกรปฟรุต", "🍊"),
    "guava": ("ฝรั่ง", "🍐"),
    "hazelnut": ("เฮเซลนัท", "🌰"),
    "huckleberry": ("ฮักเคิลเบอร์รี", "🫐"),
    "kaki": ("ลูกพลับ", "🍅"),
    "kiwi": ("กีวี", "🥝"),
    "kohlrabi": ("กะหล่ำปม", "🥬"),
    "kumquat": ("ส้มคัมควอท", "🍊"),
    "lemon": ("เลมอน", "🍋"),
    "lime": ("มะนาว", "🍋"),
    "lychee": ("ลิ้นจี่", "🍒"),
    "mandarine": ("ส้มแมนดาริน", "🍊"),
    "mango": ("มะม่วง", "🥭"),
    "mangostan": ("มังคุด", "👑"),
    "mangosteen": ("มังคุด", "👑"),
    "maracuja": ("เสาวรส", "🍹"),
    "melon": ("เมลอน", "🍈"),
    "mulberry": ("มัลเบอร์รี", "🫐"),
    "mushroom": ("เห็ด", "🍄"),
    "nectarine": ("เนคทารีน", "🍑"),
    "nut": ("ถั่ว/นัท", "🥜"),
    "onion": ("หอมหัวใหญ่", "🧅"),
    "orange": ("ส้ม", "🍊"),
    "papaya": ("มะละกอ", "🍈"),
    "passion": ("เสาวรส", "🍹"),
    "peach": ("พีช", "🍑"),
    "peanut": ("ถั่วลิสง", "🥜"),
    "pear": ("ลูกแพร์", "🍐"),
    "pepino": ("เปปิโนเมลอน", "🍈"),
    "pepper": ("พริกหวาน", "🫑"),
    "physalis": ("เคพกูสเบอร์รี", "🏮"),
    "pineapple": ("สับปะรด", "🍍"),
    "pistachio": ("พิสตาชิโอ", "🥜"),
    "pitahaya": ("แก้วมังกร", "🐲"),
    "dragonfruit": ("แก้วมังกร", "🐲"),
    "plum": ("พลัม", "🫐"),
    "pomegranate": ("ทับทิม", "🍎"),
    "pomelo": ("ส้มโอ", "🍈"),
    "potato": ("มันฝรั่ง", "🥔"),
    "quince": ("ควินซ์", "🍐"),
    "rambutan": ("เงาะ", "🔴"),
    "raspberry": ("ราสเบอร์รี", "🫐"),
    "redcurrant": ("เรดเคอร์แรนต์", "🍒"),
    "salak": ("สละ", "🌰"),
    "strawberry": ("สตรอว์เบอร์รี", "🍓"),
    "tamarillo": ("มะเขือเทศต้น", "🍅"),
    "tangelo": ("ส้มแทนเจโล", "🍊"),
    "tomato": ("มะเขือเทศ", "🍅"),
    "walnut": ("วอลนัท", "🌰"),
    "watermelon": ("แตงโม", "🍉"),
    "zucchini": ("ซูกินี", "🥒"),
}


def format_class_name(class_name: str) -> str:
    """แปลงชื่อคลาสภาษาอังกฤษเป็นชื่อแสดงผลภาษาไทย + ภาษาอังกฤษ + Emoji"""
    cls_lower = class_name.lower().replace("_", " ")
    pretty_en = " ".join([w.capitalize() for w in cls_lower.split()])

    # เรียงลำดับคำค้นหาตามความยาว (คำยาวกว่าตรวจก่อน เช่น mangostan ก่อน mango)
    sorted_keywords = sorted(FRUIT_KEYWORD_MAP.items(), key=lambda x: len(x[0]), reverse=True)
    for keyword, (th_name, emoji) in sorted_keywords:
        if keyword in cls_lower:
            return f"{th_name} ({pretty_en} {emoji})"

    return f"{pretty_en} 🌿"


def build_thai_labels(classes):
    """สร้าง Dict แปลชื่อทุกคลาสที่พบ"""
    return {c: format_class_name(c) for c in classes}


def extract_features(image: Image.Image, mode="combined", img_size=(32, 32)):
    """
    สกัด Feature จากรูปภาพ
    - mode='flatten': ย่อภาพเป็น 32x32 แล้วแปลงเป็น 1D Array (Normalized [0, 1])
    - mode='histogram': คำนวณ Color Histogram ทั้ง RGB และ HSV
    - mode='combined': รวมทั้งเวกเตอร์พิกเซล 32x32, RGB Histogram และ HSV Color Distribution
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    resized_img = image.resize(img_size)
    img_array = np.array(resized_img, dtype=np.float32)

    # 1. Flatten Features (32x32x3 = 3072 dims)
    flatten_feat = (img_array / 255.0).flatten()

    # 2. RGB Color Histogram (ช่องละ 16 bins = 48 dims)
    hist_r, _ = np.histogram(img_array[:, :, 0], bins=16, range=(0, 256), density=True)
    hist_g, _ = np.histogram(img_array[:, :, 1], bins=16, range=(0, 256), density=True)
    hist_b, _ = np.histogram(img_array[:, :, 2], bins=16, range=(0, 256), density=True)
    rgb_feat = np.hstack([hist_r, hist_g, hist_b])

    # 3. HSV Color Histogram (Hue 24 bins, Saturation 16 bins, Value 16 bins = 56 dims)
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


def load_dataset(dataset_dir="dataset", mode="combined", img_size=(32, 32)):
    """โหลดภาพจากทุกโฟลเดอร์ใน dataset/ และสกัดฟีเจอร์"""
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์ dataset: {dataset_path.resolve()}")

    X = []
    y = []

    # ค้นหาโฟลเดอร์ย่อย (แต่ละโฟลเดอร์คือ 1 คลาสผลไม้)
    subdirs = sorted([d for d in dataset_path.iterdir() if d.is_dir() and not d.name.startswith((".", "_"))])
    if not subdirs:
        raise ValueError(f"ไม่พบคลาสผลไม้ในโฟลเดอร์: {dataset_path.resolve()}")

    class_names = [d.name.lower() for d in subdirs]

    print(f"\n กำลังอ่านข้อมูลรูปภาพและสกัดฟีเจอร์จาก {len(class_names)} คลาส...")
    total_imgs = 0
    for class_dir in subdirs:
        class_name = class_dir.name.lower()
        img_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpeg"))

        for img_path in img_files:
            try:
                with Image.open(img_path) as img:
                    feat = extract_features(img, mode=mode, img_size=img_size)
                    X.append(feat)
                    y.append(class_name)
                    total_imgs += 1
            except Exception as e:
                pass

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    print(f" อ่านสำเร็จทั้งหมด {total_imgs} รูปภาพ | ขนาด X={X.shape}, y={y.shape}")
    return X, y, class_names


def train_and_evaluate(dataset_dir="dataset", feature_mode="combined", model_type="linear", output_model="fruit_model.pkl"):
    img_size = (32, 32)
    X, y, class_names = load_dataset(dataset_dir, mode=feature_mode, img_size=img_size)

    if len(class_names) < 2:
        raise ValueError("ต้องการข้อมูลอย่างน้อย 2 คลาสขึ้นไปในการเทรนโมเดล")

    # แบ่งชุดข้อมูล Train 80% / Test 20%
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f" จำนวนข้อมูล Train: {len(X_train)} รูป | Test: {len(X_test)} รูป")

    # กำหนดโมเดล
    if model_type.lower() in ["linear", "lr", "logistic"]:
        clf = LogisticRegression(max_iter=150, C=1.0, random_state=42)
        model_name = "Logistic Regression (Linear Classifier)"
    elif model_type.lower() == "rf":
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        model_name = "Random Forest Classifier"
    elif model_type.lower() == "svm":
        if len(class_names) > 40:
            print(" [คำเตือน] จำนวนคลาสมากกว่า 40 คลาส แนะนำให้ใช้ linear เพื่อความเร็วและประหยัด RAM")
            clf = LogisticRegression(max_iter=150, C=1.0, random_state=42)
            model_name = "Logistic Regression (Linear Classifier)"
        else:
            clf = SVC(kernel="rbf", C=1.0, probability=True, random_state=42)
            model_name = "Support Vector Machine (SVM RBF)"
    else:
        clf = LogisticRegression(max_iter=150, C=1.0, random_state=42)
        model_name = "Logistic Regression (Linear Classifier)"

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", clf)
    ])

    print(f"\n กำลังเทรนโมเดล {model_name} สำหรับ {len(class_names)} คลาส...")
    pipeline.fit(X_train, y_train)

    # ประเมินผลบน Test Set
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("\n" + "=" * 60)
    print(f" ผลการทดสอบโมเดล: {model_name}")
    print(f" จำนวนคลาสทั้งหมด: {len(class_names)} คลาส")
    print(f" Accuracy บน Test Set: {accuracy * 100:.2f}%")
    print("=" * 60)

    report = classification_report(y_test, y_pred, zero_division=0)
    print(" รายงานการจำแนกประเภท (Classification Report Summary):")
    # พิมพ์สรุปหัวท้าย
    report_lines = report.strip().split("\n")
    print("\n".join(report_lines[:5]))
    print(f" ... [และคลาสผลไม้อื่น ๆ รวม {len(class_names)} คลาส] ...")
    print("\n".join(report_lines[-4:]))

    # บันทึกรายงานเป็นไฟล์ Text
    with open("classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\nClasses: {len(class_names)}\nFeature Mode: {feature_mode}\nAccuracy: {accuracy * 100:.2f}%\n\n")
        f.write(report)

    # วาดและบันทึก Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, cmap="Blues", cbar=True)
    plt.title(f"Confusion Matrix ({len(class_names)} Classes)\nAccuracy: {accuracy * 100:.1f}%")
    plt.xlabel("Predicted Class Index")
    plt.ylabel("True Class Index")
    plt.tight_layout()
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path, dpi=180)
    plt.close()
    print(f" บันทึกภาพ Confusion Matrix ที่: {cm_path}")

    # สร้าง Dict ภาษาไทยสำหรับทุกคลาส
    thai_labels = build_thai_labels(class_names)

    # บันทึกโมเดลพร้อม Metadata สำหรับ Gradio
    model_payload = {
        "version": __version__,
        "pipeline": pipeline,
        "classes": class_names,
        "thai_labels": thai_labels,
        "feature_mode": feature_mode,
        "img_size": img_size,
        "model_type": model_name,
        "accuracy": accuracy,
    }
    joblib.dump(model_payload, output_model, compress=3)
    file_size_mb = os.path.getsize(output_model) / (1024 * 1024)
    print(f" บันทึกไฟล์โมเดลเสร็จเรียบร้อย: {output_model} (ขนาด: {file_size_mb:.2f} MB)")
    print("=" * 60)
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="เทรนโมเดลจำแนกผลไม้")
    parser.add_argument("--data", type=str, default="dataset", help="โฟลเดอร์ชุดข้อมูล")
    parser.add_argument("--feature", type=str, default="combined", help="ประเภทฟีเจอร์")
    parser.add_argument("--model", type=str, default="linear", help="ประเภทโมเดล: linear, svm, rf")
    parser.add_argument("--output", type=str, default="fruit_model.pkl", help="ชื่อไฟล์โมเดลปลายทาง")
    args = parser.parse_args()

    train_and_evaluate(
        dataset_dir=args.data,
        feature_mode=args.feature,
        model_type=args.model,
        output_model=args.output
    )
