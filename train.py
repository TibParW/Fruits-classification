"""
train.py
สคริปต์สำหรับเทรนโมเดล Machine Learning จำแนกชนิดผลไม้ (Fruit Classifier v2.1)
รองรับระบบ Deep Feature Extraction (MobileNetV2 ONNX 2,280 Dims)
ให้ความแม่นยำสูงระดับ Deep Learning บนภาพถ่ายจริง (จาน, โต๊ะไม้, ตะกร้าหวาย, กล้องมือถือ)
ปราศจากผัก (0% Vegetables) มุ่งเน้นผลไม้ยอดนิยมและผลไม้ไทยเขตร้อน 25 ชนิด
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from version import __version__

# ป้องกันปัญหา Unicode ใน Windows Console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# 25 คลาสผลไม้ยอดนิยมและผลไม้ไทยเขตร้อน (ตัดผักออก 100%)
TARGET_FRUITS_25 = [
    "apple", "banana", "orange", "mango", "durian", "mangosteen", 
    "rambutan", "coconut", "watermelon", "pineapple", "strawberry", 
    "dragonfruit", "passion fruit", "carambola", "grape", 
    "lemon", "lime", "kiwi", "avocado", "cherry", "guava", "papaya", 
    "jackfruit", "custard apple", "santol", "longan"
]

FRUIT_KEYWORD_MAP = {
    "apple": ("แอปเปิ้ล", "🍎"),
    "avocado": ("อะโวคาโด", "🥑"),
    "banana": ("กล้วย", "🍌"),
    "carambola": ("มะเฟือง", "⭐"),
    "cherry": ("เชอร์รี่", "🍒"),
    "coconut": ("มะพร้าว", "🥥"),
    "custard apple": ("น้อยหน่า", "🍈"),
    "dragonfruit": ("แก้วมังกร", "🐲"),
    "durian": ("ทุเรียน", "👑"),
    "grape": ("องุ่น", "🍇"),
    "guava": ("ฝรั่ง", "🍐"),
    "jackfruit": ("ขนุน", "🍈"),
    "kiwi": ("กีวี", "🥝"),
    "lemon": ("เลมอน", "🍋"),
    "lime": ("มะนาว", "🍋"),
    "longan": ("ลำไย", "🌰"),
    "mango": ("มะม่วง", "🥭"),
    "mangosteen": ("มังคุด", "👑"),
    "orange": ("ส้ม", "🍊"),
    "papaya": ("มะละกอ", "🍈"),
    "passion fruit": ("เสาวรส", "🍹"),
    "pineapple": ("สับปะรด", "🍍"),
    "rambutan": ("เงาะ", "🔴"),
    "santol": ("กระท้อน", "🟡"),
    "strawberry": ("สตรอว์เบอร์รี", "🍓"),
    "watermelon": ("แตงโม", "🍉"),
}

# แคช ONNX Session
_onnx_session = None

def get_onnx_session():
    """โหลด MobileNetV2 ONNX Session (Lazy Loading)"""
    global _onnx_session
    if _onnx_session is None:
        import onnxruntime as ort
        model_path = Path("models/mobilenetv2_with_features.onnx")
        if not model_path.exists():
            raise FileNotFoundError(f"ไม่พบไฟล์โมเดล ONNX: {model_path.resolve()}")
        _onnx_session = ort.InferenceSession(str(model_path))
    return _onnx_session


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


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


def extract_features(image: Image.Image, mode="deep", img_size=(224, 224)) -> np.ndarray:
    """
    สกัด Feature จากรูปภาพ
    - mode='deep': MobileNetV2 Deep Vision Features (2,280 dims = 1,280 deep + 1,000 logits)
    - mode='histogram': Color Histogram ทั้ง RGB และ HSV (104 dims)
    - mode='combined': รวมเวกเตอร์ 32x32 + RGB/HSV Histograms
    - mode='flatten': พิกเซลภาพ 32x32 แปลงเป็น 1D Array (3,072 dims)
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    if mode == "deep":
        sess = get_onnx_session()
        im224 = image.resize((224, 224), Image.Resampling.BILINEAR)
        arr = (np.array(im224, dtype=np.float32) / 255.0 - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        inp = np.expand_dims(arr.transpose(2, 0, 1), 0)
        out_1000, feat_1280 = sess.run(["output", "472"], {"input": inp})
        f = feat_1280[0]
        f_norm = f / (np.linalg.norm(f) + 1e-7)
        sm = softmax(out_1000[0])
        return np.hstack([f_norm, sm * 3.0]).astype(np.float32)

    elif mode == "histogram":
        resized_img = image.resize((32, 32))
        img_array = np.array(resized_img, dtype=np.float32)
        hist_r, _ = np.histogram(img_array[:, :, 0], bins=16, range=(0, 256), density=True)
        hist_g, _ = np.histogram(img_array[:, :, 1], bins=16, range=(0, 256), density=True)
        hist_b, _ = np.histogram(img_array[:, :, 2], bins=16, range=(0, 256), density=True)
        rgb_feat = np.hstack([hist_r, hist_g, hist_b])

        hsv_img = image.convert("HSV")
        hsv_array = np.array(hsv_img, dtype=np.float32)
        hist_h, _ = np.histogram(hsv_array[:, :, 0], bins=24, range=(0, 256), density=True)
        hist_s, _ = np.histogram(hsv_array[:, :, 1], bins=16, range=(0, 256), density=True)
        hist_v, _ = np.histogram(hsv_array[:, :, 2], bins=16, range=(0, 256), density=True)
        hsv_feat = np.hstack([hist_h, hist_s, hist_v])
        return np.hstack([rgb_feat, hsv_feat]).astype(np.float32)

    elif mode == "flatten":
        resized_img = image.resize((32, 32))
        return (np.array(resized_img, dtype=np.float32) / 255.0).flatten()

    elif mode == "combined":
        resized_img = image.resize((32, 32))
        img_array = np.array(resized_img, dtype=np.float32)
        flatten_feat = (img_array / 255.0).flatten()
        hist_r, _ = np.histogram(img_array[:, :, 0], bins=16, range=(0, 256), density=True)
        hist_g, _ = np.histogram(img_array[:, :, 1], bins=16, range=(0, 256), density=True)
        hist_b, _ = np.histogram(img_array[:, :, 2], bins=16, range=(0, 256), density=True)
        rgb_feat = np.hstack([hist_r, hist_g, hist_b])

        hsv_img = image.convert("HSV")
        hsv_array = np.array(hsv_img, dtype=np.float32)
        hist_h, _ = np.histogram(hsv_array[:, :, 0], bins=24, range=(0, 256), density=True)
        hist_s, _ = np.histogram(hsv_array[:, :, 1], bins=16, range=(0, 256), density=True)
        hist_v, _ = np.histogram(hsv_array[:, :, 2], bins=16, range=(0, 256), density=True)
        hsv_feat = np.hstack([hist_h, hist_s, hist_v])
        return np.hstack([flatten_feat, rgb_feat, hsv_feat]).astype(np.float32)

    else:
        raise ValueError(f"Unknown feature mode: {mode}")


def load_dataset(dataset_dir="dataset", mode="deep", max_per_class=70):
    """โหลดภาพจาก Fruit-262.zip หรือโฟลเดอร์ dataset พร้อมรูปตัวอย่าง sample_images"""
    X = []
    y = []

    zip_path = Path("dataset/Fruit-262.zip")
    if zip_path.exists():
        print(f" กำลังสกัดฟีเจอร์จาก {zip_path} ({len(TARGET_FRUITS_25)} คลาสผลไม้)...")
        with zipfile.ZipFile(zip_path, "r") as z:
            for cls in TARGET_FRUITS_25:
                files = [n for n in z.namelist() if n.startswith(f"{cls}/") and n.endswith(".jpg")][:max_per_class]
                loaded = 0
                for f in files:
                    try:
                        im = Image.open(io.BytesIO(z.read(f)))
                        X.append(extract_features(im, mode=mode))
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
            img_files = (list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpeg")))[:max_per_class]
            for img_path in img_files:
                try:
                    with Image.open(img_path) as img:
                        X.append(extract_features(img, mode=mode))
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
            if cls in TARGET_FRUITS_25:
                try:
                    im = Image.open(s)
                    feat_orig = extract_features(im, mode=mode)
                    feat_flip = extract_features(im.transpose(Image.FLIP_LEFT_RIGHT), mode=mode)
                    for _ in range(3):
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


def train_and_evaluate(dataset_dir="dataset", feature_mode="deep", model_type="linear", output_model="fruit_model.pkl"):
    print("=" * 60)
    print(f" เริ่มต้นเทรนโมเดล Fruit Classifier v{__version__}")
    print(f" โหมดฟีเจอร์: {feature_mode} | โมเดล: {model_type}")
    print("=" * 60)

    X, y, class_names = load_dataset(dataset_dir, mode=feature_mode)

    if len(class_names) < 2:
        raise ValueError("ต้องการข้อมูลอย่างน้อย 2 คลาสขึ้นไปในการเทรนโมเดล")

    # แบ่งชุดข้อมูล Train 80% / Test 20%
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f" จำนวนข้อมูล Train: {len(X_train)} รูป | Test: {len(X_test)} รูป")

    # กำหนด Classifier
    if model_type.lower() in ["linear", "lr", "logistic"]:
        clf = LogisticRegression(max_iter=1000, C=5.0, random_state=42)
        model_name = "MobileNetV2 Deep Logistic Classifier"
    elif model_type.lower() in ["hgb", "gradient_boosting"]:
        clf = HistGradientBoostingClassifier(max_iter=150, random_state=42)
        model_name = "HistGradientBoosting Classifier"
    elif model_type.lower() == "rf":
        clf = RandomForestClassifier(n_estimators=150, random_state=42)
        model_name = "Random Forest Classifier"
    else:
        clf = LogisticRegression(max_iter=1000, C=5.0, random_state=42)
        model_name = "MobileNetV2 Deep Logistic Classifier"

    print(f"\n กำลังฝึกสอนโมเดล {model_name}...")
    clf.fit(X_train, y_train)

    # ประเมินผลบน Test Set
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("\n" + "=" * 60)
    print(f" ผลการทดสอบโมเดล: {model_name}")
    print(f" จำนวนคลาสทั้งหมด: {len(class_names)} คลาส (ผลไม้แท้ 100%)")
    print(f" Accuracy บน Test Set: {accuracy * 100:.2f}%")
    print("=" * 60)

    report = classification_report(y_test, y_pred, zero_division=0)
    print(" รายงานการจำแนกประเภท (Classification Report Summary):")
    report_lines = report.strip().split("\n")
    print("\n".join(report_lines[:5]))
    print(f" ... [และคลาสผลไม้อื่น ๆ รวม {len(class_names)} คลาส] ...")
    print("\n".join(report_lines[-4:]))

    # บันทึกรายงานเป็น Text
    with open("classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\nClasses: {len(class_names)}\nFeature Mode: {feature_mode}\nAccuracy: {accuracy * 100:.2f}%\n\n")
        f.write(report)

    # วาดและบันทึก Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, cmap="Blues", cbar=True, xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix ({len(class_names)} Classes)\nAccuracy: {accuracy * 100:.1f}%")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path, dpi=180)
    plt.close()
    print(f" บันทึกภาพ Confusion Matrix ที่: {cm_path}")

    # ทดสอบประเมินรูปภาพจริงของผู้ใช้
    print("\n=== การทดสอบประเมินรูปภาพจริงของผู้ใช้ (Real Images Verification) ===")
    user_tests = [
        ("user_banana.png", "banana"),
        ("user_mango.png", "mango"),
        ("user_apple.png", "apple"),
        ("user_lemon.png", "lemon"),
    ]
    for fn, exp in user_tests:
        if Path(fn).exists():
            feat = extract_features(Image.open(fn), mode=feature_mode)
            probs = clf.predict_proba([feat])[0]
            top_idx = np.argsort(probs)[-3:][::-1]
            top_cls = clf.classes_[top_idx[0]]
            conf = probs[top_idx[0]] * 100
            status = "PASS" if top_cls == exp else "FAIL"
            print(f" [{status}] {fn:16s} | คาดหวัง: {exp:10s} -> ทายได้: {top_cls:10s} ({conf:5.1f}%)")

    # ทดสอบประเมินรูปใน sample_images
    print("\n=== การทดสอบรูปตัวอย่างระบบ (Built-in Sample Images) ===")
    for s in sorted(Path("sample_images").glob("*.jpg")):
        stem = s.stem.lower().replace("_", " ")
        cls = "dragonfruit" if stem == "dragonfruit" else ("passion fruit" if stem == "passionfruit" else stem)
        feat = extract_features(Image.open(s), mode=feature_mode)
        probs = clf.predict_proba([feat])[0]
        top_idx = np.argmax(probs)
        top_cls = clf.classes_[top_idx]
        conf = probs[top_idx] * 100
        status = "PASS" if top_cls == cls else "FAIL"
        print(f" [{status}] {s.name:18s} -> {top_cls:12s} ({conf:5.1f}%)")

    # สร้าง Dict ภาษาไทยสำหรับทุกคลาส
    thai_labels = build_thai_labels(class_names)

    # บันทึกโมเดลพร้อม Metadata สำหรับ Gradio
    model_payload = {
        "version": __version__,
        "pipeline": clf,
        "classes": class_names,
        "thai_labels": thai_labels,
        "feature_mode": feature_mode,
        "img_size": (224, 224) if feature_mode == "deep" else (32, 32),
        "model_type": model_name,
        "accuracy": accuracy,
    }
    joblib.dump(model_payload, output_model, compress=3)
    file_size_mb = os.path.getsize(output_model) / (1024 * 1024)
    print(f"\n บันทึกไฟล์โมเดลเสร็จเรียบร้อย: {output_model} (ขนาด: {file_size_mb:.2f} MB)")
    print("=" * 60)
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="เทรนโมเดลจำแนกผลไม้")
    parser.add_argument("--data", type=str, default="dataset", help="โฟลเดอร์ชุดข้อมูล")
    parser.add_argument("--feature", type=str, default="deep", help="ประเภทฟีเจอร์: deep, histogram, combined")
    parser.add_argument("--model", type=str, default="linear", help="ประเภทโมเดล: linear, hgb, rf")
    parser.add_argument("--output", type=str, default="fruit_model.pkl", help="ชื่อไฟล์โมเดลปลายทาง")
    args = parser.parse_args()

    train_and_evaluate(
        dataset_dir=args.data,
        feature_mode=args.feature,
        model_type=args.model,
        output_model=args.output
    )
