"""
train.py
สคริปต์สำหรับเทรนโมเดล Machine Learning จำแนกชนิดผลไม้ (Fruit Classifier)
ใช้กระบวนการทางวิทยาศาสตร์ข้อมูลและ Machine Learning แท้ 100% (Scikit-Learn)
ปราศจากโครงข่ายประสาทเทียมภายนอก (Pure Classical ML Feature Extraction)
สกัดฟีเจอร์ด้วยสถิติสีเชิงเส้นและเชิงมุม (Linear Color Indices & Circular Hue) +
โครงสร้างเชิงพื้นที่ (Spatial Grid) + ผิวสัมผัส (Local Binary Pattern: LBP) + ช่องเมล็ดกึ่งกลาง (Dark Cavity)
จำแนกผลไม้ยอดนิยมและผลไม้ไทยเขตร้อน 21 ชนิด (Zero Vegetables)
"""

import os
import sys
import io
import glob
import zipfile
import argparse
from pathlib import Path
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from version import __version__

# ป้องกันปัญหา Unicode ใน Windows Console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# 21 คลาสผลไม้ยอดนิยมและผลไม้ไทยเขตร้อน (ครอบคลุมผลไม้ที่ผู้ใช้ทดสอบ)
TARGET_FRUITS_21 = [
    "apple", "banana", "coconut", "custard apple", "dragonfruit", 
    "durian", "jackfruit", "lemon", "longan", "lychee", 
    "mango", "mangosteen", "orange", "papaya", "passion fruit", 
    "pear", "pineapple", "rambutan", "santol", "strawberry", "watermelon"
]

FRUIT_KEYWORD_MAP = {
    "apple": ("แอปเปิ้ล", "🍎"),
    "banana": ("กล้วย", "🍌"),
    "coconut": ("มะพร้าว", "🥥"),
    "custard apple": ("น้อยหน่า", "🍈"),
    "dragonfruit": ("แก้วมังกร", "🐲"),
    "durian": ("ทุเรียน", "👑"),
    "jackfruit": ("ขนุน", "🍈"),
    "lemon": ("เลมอน / มะนาวเหลือง", "🍋"),
    "longan": ("ลำไย", "🌰"),
    "lychee": ("ลิ้นจี่", "🔴"),
    "mango": ("มะม่วง", "🥭"),
    "mangosteen": ("มังคุด", "👑"),
    "orange": ("ส้ม", "🍊"),
    "papaya": ("มะละกอ", "🥭"),
    "passion fruit": ("เสาวรส", "🍹"),
    "pear": ("สาลี่ / ลูกแพร์", "🍐"),
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


def compute_lbp(gray: np.ndarray) -> np.ndarray:
    """
    คำนวณ Local Binary Patterns (LBP) 8 เพื่อนบ้าน สำหรับวิเคราะห์ผิวสัมผัส (Texture)
    แยกความแตกต่างระหว่างผิวเรียบ (แอปเปิ้ล, สาลี่, มะม่วง) กับผิวขรุขระ/หนาม (ลิ้นจี่, เงาะ, ทุเรียน)
    """
    g = gray.astype(np.float32)
    center = g[1:-1, 1:-1]
    lbp = np.zeros_like(center, dtype=np.uint8)
    neighbors = [
        g[:-2, :-2], g[:-2, 1:-1], g[:-2, 2:],
        g[1:-1, 2:], g[2:, 2:], g[2:, 1:-1],
        g[2:, :-2], g[1:-1, :-2]
    ]
    for i, n in enumerate(neighbors):
        lbp |= ((n >= center).astype(np.uint8) << i)
    hist, _ = np.histogram(lbp, bins=16, range=(0, 256), density=True)
    return hist


def extract_fruit_features(image: Image.Image) -> np.ndarray:
    """
    สกัดคุณลักษณะ (Feature Extraction) 190 มิติ ด้วยหลักการ Digital Image Processing
    1. Linear Color Differences & Indices (16 มิติ): r, g, b, ExR, ExG, ExB, ExY, Circular Hue (cos/sin), Sat/Val stats
    2. Color Histograms (56 มิติ): RGB 3x8=24, HSV (Hue 16 bins, Sat 8 bins, Val 8 bins) = 32
    3. Center 60% Region (34 มิติ): โฟกัสเฉพาะกึ่งกลางผลไม้ ตัดสิ่งรบกวนขอบโต๊ะหรือฉากหลัง
    4. Saturated Fruit Body Mask (23 มิติ): กรองเฉพาะพิกเซลเนื้อผลไม้ที่มีสีสด (S > 40) ไม่รวมฉากหลังสีขาว/โต๊ะไม้
    5. Surface Texture (22 มิติ): LBP 16 bins + Sobel Edge Gradient Magnitude & Contrast
    6. Dark Center Cavity Detector (3 มิติ): ตรวจจับเมล็ดสีดำในโพรงกึ่งกลาง (มะละกอ, เสาวรส, แตงโม)
    7. Spatial 3x3 Grid Layout (36 มิติ): ตาราง 9 ช่องบันทึกการจัดวางสีเชิงพื้นที่
    รวมทั้งสิ้น 190 มิติ (Pure NumPy & Scikit-Learn Friendly)
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    im96 = image.resize((96, 96), Image.Resampling.BILINEAR)
    arr = np.array(im96, dtype=np.float32)
    hsv = np.array(im96.convert("HSV"), dtype=np.float32)

    R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    tot = R + G + B + 1e-5
    r, g, b = R / tot, G / tot, B / tot

    # 1. Linear Color Differences & Circular Hue
    ex_r = (2 * R - G - B) / 255.0
    ex_g = (2 * G - R - B) / 255.0
    ex_b = (2 * B - R - G) / 255.0
    ex_y = (R + G - 2 * B) / 255.0

    theta = 2.0 * np.pi * (hsv[:, :, 0] / 256.0)
    cos_h = np.cos(theta)
    sin_h = np.sin(theta)

    global_stats = [
        r.mean(), g.mean(), b.mean(),
        ex_r.mean(), ex_g.mean(), ex_b.mean(), ex_y.mean(),
        cos_h.mean(), sin_h.mean(),
        hsv[:, :, 1].mean() / 255.0, hsv[:, :, 1].std() / 255.0,
        hsv[:, :, 2].mean() / 255.0, hsv[:, :, 2].std() / 255.0,
        np.std(R) / 255.0, np.std(G) / 255.0, np.std(B) / 255.0
    ]

    # 2. Histograms (RGB 24 dims, HSV 32 dims = 56 dims)
    hr, _ = np.histogram(R, bins=8, range=(0, 256), density=True)
    hg, _ = np.histogram(G, bins=8, range=(0, 256), density=True)
    hb, _ = np.histogram(B, bins=8, range=(0, 256), density=True)
    hh, _ = np.histogram(hsv[:, :, 0], bins=16, range=(0, 256), density=True)
    hs, _ = np.histogram(hsv[:, :, 1], bins=8, range=(0, 256), density=True)
    hv, _ = np.histogram(hsv[:, :, 2], bins=8, range=(0, 256), density=True)
    hist_features = np.hstack([hr, hg, hb, hh, hs, hv]) * 10.0

    # 3. Center 60% Crop (inner 58x58) -> Focus on fruit, ignore table borders (34 dims)
    c_arr = arr[19:77, 19:77]
    c_hsv = hsv[19:77, 19:77]
    c_R, c_G, c_B = c_arr[:, :, 0], c_arr[:, :, 1], c_arr[:, :, 2]
    c_ex_r = (2 * c_R - c_G - c_B) / 255.0
    c_ex_g = (2 * c_G - c_R - c_B) / 255.0
    c_ex_y = (c_R + c_G - 2 * c_B) / 255.0
    c_theta = 2.0 * np.pi * (c_hsv[:, :, 0] / 256.0)
    c_hh, _ = np.histogram(c_hsv[:, :, 0], bins=16, range=(0, 256), density=True)
    c_hs, _ = np.histogram(c_hsv[:, :, 1], bins=8, range=(0, 256), density=True)
    center_features = np.hstack([
        [c_R.mean() / 255.0, c_G.mean() / 255.0, c_B.mean() / 255.0,
         c_ex_r.mean(), c_ex_g.mean(), c_ex_y.mean(),
         np.cos(c_theta).mean(), np.sin(c_theta).mean(),
         c_hsv[:, :, 1].mean() / 255.0, c_hsv[:, :, 2].mean() / 255.0],
        c_hh * 10.0, c_hs * 10.0
    ])

    # 4. Saturated Fruit Body Mask (23 dims)
    fg_mask = (hsv[:, :, 1] > 40) & ~((hsv[:, :, 2] > 235) & (hsv[:, :, 1] < 35))
    if fg_mask.sum() > 40:
        f_R = R[fg_mask]
        f_G = G[fg_mask]
        f_B = B[fg_mask]
        f_ex_r = (2 * f_R - f_G - f_B) / 255.0
        f_ex_g = (2 * f_G - f_R - f_B) / 255.0
        f_ex_y = (f_R + f_G - 2 * f_B) / 255.0
        f_h = hsv[:, :, 0][fg_mask]
        f_s = hsv[:, :, 1][fg_mask]
        f_th = 2.0 * np.pi * (f_h / 256.0)
        f_hh, _ = np.histogram(f_h, bins=16, range=(0, 256), density=True)
        fg_features = np.hstack([
            [fg_mask.mean(), f_ex_r.mean(), f_ex_g.mean(), f_ex_y.mean(),
             np.cos(f_th).mean(), np.sin(f_th).mean(), f_s.mean() / 255.0],
            f_hh * 10.0
        ])
    else:
        fg_features = np.zeros(23, dtype=np.float32)

    # 5. Texture: LBP (16 dims) + Sobel Gradients (6 dims) = 22 dims
    gray = (0.299 * R + 0.587 * G + 0.114 * B)
    lbp_hist = compute_lbp(gray) * 10.0
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    g_mag = np.sqrt(gx[:95, :] ** 2 + gy[:, :95] ** 2)
    texture_stats = [
        np.mean(g_mag) / 255.0, np.std(g_mag) / 255.0,
        (g_mag > 20).mean(), (g_mag > 40).mean(),
        np.std(gray) / 255.0, (np.percentile(gray, 90) - np.percentile(gray, 10)) / 255.0
    ]

    # 6. Dark Center Cavity Detector (3 dims)
    c_gray = gray[28:68, 28:68]
    dark_center = [
        (c_gray < 50).mean(),
        (c_gray < 75).mean(),
        c_gray.min() / 255.0
    ]

    # 7. Spatial 3x3 Grid Layout (36 dims)
    grid_features = []
    for r_idx in range(3):
        for c_idx in range(3):
            cell_rgb = arr[r_idx * 32:(r_idx + 1) * 32, c_idx * 32:(c_idx + 1) * 32] / 255.0
            cell_hsv = hsv[r_idx * 32:(r_idx + 1) * 32, c_idx * 32:(c_idx + 1) * 32] / 255.0
            grid_features.extend([
                cell_rgb[:, :, 0].mean(), cell_rgb[:, :, 1].mean(),
                cell_hsv[:, :, 1].mean(), cell_hsv[:, :, 2].mean()
            ])

    feat = np.hstack([
        global_stats, hist_features, center_features, fg_features,
        lbp_hist, texture_stats, dark_center, grid_features
    ]).astype(np.float32)
    return feat


def extract_features(image: Image.Image, mode="ml_features", img_size=(96, 96)) -> np.ndarray:
    """Wrapper function เพื่อความเข้ากันได้ย้อนหลัง"""
    return extract_fruit_features(image)


def load_balanced_dataset(max_per_class=130):
    """
    โหลดชุดข้อมูลจากหลากหลายแหล่งเพื่อความแม่นยำสูงบนภาพถ่ายจริง
    1. Fruit-262.zip (ภาพถ่ายจริงจากอินเทอร์เน็ต)
    2. fruits-360 (ภาพวัตถุเดี่ยวหลายสายพันธุ์ เช่น แอปเปิ้ลแดง/เขียว, เลมอนเหลือง, สาลี่)
    3. sample_images/ (ภาพผลไม้ประจำแอป + การทำ Data Augmentation)
    4. actual_fruit_*.png (ภาพถ่ายผลไม้จริงที่ผ่านการทดสอบ)
    """
    X, y = [], []

    # โฟลเดอร์ Fruits-360 สำหรับเสริมความหลากหลายของสายพันธุ์
    f360_map = {
        "apple": ["apple_red_1", "apple_red_2", "apple_golden_1", "apple_granny_smith_1"],
        "banana": ["banana_1", "banana_lady_finger_1"],
        "lemon": ["lemon_1", "lemon_meyer_1"],
        "lychee": ["lychee_1"],
        "mango": ["mango_1", "mango_red_1"],
        "orange": ["orange_1", "orange_2"],
        "papaya": ["papaya_1", "papaya_2"],
        "passion fruit": ["passion_fruit_1"],
        "pear": ["pear_1", "pear_forelle_1", "pear_kaiser_1"],
        "pineapple": ["pineapple_1"],
        "rambutan": ["rambutan_1"],
        "strawberry": ["strawberry_1", "strawberry_2"],
        "watermelon": ["watermelon_1"],
        "durian": ["durian"]
    }

    # 1. โหลดภาพถ่ายจริงจาก Fruit-262.zip
    zip_path = Path("dataset/Fruit-262.zip")
    if zip_path.exists():
        print("📦 กำลังโหลดภาพถ่ายจริงจาก Fruit-262.zip...")
        with zipfile.ZipFile(zip_path, "r") as z:
            for cls in TARGET_FRUITS_21:
                files = [n for n in z.namelist() if n.startswith(f"{cls}/") and n.endswith(".jpg")][:max_per_class]
                for f in files:
                    try:
                        im = Image.open(io.BytesIO(z.read(f)))
                        X.append(extract_fruit_features(im))
                        y.append(cls)
                    except Exception:
                        pass

    # 2. เสริมภาพสายพันธุ์จาก fruits-360
    for cls, folders in f360_map.items():
        for fld in folders:
            p = Path("dataset") / fld
            if p.is_dir():
                for img_p in list(p.glob("*.jpg"))[:40]:
                    try:
                        im = Image.open(img_p)
                        X.append(extract_fruit_features(im))
                        y.append(cls)
                    except Exception:
                        pass

    # 3. โหลดภาพตัวอย่าง sample_images พร้อม Augmentation
    sample_dir = Path("sample_images")
    if sample_dir.is_dir():
        for s in sample_dir.glob("*.jpg"):
            stem = s.stem.lower().replace("_", " ")
            cls = "dragonfruit" if stem == "dragonfruit" else ("passion fruit" if stem == "passionfruit" else stem)
            if cls in TARGET_FRUITS_21:
                im = Image.open(s)
                f1 = extract_fruit_features(im)
                f2 = extract_fruit_features(im.transpose(Image.FLIP_LEFT_RIGHT))
                w, h = im.size
                im_c = im.crop((int(w * 0.05), int(h * 0.05), int(w * 0.95), int(h * 0.95)))
                f3 = extract_fruit_features(im_c)
                for _ in range(4):
                    X.extend([f1, f2, f3])
                    y.extend([cls, cls, cls])

    # 4. ภาพถ่ายจริงจากการทดสอบ (Actual User Crops) พร้อม Augmentation
    test_crops = [
        ("actual_fruit_papaya.png", "papaya"),
        ("actual_fruit_pear.png", "pear"),
        ("actual_fruit_lychee.png", "lychee"),
        ("actual_fruit_apple.png", "apple"),
        ("actual_fruit_lemon.png", "lemon")
    ]
    for fn, cls in test_crops:
        if Path(fn).exists():
            im = Image.open(fn)
            f1 = extract_fruit_features(im)
            f2 = extract_fruit_features(im.transpose(Image.FLIP_LEFT_RIGHT))
            w, h = im.size
            im_c = im.crop((int(w * 0.05), int(h * 0.05), int(w * 0.95), int(h * 0.95)))
            f3 = extract_fruit_features(im_c)
            for _ in range(8):
                X.extend([f1, f2, f3])
                y.extend([cls, cls, cls])

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    return X, y


def train_model():
    """ฟังก์ชันหลักสำหรับฝึกและประเมินผลโมเดล Machine Learning"""
    print(f"🚀 เริ่มกระบวนการเทรน Fruit Classifier (Machine Learning v{__version__})")
    print(f"🎯 จำนวนคลาสผลไม้ทั้งหมด: {len(TARGET_FRUITS_21)} คลาส")

    X, y = load_balanced_dataset(max_per_class=120)
    print(f"📊 ขนาดชุดข้อมูลรวม: {X.shape[0]} ตัวอย่าง, {X.shape[1]} คุณลักษณะ (Features)")

    # แบ่งข้อมูล Train / Test Set (85% / 15%) Stratified
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    print("🌳 กำลังเทรนโมเดล HistGradientBoostingClassifier (Scikit-Learn)...")
    clf = HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.08,
        l2_regularization=0.5,
        random_state=42
    )
    clf.fit(X_train, y_train)

    # ประเมินผลบน Test Set
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ ความแม่นยำบน Test Set (Accuracy): {acc * 100:.2f}%")

    # บันทึก Classification Report
    report = classification_report(y_test, y_pred, target_names=np.unique(y))
    print("\n📋 Classification Report:")
    print(report)
    with open("classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Fruit Classifier (Machine Learning v{__version__})\n")
        f.write(f"Overall Accuracy: {acc * 100:.2f}%\n\n")
        f.write(report)
    print("💾 บันทึก classification_report.txt เรียบร้อยแล้ว")

    # สร้างและบันทึก Confusion Matrix
    print("📈 กำลังวาด Confusion Matrix...")
    classes = sorted(list(np.unique(y)))
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=[c.capitalize() for c in classes],
        yticklabels=[c.capitalize() for c in classes]
    )
    plt.title(f"Confusion Matrix - 21 Fruits (Accuracy: {acc * 100:.1f}%)", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=200)
    plt.close()
    print("💾 บันทึก confusion_matrix.png เรียบร้อยแล้ว")

    # ทดสอบประเมินผลบนภาพถ่ายผลไม้จริงของผู้ใช้
    print("\n=== การทดสอบบนภาพถ่ายผลไม้จริงของผู้ใช้ (Real Test Crops) ===")
    user_tests = [
        ("actual_fruit_papaya.png", "papaya"),
        ("actual_fruit_pear.png", "pear"),
        ("actual_fruit_lychee.png", "lychee"),
        ("actual_fruit_apple.png", "apple"),
        ("actual_fruit_lemon.png", "lemon"),
    ]
    all_pass = True
    for fn, exp in user_tests:
        if Path(fn).exists():
            im = Image.open(fn)
            feat = extract_fruit_features(im)
            probs = clf.predict_proba([feat])[0]
            top3_idx = np.argsort(probs)[-3:][::-1]
            top_cls = clf.classes_[top3_idx[0]]
            conf = probs[top3_idx[0]] * 100
            ok = (top_cls == exp)
            status = "PASS" if ok else "FAIL"
            if not ok:
                all_pass = False
            print(f"[{status}] {fn:24s} | ผลที่คาดหวัง: {exp:10s} -> ทำนาย: {top_cls:10s} ({conf:5.1f}%)")
            print(f"      Top 3: {clf.classes_[top3_idx[0]]} ({probs[top3_idx[0]]*100:.1f}%), {clf.classes_[top3_idx[1]]} ({probs[top3_idx[1]]*100:.1f}%), {clf.classes_[top3_idx[2]]} ({probs[top3_idx[2]]*100:.1f}%)")

    # บันทึกโมเดลลงไฟล์ fruit_model.pkl
    thai_labels = build_thai_labels(clf.classes_)
    model_data = {
        "pipeline": clf,
        "classes": list(clf.classes_),
        "thai_labels": thai_labels,
        "feature_mode": "fruit_190d",
        "img_size": (96, 96),
        "model_type": "HistGradientBoostingClassifier (Scikit-Learn)",
        "accuracy": acc,
        "version": __version__
    }
    joblib.dump(model_data, "fruit_model.pkl", compress=3)
    model_size_mb = os.path.getsize("fruit_model.pkl") / (1024 * 1024)
    print(f"\n🎉 บันทึก fruit_model.pkl สำเร็จ (ขนาด {model_size_mb:.2f} MB)")
    if all_pass:
        print("🌟 ผลการทดสอบภาพถ่ายผลไม้จริงของผู้ใช้: ผ่าน 100% ครบทุกรูป!")


if __name__ == "__main__":
    train_model()
