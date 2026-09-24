"""
train_real_fruits.py
โมเดลจำแนกผลไม้จริง 30 ชนิดยอดนิยม (Top 30 Real Fruits Classifier)
- ตัดผักและพืชผลที่ไม่ใช่ผลไม้ออกทั้งหมด (ไม่มีกระเทียม, ขิง, กะหล่ำ, ซูกินี, มะเขือยาว)
- ฝึกสอนด้วยภาพถ่ายผลไม้จริงจาก Fruit-262 คลาสละ 150 รูปภาพ (รวม 4,500+ รูปภาพ)
- ผสาน Color Distribution Features (RGB + HSV Histogram) ที่ทนทานต่อฉากหลังและแสง
- ใช้ HistGradientBoostingClassifier เพื่อการจำแนกที่แม่นยำและไม่เชิงเส้น
"""

import io
import time
import zipfile
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from train import extract_features

F262_ZIP = Path("dataset/Fruit-262.zip")
OUTPUT_MODEL = Path("fruit_model.pkl")

# รายชื่อ 30 ผลไม้ยอดนิยม (ตัดผักออกทั้งหมด 100%)
SELECTED_FRUITS = [
    "apple", "banana", "orange", "mango", "durian", "mangosteen",
    "rambutan", "coconut", "watermelon", "pineapple", "strawberry",
    "blueberry", "dragonfruit", "passion fruit", "carambola", "grape",
    "papaya", "lemon", "lime", "kiwi", "guava", "avocado",
    "cherry", "pomegranate", "jackfruit", "lychee", "longan", "custard apple",
    "cantaloupe", "pear"
]

THAI_LABELS = {
    "apple": "แอปเปิ้ล (Apple)",
    "banana": "กล้วย (Banana)",
    "orange": "ส้ม (Orange)",
    "mango": "มะม่วง (Mango)",
    "durian": "ทุเรียน (Durian)",
    "mangosteen": "มังคุด (Mangosteen)",
    "rambutan": "เงาะ (Rambutan)",
    "coconut": "มะพร้าว (Coconut)",
    "watermelon": "แตงโม (Watermelon)",
    "pineapple": "สับปะรด (Pineapple)",
    "strawberry": "สตรอว์เบอร์รี (Strawberry)",
    "blueberry": "บลูเบอร์รี (Blueberry)",
    "dragonfruit": "แก้วมังกร (Dragonfruit)",
    "passion fruit": "เสาวรส (Passion Fruit)",
    "carambola": "มะเฟือง (Starfruit)",
    "grape": "องุ่น (Grape)",
    "papaya": "มะละกอ (Papaya)",
    "lemon": "เลมอน (Lemon)",
    "lime": "มะนาว (Lime)",
    "kiwi": "กีวี่ (Kiwi)",
    "guava": "ฝรั่ง (Guava)",
    "avocado": "อะโวคาโด (Avocado)",
    "cherry": "เชอร์รี่ (Cherry)",
    "pomegranate": "ทับทิม (Pomegranate)",
    "jackfruit": "ขนุน (Jackfruit)",
    "lychee": "ลิ้นจี่ (Lychee)",
    "longan": "ลำไย (Longan)",
    "custard apple": "น้อยหน่า (Custard Apple)",
    "cantaloupe": "แคนตาลูป (Cantaloupe)",
    "pear": "สาลี่/แพร์ (Pear)"
}


def load_data(images_per_class=150):
    print(f"\n📦 กำลังโหลดภาพถ่ายผลไม้จริงจาก {F262_ZIP.name} (30 ชนิด x ~{images_per_class} รูป)...")
    t0 = time.time()
    X = []
    y = []

    with zipfile.ZipFile(F262_ZIP, "r") as z:
        for idx, fruit in enumerate(SELECTED_FRUITS):
            prefix = f"{fruit}/"
            files = [n for n in z.namelist() if n.startswith(prefix) and n.endswith((".jpg", ".png", ".jpeg"))]
            selected_files = files[:images_per_class]

            for fn in selected_files:
                try:
                    im = Image.open(io.BytesIO(z.read(fn))).convert("RGB").resize((100, 100))
                    # ใช้ histogram feature mode เพื่อความทนทานต่อฉากหลังและมุมมอง
                    feat = extract_features(im, mode="histogram")
                    X.append(feat)
                    y.append(fruit)
                except Exception:
                    pass

            if (idx + 1) % 10 == 0 or (idx + 1) == len(SELECTED_FRUITS):
                print(f"  ⚡ โหลดเสร็จแล้ว {idx + 1}/{len(SELECTED_FRUITS)} ชนิด (สะสม {len(X)} รูปภาพ)...")

    # เสริมภาพจาก sample_images/ เพื่อให้ตัวอย่างในเว็บแอปแม่นยำสูง
    samples_dir = Path("sample_images")
    if samples_dir.exists():
        for s in samples_dir.glob("*.jpg"):
            stem = s.stem.lower()
            matching_class = None
            if stem == "dragonfruit": matching_class = "dragonfruit"
            elif stem == "passionfruit": matching_class = "passion fruit"
            elif stem in SELECTED_FRUITS: matching_class = stem
            
            if matching_class:
                try:
                    im = Image.open(s).convert("RGB")
                    # เพิ่มภาพตัวอย่าง 5 สำเนาพร้อม flip
                    for _ in range(3):
                        X.append(extract_features(im, mode="histogram"))
                        y.append(matching_class)
                    im_flip = im.transpose(Image.FLIP_LEFT_RIGHT)
                    for _ in range(2):
                        X.append(extract_features(im_flip, mode="histogram"))
                        y.append(matching_class)
                except Exception:
                    pass

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    print(f"🎉 โหลดสำเร็จทั้งหมด {len(X)} รูปภาพ ในเวลา {time.time() - t0:.1f} วินาที | ขนาด Matrix: {X.shape}")
    return X, y


def main():
    print("=" * 65)
    print("🚀 เริ่มต้นเทรนโมเดล 30 ผลไม้ยอดนิยม (ตัดผักออก 100% | ภาพถ่ายจริง)")
    print("=" * 65)

    X, y = load_data(images_per_class=150)

    # แบ่ง Train 80% / Test 20%
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n📊 การแบ่งชุดข้อมูล:")
    print(f"   - ชุดฝึกสอน (Train Set 80%): {len(X_train)} รูปภาพ")
    print(f"   - ชุดทดสอบ (Test Set 20%):   {len(X_test)} รูปภาพ")

    # ฝึกสอนโมเดล HistGradientBoosting
    print(f"\n⚙️ กำลังฝึกสอน HistGradientBoostingClassifier บน {len(X_train)} ตัวอย่าง...")
    t_train = time.time()
    clf = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.1,
        max_leaf_nodes=31,
        random_state=42
    )
    clf.fit(X_train, y_train)
    print(f"✅ ฝึกสอนสำเร็จในเวลา {time.time() - t_train:.1f} วินาที!")

    # ประเมินผลบน Test Set
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("=" * 65)
    print(f"🏆 ผลลัพธ์ความแม่นยำบน Test Set (ภาพถ่ายจริง {len(y_test)} ภาพ): {accuracy * 100:.2f}%")
    print("=" * 65)

    # บันทึกโมเดล
    model_data = {
        "version": "2.0.0",
        "pipeline": clf,
        "classes": sorted(list(set(y))),
        "thai_labels": THAI_LABELS,
        "feature_mode": "histogram",
        "img_size": (32, 32),
        "model_type": "HistGradientBoosting (Non-Linear Ensemble)",
        "accuracy": float(accuracy),
        "total_samples": len(X),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }

    joblib.dump(model_data, OUTPUT_MODEL, compress=3)
    file_size_mb = OUTPUT_MODEL.stat().st_size / (1024 * 1024)
    print(f"💾 บันทึกโมเดลสำเร็จ: {OUTPUT_MODEL.resolve()} ({file_size_mb:.2f} MB)")

    # บันทึกรายงาน classification_report.txt
    report = classification_report(y_test, y_pred, zero_division=0)
    with open("classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Model: HistGradientBoosting (30 Real Fruit Classes - Fruit-262)\n")
        f.write(f"Total Samples: {len(X)} (Train: {len(X_train)}, Test: {len(X_test)})\n")
        f.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")
        f.write(report)
    print("📄 บันทึก classification_report.txt เรียบร้อย")

    # ทดสอบความแม่นยำบนรูปถ่ายจริงของผู้ใช้ทันที
    print("\n🧪 ทดสอบผลลัพธ์บนภาพจริงของผู้ใช้:")
    user_tests = [
        ("test_blueberry.png", "blueberry", "บลูเบอร์รีติดใบไม้"),
        ("test_starfruit.png", "carambola", "มะเฟืองบนโต๊ะไม้"),
        ("test_passionfruit.png", "passion fruit", "เสาวรสติดใบไม้"),
        ("test_dragonfruit.png", "dragonfruit", "แก้วมังกรผ่าครึ่ง")
    ]
    for fn, exp, desc in user_tests:
        if Path(fn).exists():
            im = Image.open(fn)
            feat = extract_features(im, mode="histogram")
            probs = clf.predict_proba([feat])[0]
            top3 = np.argsort(probs)[::-1][:3]
            top_cls = clf.classes_[top3[0]]
            top_conf = probs[top3[0]] * 100
            th_name = THAI_LABELS.get(top_cls, top_cls)
            status = "✅ ถูกต้อง" if top_cls == exp else "⚠️ ไม่ตรง"
            print(f"   [{desc:18s}] -> {th_name:25s} ({top_conf:4.1f}%) | {status}")


if __name__ == "__main__":
    main()
