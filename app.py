"""
app.py
เว็บแอปพลิเคชัน Gradio สำหรับจำแนกชนิดผลไม้ (Fruit Classifier)
รองรับการอัปโหลดภาพ หรือถ่ายภาพจากกล้องเว็บแคม
แสดงผลชื่อชนิดผลไม้ (ภาษาไทย + อังกฤษ) พร้อมเปอร์เซ็นต์ความมั่นใจ (Confidence Score)
"""

import os
import sys
from pathlib import Path
import joblib
import numpy as np
from PIL import Image
import gradio as gr

# ป้องกันปัญหา Unicode ใน Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# นำเข้าฟังก์ชันดึง Feature และข้อมูลเวอร์ชัน
from train import extract_features
from version import __version__, __release_date__

MODEL_FILE = Path("fruit_model.pkl")
CM_FILE = Path("confusion_matrix.png")
REPORT_FILE = Path("classification_report.txt")

# โหลดโมเดล
model_data = None
if MODEL_FILE.exists():
    try:
        model_data = joblib.load(MODEL_FILE)
        print(f" โหลดโมเดลสำเร็จ: {MODEL_FILE} (Accuracy: {model_data.get('accuracy', 0)*100:.1f}%)")
    except Exception as e:
        print(f" เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")


def predict_fruit(image: Image.Image):
    """ฟังก์ชันทำนายผลสำหรับ Gradio"""
    if image is None:
        return None, None, "กรุณาอัปโหลดรูปภาพผลไม้ก่อนครับ"

    if model_data is None:
        return None, None, "❌ ยังไม่พบไฟล์โมเดล fruit_model.pkl กรุณารัน train.py ก่อนเพื่อเทรนโมเดล"

    pipeline = model_data["pipeline"]
    classes = model_data["classes"]
    thai_labels = model_data.get("thai_labels", {})
    feature_mode = model_data.get("feature_mode", "combined")
    img_size = model_data.get("img_size", (64, 64))

    # สกัดฟีเจอร์จากรูปที่ผู้ใช้อัปโหลด
    try:
        features = extract_features(image, mode=feature_mode, img_size=img_size)
    except Exception as e:
        return None, None, f"เกิดข้อผิดพลาดในการแปลงรูปภาพ: {e}"

    # คำนวณความน่าจะเป็นของแต่ละคลาส
    probabilities = pipeline.predict_proba([features])[0]
    pred_idx = np.argmax(probabilities)
    pred_class = classes[pred_idx]
    pred_confidence = probabilities[pred_idx] * 100

    # จัดรูปแบบสำหรับ gr.Label
    confidence_dict = {}
    for idx, cls_name in enumerate(classes):
        display_name = thai_labels.get(cls_name, cls_name.capitalize())
        confidence_dict[display_name] = float(probabilities[idx])

    # ภาพย่อ 64x64 สำหรับแสดงว่าโมเดลมองเห็นภาพอย่างไร
    thumbnail_preview = image.convert("RGB").resize(img_size)

    # สร้างข้อความสรุปผลลัพธ์
    top_label_th = thai_labels.get(pred_class, pred_class.capitalize())
    summary_markdown = f"""
### 🎯 ผลการวิเคราะห์: **{top_label_th}**
- **ความมั่นใจของโมเดล (Confidence):** `{pred_confidence:.2f}%`
- **โครงสร้างโมเดล:** `{model_data.get('model_type', 'SVM').upper()}`
- **วิธีการสกัดฟีเจอร์:** `{feature_mode}` (ขนาดภาพ {img_size[0]}×{img_size[1]} px)
"""
    return confidence_dict, thumbnail_preview, summary_markdown


def get_examples():
    """รวบรวมตัวอย่างรูปภาพจากโฟลเดอร์ dataset สำหรับปุ่มคลิกทดสอบด่วน"""
    examples = []
    dataset_dir = Path("dataset")
    if dataset_dir.exists():
        for class_dir in sorted(dataset_dir.iterdir()):
            if class_dir.is_dir() and not class_dir.name.startswith("_"):
                # สุ่มหรือเลือกรูปแรกๆ ของแต่ละคลาส
                images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png"))
                if images:
                    examples.append(str(images[0]))
    return examples


# สร้างหน้าตาเว็บด้วย Gradio Blocks
with gr.Blocks(title=f"ระบบจำแนกชนิดผลไม้ v{__version__} (Fruit Classifier)") as demo:
    gr.Markdown(f"""
    # 🍎🍌🍊 Fruit Classifier `v{__version__}`
    ### ระบบจำแนกชนิดผลไม้ด้วย **SVM & Color Feature Extraction** (อัปเดต: {__release_date__})
    อัปโหลดรูปภาพผลไม้ (ส้ม, กล้วย, แอปเปิ้ล) หรือถ่ายรูปจากกล้อง เพื่อให้โมเดลทำนายพร้อมคำนวณความมั่นใจ (%)
    """)

    with gr.Tabs():
        with gr.TabItem("🔮 ทำนายผล (Predict)"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(
                        type="pil",
                        label="📷 อัปโหลดรูปภาพผลไม้ (Upload Fruit Image)",
                        sources=["upload", "webcam", "clipboard"]
                    )
                    predict_btn = gr.Button("🔍 เริ่มทำนายชนิดผลไม้", variant="primary", size="lg")

                with gr.Column(scale=1):
                    output_label = gr.Label(
                        label="📊 ผลการทำนายและระดับความมั่นใจ (Confidence Breakdown)",
                        num_top_classes=3
                    )
                    output_summary = gr.Markdown("กรุณาเลือกหรืออัปโหลดภาพผลไม้เพื่อดูผลลัพธ์")
                    output_thumb = gr.Image(
                        label="🖼️ ภาพที่ถูก Resize (64x64 px) ที่โมเดลใช้ประมวลผล",
                        height=160,
                        width=160
                    )

            # ใส่ตัวอย่างรูปภาพ
            example_list = get_examples()
            if example_list:
                gr.Markdown("### 💡 คลิกรูปตัวอย่างด้านล่างเพื่อทดสอบได้ทันที:")
                gr.Examples(
                    examples=example_list,
                    inputs=input_image,
                    label="รูปภาพตัวอย่างใน Dataset"
                )

            predict_btn.click(
                fn=predict_fruit,
                inputs=input_image,
                outputs=[output_label, output_thumb, output_summary]
            )

        with gr.TabItem("📊 ประสิทธิภาพโมเดล (Model Evaluation)"):
            gr.Markdown("### 📈 Confusion Matrix & Classification Report")
            with gr.Row():
                with gr.Column():
                    if CM_FILE.exists():
                        gr.Image(value=str(CM_FILE), label="Confusion Matrix")
                    else:
                        gr.Markdown("*ยังไม่มีภาพ Confusion Matrix (จะสร้างอัตโนมัติเมื่อรัน train.py)*")

                with gr.Column():
                    report_text = ""
                    if REPORT_FILE.exists():
                        try:
                            report_text = REPORT_FILE.read_text(encoding="utf-8")
                        except Exception:
                            report_text = "เปิดไฟล์รายงานไม่สำเร็จ"
                    else:
                        report_text = "ยังไม่มีไฟล์รายงาน (จะสร้างอัตโนมัติเมื่อรัน train.py)"

                    gr.Textbox(
                        value=report_text,
                        label="Classification Report (Precision, Recall, F1-Score, Accuracy)",
                        lines=16,
                        interactive=False
                    )

    gr.Markdown(f"""
    ---
    **Fruit Classifier `v{__version__}`** | พัฒนาโดย TibParW  
    - ภาพที่มีผลไม้อยู่ตรงกลาง ชัดเจน ไม่มีพื้นหลังรบกวนมากเกินไป จะให้ผลลัพธ์ที่แม่นยำที่สุด
    - สกัดทั้ง Flatten Pixel Array (64x64) และ Color Histogram RGB ตามแนวคิดเดียวกับแล็บ `digit_svm_app`
    """)


if __name__ == "__main__":
    print("\n กำลังเริ่มรัน Gradio Web Server...")
    demo.launch(theme=gr.themes.Soft(), inbrowser=True)
