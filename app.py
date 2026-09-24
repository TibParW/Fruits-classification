"""
app.py
เว็บแอปพลิเคชัน Gradio สำหรับจำแนกชนิดผลไม้ (Fruit Classifier)
ดีไซน์แบบ Minimalist สไตล์โมเดิร์น สะอาดตา พร้อมระบบ Auto-Predict
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

EMPTY_STATE_HTML = """
<div style="background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 12px; padding: 36px 16px; text-align: center; color: #64748b;">
    <div style="font-size: 2.2rem; margin-bottom: 6px; opacity: 0.85;">🍎🍌🍊🍇🍉🍋🍓🍍🥭👑🔴🐲🍈🥥🍐🍒🌰🥝🥑</div>
    <div style="font-size: 0.95rem; font-weight: 600; color: #334155;">พร้อมวิเคราะห์ภาพผลไม้และพืชผล 300 ชนิด</div>
    <div style="font-size: 0.82rem; color: #94a3b8; margin-top: 4px;">อัปโหลดรูปภาพทางซ้ายเพื่อเริ่มการวิเคราะห์อัตโนมัติ</div>
</div>
"""


def predict_fruit(image: Image.Image):
    """ฟังก์ชันทำนายผลสำหรับ Gradio"""
    if image is None:
        return None, EMPTY_STATE_HTML, None

    if model_data is None:
        error_html = """
        <div style="background: #fef2f2; border: 1px solid #fecaca; border-radius: 12px; padding: 16px; color: #991b1b; font-size: 0.9rem;">
            ❌ ไม่พบไฟล์โมเดล <code>fruit_model.pkl</code> กรุณารัน train.py ก่อน
        </div>
        """
        return None, error_html, None

    pipeline = model_data["pipeline"]
    classes = model_data["classes"]
    thai_labels = model_data.get("thai_labels", {})
    feature_mode = model_data.get("feature_mode", "combined")
    img_size = model_data.get("img_size", (32, 32))

    # สกัดฟีเจอร์จากรูปที่ผู้ใช้อัปโหลด
    try:
        features = extract_features(image, mode=feature_mode, img_size=img_size)
    except Exception as e:
        error_html = f"""
        <div style="background: #fef2f2; border: 1px solid #fecaca; border-radius: 12px; padding: 16px; color: #991b1b; font-size: 0.9rem;">
            เกิดข้อผิดพลาดในการแปลงรูปภาพ: {e}
        </div>
        """
        return None, error_html, None

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

    # ภาพย่อ 64x64 สำหรับเทคนิคการประมวลผล
    thumbnail_preview = image.convert("RGB").resize(img_size)

    # สร้างการ์ดผลลัพธ์แบบ Minimal HTML
    top_label_th = thai_labels.get(pred_class, pred_class.capitalize())
    model_type = model_data.get("model_type", "SVM").upper()

    summary_html = f"""
    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 18px 20px; box-shadow: 0 2px 8px -2px rgba(0,0,0,0.04); margin-bottom: 14px;">
        <div style="font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: #64748b; margin-bottom: 4px;">
            ผลลัพธ์การจำแนก (Prediction)
        </div>
        <div style="font-size: 1.6rem; font-weight: 700; color: #0f172a; margin-bottom: 8px;">
            {top_label_th}
        </div>
        <div style="display: flex; gap: 8px; flex-wrap: wrap;">
            <span style="background: #ecfdf5; color: #047857; font-size: 0.8rem; font-weight: 600; padding: 3px 10px; border-radius: 999px;">
                ความมั่นใจ {pred_confidence:.1f}%
            </span>
            <span style="background: #f1f5f9; color: #475569; font-size: 0.8rem; font-weight: 500; padding: 3px 10px; border-radius: 999px;">
                โมเดล {model_type}
            </span>
            <span style="background: #f1f5f9; color: #475569; font-size: 0.8rem; font-weight: 500; padding: 3px 10px; border-radius: 999px;">
                {feature_mode} ({img_size[0]}×{img_size[1]} px)
            </span>
        </div>
    </div>
    """
    return confidence_dict, summary_html, thumbnail_preview




# กำหนดสไตล์ Minimalist CSS
MINIMAL_CSS = """
<style>
/* ตั้งค่าขนาด Container ให้กึ่งกลางและไม่ยืดกว้างเกินไป */
.gradio-container {
    max-width: 900px !important;
    margin: 0 auto !important;
    padding: 24px 16px !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Prompt', sans-serif !important;
    background-color: #fafbfc !important;
}

/* ปรับแต่งบล็อกการ์ดให้เรียบเนียน */
.block {
    border-radius: 14px !important;
    border: 1px solid #e2e8f0 !important;
    background: #ffffff !important;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.02) !important;
}

/* ปุ่มสีเข้มสไตล์มินิมอล */
button.primary {
    background-color: #0f172a !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 0.92rem !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}
button.primary:hover {
    background-color: #1e293b !important;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.15) !important;
}

/* ซ่อนแถบฟุตเตอร์รกๆ */
footer {
    display: none !important;
}

/* ตกแต่งส่วน Examples ให้เรียบเนียนสไตล์ Minimal */
.gallery {
    border: none !important;
    background: transparent !important;
}
.gallery button {
    border-radius: 10px !important;
    border: 1px solid #e2e8f0 !important;
    overflow: hidden !important;
    transition: all 0.2s ease !important;
}
.gallery button:hover {
    border-color: #0f172a !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08) !important;
}
</style>
"""

# สร้างหน้าตาเว็บด้วย Gradio Blocks
with gr.Blocks(title=f"Fruit Classifier v{__version__}") as demo:
    # ฝัง Minimal CSS
    gr.HTML(MINIMAL_CSS)

    # Minimal Header
    gr.HTML(f"""
    <div style="text-align: center; margin-bottom: 1.8rem; margin-top: 0.5rem;">
        <div style="display: inline-flex; align-items: center; gap: 8px; background: #f1f5f9; padding: 4px 14px; border-radius: 999px; font-size: 0.8rem; color: #475569; font-weight: 500; margin-bottom: 10px;">
            <span>🍎 Fruit Classifier</span>
            <span style="opacity: 0.4;">•</span>
            <span>v{__version__}</span>
        </div>
        <h1 style="font-size: 1.85rem; font-weight: 700; color: #0f172a; margin: 0 0 6px 0; letter-spacing: -0.02em;">
            ระบบจำแนกชนิดผลไม้
        </h1>
        <p style="color: #64748b; font-size: 0.92rem; margin: 0; font-weight: 400;">
            ระบบจำแนกผลไม้และพืชผลครอบคลุม 300 ชนิด (ทุเรียน • มังคุด • เงาะ • ขนุน • ลำไย • กระท้อน • น้อยหน่า • แก้วมังกร ฯลฯ ครบทุกสายพันธุ์)
        </p>
    </div>
    """)

    with gr.Tabs():
        with gr.TabItem("✨ จำแนกภาพผลไม้ (Classifier)"):
            # คำอธิบายวิธีใช้งาน (User Guide Banner)
            gr.HTML("""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 12px 18px; margin-bottom: 16px; font-size: 0.88rem; color: #475569; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                <div>💡 <b>วิธีใช้งาน:</b> อัปโหลดรูปภาพผลไม้ หรือคลิกเลือกภาพตัวอย่างด้านล่างเพื่อทดสอบจำแนกผลไม้ทันที (Auto-Predict)</div>
                <div style="font-size: 0.78rem; background: #ecfdf5; color: #047857; font-weight: 600; padding: 3px 10px; border-radius: 999px;">
                    ✨ 300 ชนิดผลไม้ & พืชผล
                </div>
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(
                        type="pil",
                        label="อัปโหลดภาพผลไม้ (Drop or Select Image)",
                        sources=["upload", "webcam", "clipboard"],
                        height=260
                    )
                    predict_btn = gr.Button("🔍 วิเคราะห์ภาพ", variant="primary")

                with gr.Column(scale=1):
                    output_summary = gr.HTML(value=EMPTY_STATE_HTML)
                    output_label = gr.Label(
                        label="ระดับความมั่นใจ (Confidence Breakdown)",
                        num_top_classes=5
                    )
                    with gr.Accordion("🔍 ภาพที่โมเดลประมวลผล (32×32 px)", open=False):
                        output_thumb = gr.Image(
                            label="Resized Thumbnail",
                            height=120,
                            width=120,
                            interactive=False
                        )

            # ภาพตัวอย่างสำหรับให้อาจารย์และผู้ใช้ทดสอบทันที (Sample Images)
            sample_files = [
                ["sample_images/durian.jpg"],
                ["sample_images/mangosteen.jpg"],
                ["sample_images/rambutan.jpg"],
                ["sample_images/jackfruit.jpg"],
                ["sample_images/longan.jpg"],
                ["sample_images/santol.jpg"],
                ["sample_images/custard_apple.jpg"],
                ["sample_images/dragonfruit.jpg"],
                ["sample_images/mango.jpg"],
                ["sample_images/coconut.jpg"],
                ["sample_images/banana.jpg"],
                ["sample_images/apple.jpg"],
            ]
            valid_samples = [s for s in sample_files if Path(s[0]).exists()]
            if valid_samples:
                with gr.Accordion("📁 ภาพตัวอย่างสำหรับทดสอบ (Click Sample Image to Test)", open=True):
                    gr.Markdown(
                        "<div style='font-size: 0.85rem; color: #64748b; margin-bottom: 8px;'>"
                        "คลิกเลือกภาพผลไม้ตัวอย่างด้านล่างเพื่อทดสอบระบบได้ทันที: "
                        "<b>ทุเรียน 👑 • มังคุด 👑 • เงาะ 🔴 • ขนุน 🍈 • ลำไย 🌰 • กระท้อน 🟡 • น้อยหน่า 🍈 • แก้วมังกร 🐲 • มะม่วง 🥭 • มะพร้าว 🥥 • กล้วย 🍌 • แอปเปิ้ล 🍎</b>"
                        "</div>"
                    )
                    gr.Examples(
                        examples=valid_samples,
                        inputs=input_image,
                        outputs=[output_label, output_summary, output_thumb],
                        fn=predict_fruit,
                        cache_examples=False,
                        label="ภาพตัวอย่างผลไม้ (Sample Images)"
                    )

            # รองรับทั้งคลิกปุ่ม และ Auto-predict ทันทีที่อัปโหลดรูป
            predict_btn.click(
                fn=predict_fruit,
                inputs=input_image,
                outputs=[output_label, output_summary, output_thumb]
            )
            input_image.change(
                fn=predict_fruit,
                inputs=input_image,
                outputs=[output_label, output_summary, output_thumb]
            )

        with gr.TabItem("📊 ประสิทธิภาพโมเดล (Metrics)"):
            gr.Markdown("#### 📈 ผลการประเมินโมเดลบน Test Set (Confusion Matrix & Classification Report)")
            with gr.Row():
                with gr.Column():
                    if CM_FILE.exists():
                        gr.Image(value=str(CM_FILE), label="Confusion Matrix", interactive=False)
                    else:
                        gr.Markdown("*ยังไม่มีภาพ Confusion Matrix*")

                with gr.Column():
                    report_text = ""
                    if REPORT_FILE.exists():
                        try:
                            report_text = REPORT_FILE.read_text(encoding="utf-8")
                        except Exception:
                            report_text = "เปิดไฟล์รายงานไม่สำเร็จ"
                    else:
                        report_text = "ยังไม่มีไฟล์รายงาน"

                    gr.Textbox(
                        value=report_text,
                        label="Classification Report",
                        lines=14,
                        interactive=False
                    )


# Export top-level 'app' สำหรับ ASGI / Render / Vercel
from fastapi import FastAPI
app = gr.mount_gradio_app(FastAPI(), demo, path="/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="รันเว็บแอปจำแนกชนิดผลไม้ด้วย Gradio")
    parser.add_argument("--share", action="store_true", help="สร้าง Public URL ผ่าน gradio.live เพื่อแชร์ออนไลน์")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)), help="พอร์ตที่ใช้รันเซิร์ฟเวอร์")
    args = parser.parse_args()

    print(f"\n กำลังเริ่มรัน Gradio Web Server บนพอร์ต {args.port} (share={args.share})...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
        inbrowser=not args.share
    )
