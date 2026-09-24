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
<div style="background: var(--card-bg, #ffffff); border: 1px dashed var(--card-border, #cbd5e1); border-radius: 12px; padding: 32px 16px; text-align: center; color: var(--text-body, #334155);">
    <div style="font-size: 2.2rem; margin-bottom: 8px; opacity: 0.9;">🍎🍌🍊🍇🍉🍋🍓🍍🥭🍈🥥🍐🍒🌰🥝🥑</div>
    <div style="font-size: 1rem; font-weight: 700; color: var(--text-title, #0f172a); margin-bottom: 6px;">พร้อมวิเคราะห์ภาพผลไม้และพืชผล 300 ชนิด</div>
    <div style="font-size: 0.85rem; color: var(--text-muted, #64748b);">อัปโหลดรูปภาพด้านซ้าย หรือคลิกเลือกภาพตัวอย่างด้านล่างเพื่อเริ่มการวิเคราะห์</div>
</div>
"""


def predict_fruit(image: Image.Image):
    """ฟังก์ชันทำนายผลสำหรับ Gradio (พร้อมระบบย่อภาพขนาดใหญ่จากกล้องมือถืออัตโนมัติ)"""
    if image is None:
        return None, EMPTY_STATE_HTML, None

    # ย่อภาพความละเอียดสูงจากกล้องมือถือทันที (แก้ปัญหาค้าง / โหลดนาน / RAM เต็มบน Render)
    try:
        image = image.copy()
        image.thumbnail((640, 640), Image.Resampling.LANCZOS)
    except Exception:
        pass

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

    # ภาพย่อ 32x32 สำหรับเทคนิคการประมวลผล
    thumbnail_preview = image.convert("RGB").resize(img_size)

    # สร้างการ์ดผลลัพธ์แบบ Dynamic Theme HTML (รองรับทั้ง Light และ Dark mode)
    top_label_th = thai_labels.get(pred_class, pred_class.capitalize())
    model_type = model_data.get("model_type", "Linear").upper()

    summary_html = f"""
    <div style="background: var(--card-bg, #ffffff); border: 1px solid var(--card-border, #e2e8f0); border-radius: 14px; padding: 18px 20px; box-shadow: 0 2px 8px -2px rgba(0,0,0,0.06); margin-bottom: 14px;">
        <div style="font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-muted, #64748b); margin-bottom: 4px;">
            ผลลัพธ์การจำแนก (Prediction)
        </div>
        <div style="font-size: 1.65rem; font-weight: 800; color: var(--text-title, #0f172a); margin-bottom: 10px;">
            {top_label_th}
        </div>
        <div style="display: flex; gap: 8px; flex-wrap: wrap;">
            <span style="background: var(--highlight-bg, #ecfdf5); color: var(--highlight-text, #047857); font-size: 0.82rem; font-weight: 700; padding: 4px 12px; border-radius: 999px; border: 1px solid rgba(4, 120, 87, 0.2);">
                ความมั่นใจ {pred_confidence:.1f}%
            </span>
            <span style="background: var(--badge-bg, #f1f5f9); color: var(--badge-text, #334155); font-size: 0.82rem; font-weight: 600; padding: 4px 12px; border-radius: 999px; border: 1px solid var(--card-border, #e2e8f0);">
                โมเดล {model_type}
            </span>
            <span style="background: var(--badge-bg, #f1f5f9); color: var(--badge-text, #334155); font-size: 0.82rem; font-weight: 600; padding: 4px 12px; border-radius: 999px; border: 1px solid var(--card-border, #e2e8f0);">
                {feature_mode} ({img_size[0]}×{img_size[1]} px)
            </span>
        </div>
    </div>
    """
    return confidence_dict, summary_html, thumbnail_preview




# กำหนดสไตล์ Minimalist CSS รองรับทั้ง Light และ Dark Mode
MINIMAL_CSS = """
<style>
/* CSS Variables สำหรับรองรับทั้ง Light Mode และ Dark Mode (แก้ปัญหาตัวอักษรกลืนกับพื้นหลังในมือถือ 100%) */
:root {
    --app-bg: #f8fafc;
    --card-bg: #ffffff;
    --card-border: #e2e8f0;
    --text-title: #0f172a;
    --text-body: #1e293b;
    --text-muted: #475569;
    --badge-bg: #f1f5f9;
    --badge-text: #1e293b;
    --highlight-bg: #ecfdf5;
    --highlight-text: #047857;
}

@media (prefers-color-scheme: dark) {
    :root {
        --app-bg: #0b0f19;
        --card-bg: #1e293b;
        --card-border: #334155;
        --text-title: #f8fafc;
        --text-body: #e2e8f0;
        --text-muted: #94a3b8;
        --badge-bg: #334155;
        --badge-text: #f1f5f9;
        --highlight-bg: #064e3b;
        --highlight-text: #6ee7b7;
    }
}

.dark {
    --app-bg: #0b0f19;
    --card-bg: #1e293b;
    --card-border: #334155;
    --text-title: #f8fafc;
    --text-body: #e2e8f0;
    --text-muted: #94a3b8;
    --badge-bg: #334155;
    --badge-text: #f1f5f9;
    --highlight-bg: #064e3b;
    --highlight-text: #6ee7b7;
}

/* ตั้งค่าขนาด Container ให้กึ่งกลางและมี padding พอดี */
.gradio-container {
    max-width: 900px !important;
    margin: 0 auto !important;
    padding: 16px 12px !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Prompt', sans-serif !important;
    background-color: var(--app-bg) !important;
}

/* บล็อกการ์ดพื้นฐาน */
.block {
    border-radius: 14px !important;
    border: 1px solid var(--card-border) !important;
    background: var(--card-bg) !important;
}

/* ป้องกันตัวอักษรสีขาวกลืนกับพื้นหลัง */
.block p, .block span, .block label, .block div {
    color: var(--text-body);
}

/* ปรับแต่งปุ่มวิเคราะห์ภาพ */
button.primary {
    background-color: #0f172a !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.dark button.primary {
    background-color: #2563eb !important;
    color: #ffffff !important;
}
button.primary:hover {
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
    transform: translateY(-1px) !important;
}

/* ซ่อนแถบฟุตเตอร์ */
footer {
    display: none !important;
}

/* ปรับแต่งตัวอย่างภาพ */
.gallery {
    border: none !important;
    background: transparent !important;
}
.gallery button {
    border-radius: 10px !important;
    border: 1px solid var(--card-border) !important;
    overflow: hidden !important;
    transition: all 0.2s ease !important;
}
</style>
"""

# สร้างหน้าตาเว็บด้วย Gradio Blocks
with gr.Blocks(title=f"Fruit Classifier v{__version__}") as demo:
    # ฝัง Minimal CSS
    gr.HTML(MINIMAL_CSS)

    # Minimal Header
    gr.HTML(f"""
    <div style="text-align: center; margin-bottom: 1.6rem; margin-top: 0.5rem;">
        <div style="display: inline-flex; align-items: center; gap: 8px; background: var(--badge-bg); padding: 5px 16px; border-radius: 999px; font-size: 0.82rem; color: var(--text-title); font-weight: 600; margin-bottom: 10px; border: 1px solid var(--card-border);">
            <span>🍎 Fruit Classifier</span>
            <span style="opacity: 0.4;">•</span>
            <span>v{__version__}</span>
        </div>
        <h1 style="font-size: 1.85rem; font-weight: 800; color: var(--text-title); margin: 0 0 6px 0; letter-spacing: -0.02em;">
            ระบบจำแนกชนิดผลไม้
        </h1>
        <p style="color: var(--text-muted); font-size: 0.92rem; margin: 0; font-weight: 500;">
            ระบบจำแนกผลไม้และพืชผลครอบคลุม 300 ชนิด (ทุเรียน • มังคุด • เงาะ • ขนุน • ลำไย • กระท้อน • น้อยหน่า • แก้วมังกร ฯลฯ ครบทุกสายพันธุ์)
        </p>
    </div>
    """)

    with gr.Tabs():
        with gr.TabItem("✨ จำแนกภาพผลไม้ (Classifier)"):
            # คำอธิบายวิธีใช้งาน (User Guide Banner)
            gr.HTML("""
            <div style="background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 12px; padding: 14px 18px; margin-bottom: 16px; font-size: 0.9rem; color: var(--text-body); display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
                <div>💡 <b style="color: var(--text-title);">วิธีใช้งาน:</b> อัปโหลดรูปภาพผลไม้ หรือคลิกเลือกภาพตัวอย่างด้านล่างเพื่อทดสอบจำแนกผลไม้ทันที (Auto-Predict)</div>
                <div style="font-size: 0.8rem; background: var(--highlight-bg); color: var(--highlight-text); font-weight: 700; padding: 4px 12px; border-radius: 999px; border: 1px solid rgba(4, 120, 87, 0.2);">
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
                        "<div style='font-size: 0.88rem; color: var(--text-body, #334155); margin-bottom: 8px;'>"
                        "คลิกเลือกภาพผลไม้ตัวอย่างด้านล่างเพื่อทดสอบระบบได้ทันที: "
                        "<b style='color: var(--text-title, #0f172a);'>ทุเรียน • มังคุด • เงาะ • ขนุน • ลำไย • กระท้อน • น้อยหน่า • แก้วมังกร • มะม่วง • มะพร้าว • กล้วย • แอปเปิ้ล</b>"
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
                outputs=[output_label, output_summary, output_thumb],
                show_progress="minimal"
            )
            input_image.change(
                fn=predict_fruit,
                inputs=input_image,
                outputs=[output_label, output_summary, output_thumb],
                show_progress="minimal"
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


# ตั้งค่า Queue ให้รองรับหลายคำขอพร้อมกันโดยไม่ค้าง
demo.queue(default_concurrency_limit=10)

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
