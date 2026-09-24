# 🍎🍌🍊 Fruit Classifier (ระบบจำแนกชนิดผลไม้ด้วย Machine Learning)

![Version](https://img.shields.io/badge/version-1.0.3-blue.svg)
![Python](https://img.shields.io/badge/Python-3.12-brightgreen.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9-orange.svg)
![Gradio](https://img.shields.io/badge/Gradio-6.28-red.svg)

ระบบจำแนกชนิดผลไม้พัฒนาด้วย **Python**, **Scikit-Learn (SVM / Random Forest)** และ **Gradio** โดยต่อยอดแนวคิดจากโปรเจกต์ `digit_svm_app` เปลี่ยนจากการรู้จำลายมือตัวเลขมาเป็นการจำแนกภาพผลไม้จริง (แอปเปิ้ล, กล้วย, ส้ม)

---

## 📁 โครงสร้างโปรเจกต์ (Directory Structure)

```text
Fruit Classifier/
├── dataset/                    # โฟลเดอร์เก็บภาพแบ่งตามคลาส (55 รูป/คลาส)
│   ├── apple/                  # ภาพแอปเปิ้ล
│   ├── banana/                 # ภาพกล้วย
│   └── orange/                 # ภาพส้ม
├── download_images.py          # สคริปต์ดาวน์โหลดภาพจากอินเทอร์เน็ตอัตโนมัติ
├── train.py                    # สคริปต์สกัดฟีเจอร์ เทรน และประเมินผลโมเดล
├── app.py                      # หน้าเว็บ Interactive UI ด้วย Gradio
├── version.py                  # ไฟล์ระบุเวอร์ชันและ Metadata ของโครงการ
├── fruit_model.pkl             # โมเดล SVM ที่เทรนเสร็จแล้ว (พร้อมใช้งาน)
├── fruit_rf_model.pkl          # โมเดล Random Forest ทางเลือก
├── confusion_matrix.png        # กราฟ Confusion Matrix แสดงผลความแม่นยำ
├── classification_report.txt   # รายงานผล Precision, Recall, F1-Score
└── requirements.txt            # รายการไลบรารีที่จำเป็น
```

---

## ⚙️ การติดตั้ง (Setup & Installation)

เปิด Terminal หรือ PowerShell ในโฟลเดอร์นี้ แล้วติดตั้งแพ็กเกจที่ต้องใช้:

```bash
pip install -r requirements.txt
```

*(หรือหากมีหลายเวอร์ชันในเครื่อง สามารถใช้ `py -3.12 -m pip install -r requirements.txt`)*

---

## 🚀 ขั้นตอนการใช้งาน (Workflow)

### ขั้นตอนที่ 1: เตรียมชุดข้อมูล (Download Dataset)
สามารถสั่งดาวน์โหลดภาพผลไม้จากอินเทอร์เน็ตเข้าโฟลเดอร์คลาสโดยอัตโนมัติ พร้อมตรวจสอบและตัดภาพเสียออก:

```bash
python download_images.py --limit 50
```
- ค่าเริ่มต้นจะดาวน์โหลดคลาสละ 50 รูป จัดเก็บลงใน `dataset/apple`, `dataset/banana`, `dataset/orange`

---

### ขั้นตอนที่ 2: เทรนโมเดล (Train Model)
รันสคริปต์เทรนโมเดล ซึ่งจะทำการย่อภาพ สกัดฟีเจอร์ และสร้างโมเดล Machine Learning:

```bash
# เทรนด้วยโมเดล SVM แบบฟีเจอร์ผสมผสาน (Flatten + Color Histogram) แนะนำที่สุด
python train.py --feature combined --model svm
```

**ตัวเลือกเสริม (Arguments):**
- `--feature`:
  - `combined`: รวมทั้งเวกเตอร์พิกเซลและฮิสโตแกรมสี (ความแม่นยำสูงสุด > 90%)
  - `flatten`: แปลงพิกเซล $64 \times 64 \times 3$ เป็น 1D Array (ตรงตามแล็บพื้นฐาน)
  - `histogram`: สกัดเฉพาะค่าการกระจายตัวของแม่สี RGB
- `--model`:
  - `svm`: Support Vector Machine ด้วย RBF Kernel (ค่าเริ่มต้น)
  - `rf`: Random Forest Classifier

เมื่อรันเสร็จ โปรแกรมจะแสดงค่า **Accuracy**, **Classification Report** และเซฟไฟล์ `confusion_matrix.png` และ `fruit_model.pkl`

---

### ขั้นตอนที่ 3: เปิดหน้าเว็บทดสอบ (Run Gradio App)
รันเว็บแอปพลิเคชันเพื่อให้ผู้ใช้ทดสอบอัปโหลดรูปภาพ:

```bash
python app.py
```
- ระบบจะเปิดเบราว์เซอร์ให้อัตโนมัติที่ `http://127.0.0.1:7860`
- สามารถลากรูปผลไม้มาวาง หรือกดถ่ายภาพจากกล้องเว็บแคม
- มีปุ่มตัวอย่างภาพด้านล่างให้คลิกทดสอบได้ทันทีใน 1 วินาที
- ระบบจะแสดงชื่อผลไม้ภาษาไทย-อังกฤษ พร้อมแถบเปอร์เซ็นต์ความมั่นใจ (Confidence Score)
- มีแท็บ **"ประสิทธิภาพโมเดล (Model Evaluation)"** แสดง Confusion Matrix และตารางคะแนน

---

## 🧠 สรุปแนวคิดทางเทคนิค (สำหรับส่งและตอบคำถามอาจารย์)

1. **การแปลงขนาดรูปภาพ (Image Resizing $64 \times 64$):**
   - รูปภาพผลไม้ที่ดาวน์โหลดมามีขนาดกว้างคูณยาวไม่เท่ากัน โมเดล Machine Learning แบบคลาสสิกต้องการ Input Vector ขนาดคงที่ จึงจำเป็นต้อง Resize ภาพทั้งหมดให้เป็นขนาดเดียวกัน (เช่น $64 \times 64$ พิกเซล)
2. **Flattening Array:**
   - คล้ายกับแล็บตัวเลข `digits` ภาพ $64 \times 64 \times 3$ ช่องสี จะถูกคลี่ออกเป็นเวกเตอร์ขนาดยาว $64 \times 64 \times 3 = 12,288$ มิติ และ Normalize ค่าพิกเซล $[0, 255] \rightarrow [0.0, 1.0]$
3. **Color Histogram:**
   - ผลไม้แต่ละชนิดมีเอกลักษณ์ด้านสีที่เด่นชัดมาก (ส้ม = สีส้ม, กล้วย = สีเหลือง, แอปเปิ้ล = สีแดง/เขียว) การทำ Color Histogram ช่วยให้โมเดลจับคู่สีได้แม่นยำแม้ผลไม้จะหมุนอยู่คนละมุม
4. **Support Vector Machine (SVM):**
   - ทำงานโดยการค้นหา Hyperplane ที่แบ่งแยกกลุ่มข้อมูลออกจากกันให้มี Margin กว้างที่สุด
   - การใช้ **RBF Kernel** ช่วยให้โมเดลสามารถแบ่งข้อมูลที่มีความซับซ้อนและไม่เป็นเชิงเส้น (Non-linear boundary) ในภาพผลไม้ได้เป็นอย่างดี
5. **การวัดผล (Evaluation):**
   - ประเมินผลด้วยข้อมูล Test Set ที่แยกไว้ 20% ที่โมเดลไม่เคยเห็นมาก่อน
   - ดูค่า Accuracy, Precision, Recall, F1-Score และตรวจสอบผ่าน Confusion Matrix

---

## 📌 บันทึกประวัติเวอร์ชัน (Version History & Changelog)

### **v1.0.3 (2026-09-24) - Clean Minimalist UI Cleanup**
- 🧹 **UI Cleanup:**
  - ลบส่วนภาพตัวอย่าง (Examples) และแถบข้อความตัวอย่างด้านล่างออกตามความต้องการของผู้ใช้
  - ลบแถบ Footer ท้ายเว็บออกเพื่อความมินิมอล เรียบง่าย และสบายตาสูงสุด

### **v1.0.2 (2026-09-24) - Minimalist UI Redesign**
- 🎨 **Minimalist Design & UX:**
  - ปรับดีไซน์ใหม่สไตล์ Minimalist สะอาดตา จัดกึ่งกลางหน้าจอ (Max-width 900px) ไม่ยืดกว้างเกินไปบนจอใหญ่
  - เปลี่ยนจากการแสดงภาพย่อเปล่าๆ เป็น **Empty State Card** สวยงาม
  - เพิ่มระบบ **Auto-Predict** วิเคราะห์ผลทันทีที่อัปโหลดรูปภาพหรือคลิกเลือกตัวอย่าง ไม่จำเป็นต้องกดปุ่ม
  - ซ่อนภาพ 64x64 px ทางเทคนิคไว้ใน Accordion เพื่อความสบายตา
  - ปรับแต่งปุ่มและฟอนต์ให้อ่านง่าย มีระดับ และเป็นระเบียบ

### **v1.0.1 (2026-09-24) - Vercel & ASGI Support**
- 🛠️ **Deployment Compatibility:**
  - Export top-level ASGI `app` ผ่าน `gr.mount_gradio_app(FastAPI(), demo, path="/")` เพื่อแก้ปัญหา Vercel Serverless Function error: *"Found app.py but it does not export a top-level 'app'"*
  - เพิ่มไฟล์คอนฟิก `vercel.json` สำหรับการทำ Route Rewrites
  - เพิ่ม `fastapi` และ `uvicorn` ลงใน `requirements.txt`
  - อัปเดต metadata ของระบบและ UI เป็นเวอร์ชัน `v1.0.1`

### **v1.0.0 (2026-09-24) - Initial Release**
- 🎉 เปิดตัวโครงงาน Fruit Classifier เวอร์ชันแรกอย่างเป็นทางการ
- 📥 **Data Pipeline:**
  - สคริปต์ `download_images.py` ดาวน์โหลดรูปภาพผลไม้อัตโนมัติจาก Bing Search
  - ระบบตรวจสอบความถูกต้องของไฟล์ภาพ (Corrupted Image Validation)
  - ดาวน์โหลดข้อมูลตัวอย่าง 165 รูป (Apple: 55, Banana: 55, Orange: 55)
- 🔬 **Feature Extraction & Modeling:**
  - รองรับการสกัดฟีเจอร์ 3 โหมด: `flatten` (เวกเตอร์พิกเซลย่อขนาด 64x64), `histogram` (การกระจายตัวของสี RGB), และ `combined` (รวมทั้งสองฟีเจอร์)
  - พัฒนาโมเดล **SVM (RBF Kernel)** พร้อม Probability Calibration สำหรับส่งเปอร์เซ็นต์ความมั่นใจ
  - พัฒนาโมเดลทางเลือก **Random Forest Classifier** เพื่อการเปรียบเทียบ
  - ผลประเมินโมเดล: Accuracy บน Test Set สูงถึง **87.88%** (SVM) และ **90.91%** (Random Forest)
- 🌐 **Web Interface (Gradio):**
  - หน้าเว็บ Interactive รองรับการ Drag & Drop ไฟล์ภาพ, ป้อนภาพผ่านเว็บแคม และคลิกเลือกภาพตัวอย่าง
  - แสดงผลลัพธ์เป็นชื่อผลไม้พร้อม Emoji และกราฟแสดงเปอร์เซ็นต์ความมั่นใจ (Confidence Breakdown)
  - แท็บรายงานประสิทธิภาพโมเดล แสดงภาพ Confusion Matrix และตาราง Classification Report ในตัว

