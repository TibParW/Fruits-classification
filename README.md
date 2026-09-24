# 🍎 โครงการพัฒนาระบบจำแนกชนิดผลไม้ (Fruit Classifier)
## Machine Learning Image Classification & Gradio Web Application (16 Pure Fruit Classes)

![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.12-brightgreen.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9-orange.svg)
![Gradio](https://img.shields.io/badge/Gradio-6.28-red.svg)
![Classes](https://img.shields.io/badge/Classes-16%20Pure%20Fruits-success.svg)
![Model](https://img.shields.io/badge/Model-HistGradientBoosting-blueviolet.svg)
![Samples](https://img.shields.io/badge/Samples-1%2C440%20Real%20Photos-blue.svg)
![Deployment](https://img.shields.io/badge/Deployment-Render%20Live-brightgreen.svg)

---

## 👥 ผู้จัดทำโครงการ (Authors)

> **หมายเหตุ:** เรียงลำดับตามรหัสนิสิตจากน้อยไปมากตามเกณฑ์ข้อกำหนดของรายวิชา

| ลำดับ | รหัสนิสิต | ชื่อ - นามสกุล | สาขาวิชา / ภาควิชา |
| :---: | :---: | :--- | :--- |
| 1 | `[รหัสนิสิต 10 หลัก]` | [นาย / นางสาว ชื่อ - นามสกุล] | ภาควิชาวิศวกรรมคอมพิวเตอร์ / วิทยาการคอมพิวเตอร์ |
| 2 | `[รหัสนิสิต 10 หลัก]` | [นาย / นางสาว ชื่อ - นามสกุล] | ภาควิชาวิศวกรรมคอมพิวเตอร์ / วิทยาการคอมพิวเตอร์ *(หากมี)* |

---

## 🌐 ลิงก์ระบบและเว็บแอปพลิเคชัน (Live Application & Repository URLs)

* **🌐 Web Application (Render Cloud):** [https://fruits-classification-j3ss.onrender.com](https://fruits-classification-j3ss.onrender.com)  
  *(สามารถเปิดทดลองใช้งานผ่านอินเทอร์เน็ตได้ทันที 24/7 โดยไม่ต้องเปิดโปรแกรมบนเครื่องของนักศึกษา)*
* **🐙 GitHub Repository:** [https://github.com/TibParW/Fruits-classification](https://github.com/TibParW/Fruits-classification)
* **📈 Regression App:** *(ส่วนของ Regression Model เป็นโมเดลทำนายค่าต่อเนื่องที่พัฒนาในพาร์ทคู่ขนาน สามารถศึกษาและทดสอบการรันผ่านซอร์สโค้ดในโครงการ)*

---

## 💡 1. แนวคิด ปัญหาที่ต้องการแก้ และประโยชน์ของระบบ (Problem & Significance)

### 1.1 ปัญหาที่ต้องการแก้ (Problem Statement)
การระบุและจำแนกสายพันธุ์ผลไม้และพืชผลทางการเกษตรมีความสำคัญอย่างยิ่งต่อระบบห่วงโซ่อุปทาน (Supply Chain) การคัดเกรดผลผลิตอัตโนมัติ (Automated Agricultural Sorting) และระบบคิดเงินอัตโนมัติในซูเปอร์มาร์เก็ต (Smart Retail Self-Checkout)

อย่างไรก็ดี ในการจำแนกภาพผลไม้ด้วยแบบจำลอง Machine Learning แบบดั้งเดิม หากใช้ข้อมูลพิกเซลสีดิบ (Raw Pixels) เพียงอย่างเดียว แบบจำลองจะไม่ทนทานต่อการหมุนและการเปลี่ยนมุมมอง การออกแบบกระบวนการสกัดคุณลักษณะ (Feature Extraction) ที่ผสานทั้งค่าสถิติสี (Color Histograms) โครงสร้างเชิงพื้นที่ (Spatial Grid Layout) และผิวสัมผัส (Texture) จึงเป็นหัวใจสำคัญในการแก้ปัญหานี้

### 1.2 ประโยชน์ของระบบ (Benefits)
1. **ตัดผักและสิ่งรบกวนออก 100%:** คัดสรรเฉพาะ **16 ผลไม้ยอดนิยมและผลไม้ไทยเขตร้อนแท้** (ทุเรียน มังคุด เงาะ มะม่วง มะพร้าว กล้วย แอปเปิ้ล สตรอว์เบอร์รี แตงโม เสาวรส สับปะรด แก้วมังกร ขนุน น้อยหน่า กระท้อน ลำไย)
2. **การสกัดคุณลักษณะหลายมิติ (Feature Engineering 208 มิติ):** ผสานฮิสโตแกรมสี RGB (48 มิติ), การกระจายตัวของสี HSV (56 มิติ), ตารางเชิงพื้นที่ 4x4 (96 มิติ) และผิวสัมผัส Texture (8 มิติ) ด้วย Scikit-Learn และ NumPy
3. **เบาและรวดเร็วสูง (Pure Scikit-Learn):** ขนาดโมเดลเพียง **1.73 MB** รันบน CPU ด้วยความเร็วเพียง 10 ms ต่อภาพ และใช้ RAM ต่ำกว่า 60 MB ปลอดภัย ไม่เสี่ยงโดนตัด RAM บน Render Free Tier

---

## 📦 2. แหล่งที่มา จำนวนข้อมูล และการแบ่งชุดข้อมูล (Dataset & Splitting)

### 2.1 แหล่งที่มาของข้อมูล (Data Sources)
* **Fruit-262 Dataset (Kaggle):** ชุดข้อมูลภาพถ่ายผลไม้จริงจากสภาพแวดล้อมธรรมชาติทั่วโลก (คัดเลือก 16 ชนิดผลไม้แท้ x 80 รูปภาพ)
* **Sample Images & Augmentation:** ภาพตัวอย่างทดสอบพร้อมเทคนิค Data Augmentation (Horizontal Flip)

### 2.2 จำนวนข้อมูลและการแบ่งชุดฝึก/ชุดทดสอบ (Dataset Distribution)
* **จำนวนคลาสทั้งหมด:** **16 คลาสผลไม้ยอดนิยมแท้ (Zero Vegetables)**
* **จำนวนภาพรวมทั้งหมด:** **1,440 รูปภาพ**
* **การแบ่งชุดข้อมูล (Data Splitting):** ใช้วิธี **Stratified Train-Test Split (80:20)**:
  * **ชุดฝึกสอน (Train Set 80%):** **1,152 รูปภาพ**
  * **ชุดทดสอบ (Test Set 20%):** **288 รูปภาพ**

---

## 🔬 3. แบบจำลองที่ใช้ การสกัดคุณลักษณะ ผลการประเมิน และข้อจำกัด

### 3.1 การสกัดคุณลักษณะ (Feature Extraction: 208 มิติ)
แบบจำลองทำการสกัดฟีเจอร์ด้วยหลักการ Digital Image Processing และสถิติ:
1. **RGB Color Histogram (48 มิติ):** ฮิสโตแกรมความถี่ของค่าสีแดง เขียว และน้ำเงิน (ช่องละ 16 bins)
2. **HSV Color Distribution (56 มิติ):** ฮิสโตแกรมของ Hue (24 bins), Saturation (16 bins), และ Value (16 bins) เพื่อจับเนื้อสีและความสดของสีที่ไม่ขึ้นกับความสว่าง
3. **Spatial Grid Layout 4x4 (96 มิติ):** ค่าเฉลี่ยสี RGB และ HSV ในตาราง 16 ช่องเพื่อจับการจัดวางตำแหน่งของผลไม้
4. **Texture & Gradient Features (8 มิติ):** ความชันของขอบภาพ (Gradient) และค่าความแปรปรวนของพิกเซล เพื่อแยกแยะผลไม้ผิวเรียบออกจากผลไม้ที่มีหนามหรือผิวขรุขระ

### 3.2 แบบจำลองที่ใช้ (Model Architecture)
* **อัลกอริทึม:** HistGradientBoostingClassifier (Gradient Boosted Decision Trees จาก Scikit-Learn)
* **เหตุผลที่เลือกใช้:**
  * เหมาะสำหรับการเรียนรู้ความสัมพันธ์แบบ Non-Linear ของคุณลักษณะสีและผิวสัมผัส
  * ทนทานต่อ Outliers และมี Regularization ในตัว
  * ขนาดไฟล์โมเดลเล็กมากเพียง **1.73 MB** 
  * ประหยัด RAM สูงสุด เหมาะสม 100% สำหรับการรันบน Render Cloud Free Tier

### 3.3 ผลการประเมินหลักบนชุดทดสอบ (Evaluation Results)
* **Accuracy บน Test Set (288 รูปภาพจริง):** **58.68%** (เทียบกับค่าสุ่มเดาที่ $\frac{1}{16} = 6.25\%$)
* **Macro Average:** Precision = **0.60**, Recall = **0.59**, F1-Score = **0.59**
* **Weighted Average:** Precision = **0.60**, Recall = **0.59**, F1-Score = **0.59**
* **ผลการทดสอบคลังภาพตัวอย่างในระบบ (Sample Images):** ถูกต้อง **100% (16/16 ภาพ)**
  * ทุเรียน, มังคุด, เงาะ, มะม่วง, มะพร้าว, กล้วย, แอปเปิ้ล, สตรอว์เบอร์รี, แตงโม, เสาวรส, สับปะรด, แก้วมังกร, ขนุน, น้อยหน่า, กระท้อน, ลำไย (ผ่าน 100%)

### 3.4 ข้อจำกัดและการวิเคราะห์ข้อผิดพลาด (Model Limitations & Error Analysis)
1. **ผลไม้ที่มีเฉดสีและรูปทรงใกล้เคียงกัน:** เช่น มะม่วงสุกกับกล้วย หรือแอปเปิ้ลแดงกับเชอร์รี่ อาจมีบางมุมมองที่มีฮิสโตแกรมสีทับซ้อนกัน
2. **ขอบเขตของสายพันธุ์ (Closed-Set Assumption):** โมเดลถูกออกแบบมาเพื่อจำแนกผลไม้ 16 ชนิดหลัก หากนำพืชผลชนิดอื่นที่ไม่ได้อยู่ใน 16 คลาสเข้ามา โมเดลจะจับคู่กับผลไม้ที่มีคุณลักษณะใกล้เคียงที่สุดแทน

---

## 🛠️ 4. ขั้นตอนการติดตั้งและคำสั่งรันระบบบนเครื่อง (Local Setup)

### 4.1 ความต้องการของระบบ (Prerequisites)
* Python 3.10 ขึ้นไป (แนะนำ Python 3.12)
* พื้นที่ว่างบนดิสก์ประมาณ 100 MB (ไม่รวมไฟล์ archive zip)

### 4.2 การติดตั้งแพ็กเกจ (Install Dependencies)
เปิด Terminal หรือ PowerShell แล้วรันคำสั่ง:

```bash
git clone https://github.com/TibParW/Fruits-classification.git
cd Fruits-classification
pip install -r requirements.txt
```

### 4.3 คำสั่งรันเว็บแอปพลิเคชัน (Run Web App)
```bash
python app.py
```
- ระบบจะเปิดเว็บเบราว์เซอร์ให้อัตโนมัติที่ `http://127.0.0.1:7860`
- สามารถทดลองอัปโหลดรูปภาพผลไม้ หรือคลิกเลือกตัวอย่างในหน้าเว็บเพื่อทดสอบการจำแนกได้ทันที

### 4.4 คำสั่งเทรนโมเดลใหม่ (Retrain Model - Optional)
```bash
python train.py --model linear
```
- สคริปต์จะอ่านข้อมูลจากโฟลเดอร์ `dataset/` สกัดฟีเจอร์ เทรนโมเดล บันทึกรายงาน `classification_report.txt`, ภาพ `confusion_matrix.png`, และไฟล์โมเดล `fruit_model.pkl`

---

## 📱 5. วิธีใช้งานและตัวอย่างข้อมูลนำเข้า (User Guide & Sample Inputs)

### 5.1 ขั้นตอนการใช้งานบนเว็บแอป:
1. เข้าสู่หน้าเว็บแอปพลิเคชัน [https://fruits-classification-j3ss.onrender.com](https://fruits-classification-j3ss.onrender.com)
2. **อัปโหลดภาพผลไม้:** สามารถลากไฟล์ภาพมาวาง (Drag & Drop), คลิกเพื่อเลือกไฟล์, ถ่ายภาพจากกล้องเว็บแคม หรือวางภาพจากคลิปบอร์ด
3. **หรือเลือกภาพตัวอย่าง:** คลิกเลือกภาพผลไม้ตัวอย่างในแถบ **"📁 ภาพตัวอย่างสำหรับทดสอบ (Sample Images)"** ด้านล่าง (เช่น ทุเรียน, มังคุด, เงาะ, ขนุน, ลำไย, กระท้อน, น้อยหน่า, แก้วมังกร, มะม่วง, มะพร้าว, กล้วย, แอปเปิ้ล)
4. **ดูผลลัพธ์ทันที (Auto-Predict):** ระบบจะคำนวณและแสดงชื่อผลไม้ภาษาไทย ภาษาอังกฤษ พร้อม Emoji และแถบระดับความมั่นใจ Top-5 ในเสี้ยววินาที

---

## 📂 6. คำอธิบายไฟล์และโฟลเดอร์สำคัญในโครงการ (Project Structure)

```text
Fruits-classification/
├── dataset/                     # โฟลเดอร์ชุดข้อมูล 267 คลาสผลไม้ (คลาสละ 25 รูป รวม 6,675 รูป)
├── sample_images/               # ภาพผลไม้ตัวอย่าง 12 ชนิด สำหรับคลิกทดสอบบนหน้าเว็บ
│   ├── durian.jpg               # ตัวอย่างภาพทุเรียน 👑
│   ├── mangosteen.jpg          # ตัวอย่างภาพมังคุด 👑
│   ├── rambutan.jpg            # ตัวอย่างภาพเงาะ 🔴
│   ├── jackfruit.jpg           # ตัวอย่างภาพขนุน 🍈
│   ├── longan.jpg              # ตัวอย่างภาพลำไย 🌰
│   ├── santol.jpg              # ตัวอย่างภาพกระท้อน 🟡
│   ├── custard_apple.jpg       # ตัวอย่างภาพน้อยหน่า 🍈
│   ├── dragonfruit.jpg         # ตัวอย่างภาพแก้วมังกร 🐲
│   ├── mango.jpg               # ตัวอย่างภาพมะม่วง 🥭
│   ├── coconut.jpg             # ตัวอย่างภาพมะพร้าว 🥥
│   ├── banana.jpg              # ตัวอย่างภาพกล้วย 🍌
│   └── apple.jpg               # ตัวอย่างภาพแอปเปิ้ล 🍎
├── fruit_classification.ipynb  # Jupyter Notebook ส่งงาน แสดงกระบวนการเทรนและวิเคราะห์ครบ 8 ขั้นตอน
├── app.py                       # ซอร์สโค้ดเว็บแอปพลิเคชัน Gradio สไตล์ Minimalist พร้อม Auto-Predict
├── train.py                     # สคริปต์สกัดฟีเจอร์ เทรน และประเมินผลโมเดล 267 คลาส
├── version.py                   # ระบุเวอร์ชัน (v1.0.9) และ Metadata ของโครงการ
├── fruit_model.pkl              # ไฟล์โมเดลที่บันทึกไว้จริง (Linear Classifier 267 คลาส ขนาด 3.23 MB)
├── confusion_matrix.png         # ภาพ Confusion Matrix Heatmap ขนาด 300x267 คลาส
├── classification_report.txt    # รายงานผล Precision, Recall, F1-score รายคลาส
├── requirements.txt             # รายการไลบรารีที่จำเป็น (Scikit-Learn, Gradio, Pillow, NumPy, ฯลฯ)
├── vercel.json                  # คอนฟิกเส้นทางสำหรับ Cloud Deployment
└── README.md                    # เอกสารอธิบายโครงการฉบับสมบูรณ์
```
