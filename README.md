# 🍎 โครงการพัฒนาระบบจำแนกชนิดผลไม้ (Fruit Classifier)
## Deep Vision Image Classification & Gradio Web Application (26 Real Fruit Classes)

![Version](https://img.shields.io/badge/version-2.1.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.12-brightgreen.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9-orange.svg)
![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.30-blue.svg)
![Gradio](https://img.shields.io/badge/Gradio-6.28-red.svg)
![Classes](https://img.shields.io/badge/Classes-26%20Pure%20Fruits-success.svg)
![Model](https://img.shields.io/badge/Model-MobileNetV2%20Deep%20Features-blueviolet.svg)
![Samples](https://img.shields.io/badge/Samples-1%2C916%20Real%20Photos-blue.svg)
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

อย่างไรก็ดี ในการใช้งานจริง รูปภาพที่ได้จากกล้องมือถือหรืออินเทอร์เน็ตมักมีสิ่งรบกวน เช่น เขียงไม้ ตะกร้าหวาย โต๊ะอาหาร หรือผลไม้ที่ถูกหั่น/ผ่าครึ่ง หากใช้การสกัดฟีเจอร์แบบพื้นฐาน (Raw Pixels หรือ Color Histograms เพียงอย่างเดียว) โมเดลจะไม่เข้าใจรูปทรงเรขาคณิต (Shape & Semantics) ทำให้ผลไม้ที่มีสีเหลืองเหมือนกัน (กล้วย มะม่วง เลมอน) หรือสีแดงเหมือนกัน (แอปเปิ้ล ลิ้นจี่ เชอร์รี่) ทำนายสับสนปนเปกัน

### 1.2 ประโยชน์ของระบบ (Benefits)
1. **ตัดผักและสิ่งรบกวนออก 100%:** คัดสรรเฉพาะ **26 ผลไม้ยอดนิยมและผลไม้ไทยเขตร้อนแท้** (ทุเรียน มังคุด เงาะ มะม่วง มะพร้าว กล้วย แอปเปิ้ล สตรอว์เบอร์รี แตงโม เสาวรส สับปะรด แก้วมังกร เลมอน มะนาว ส้ม ขนุน น้อยหน่า กระท้อน ลำไย ฯลฯ)
2. **ระบบสกัดคุณลักษณะ Deep Vision (MobileNetV2 2,280 มิติ):** ผสานคุณลักษณะเชิงลึก (Deep Semantic Representations 1,280 มิติ) ร่วมกับ Normalized Likelihoods (1,000 มิติ) จับรูปทรง พื้นผิวเปลือก เมล็ด และเนื้อผลไม้ได้แม่นยำ ไม่สับสนแม้มีฉากหลังเป็นเขียงไม้หรือตะกร้า
3. **เบาและรวดเร็วสูง (ONNX Runtime CPU):** รันบน CPU ด้วยความเร็วเพียง 15 ms ต่อภาพ และใช้ RAM ต่ำกว่า 80 MB ปลอดภัย ไม่เสี่ยงโดนตัด RAM บน Render Free Tier

---

## 📦 2. แหล่งที่มา จำนวนข้อมูล และการแบ่งชุดข้อมูล (Dataset & Splitting)

### 2.1 แหล่งที่มาของข้อมูล (Data Sources)
* **Fruit-262 Dataset (Kaggle):** ชุดข้อมูลภาพถ่ายผลไม้จริงจากสภาพแวดล้อมธรรมชาติทั่วโลก (คัดเลือก 26 ชนิดผลไม้แท้ x 70+ รูปภาพ)
* **Sample Images & Augmentation:** ภาพตัวอย่างทดสอบพร้อมเทคนิค Data Augmentation (Horizontal Flip)

### 2.2 จำนวนข้อมูลและการแบ่งชุดฝึก/ชุดทดสอบ (Dataset Distribution)
* **จำนวนคลาสทั้งหมด:** **26 คลาสผลไม้ยอดนิยมแท้ (Zero Vegetables)**
* **จำนวนภาพรวมทั้งหมด:** **1,916 รูปภาพ**
* **การแบ่งชุดข้อมูล (Data Splitting):** ใช้วิธี **Stratified Train-Test Split (80:20)**:
  * **ชุดฝึกสอน (Train Set 80%):** **1,532 รูปภาพ**
  * **ชุดทดสอบ (Test Set 20%):** **384 รูปภาพ**

---

## 🔬 3. แบบจำลองที่ใช้ การสกัดคุณลักษณะ ผลการประเมิน และข้อจำกัด

### 3.1 การสกัดคุณลักษณะ (Deep Feature Extraction: 2,280 มิติ)
แบบจำลองทำการสกัดฟีเจอร์ระดับ Deep Convolutional Feature Space ผ่านโครงข่ายประสาทเทียม MobileNetV2 (ONNX Graph):
1. **Deep Semantic Representations (1,280 มิติ):** สกัดจาก Global Average Pooling Layer (Node 472) ก่อนชั้น Classification Head แล้วทำ $L_2$ Normalization เพื่อจับความสัมพันธ์ของรูปทรงเรขาคณิต ผิวสัมผัส ลายเปลือก และริ้วรอยของผลไม้
2. **Normalized Semantic Likelihoods (1,000 มิติ):** สกัดจาก Softmax Likelihoods ของ ImageNet Categories ถ่วงน้ำหนักเพื่อเพิ่มพลังการแยกแยะผลไม้กลุ่มสากล เช่น เลมอน กล้วย ส้ม แอปเปิ้ล

### 3.2 แบบจำลองที่ใช้ (Model Architecture)
* **อัลกอริทึม:** Multinomial Logistic Regression ($C=5.0$, `max_iter=1000`) บน Deep Feature Representation
* **เหตุผลที่เลือกใช้:**
  * สามารถแยกแยะ Hyperplane ใน Deep Embedding Space ขนาด 2,280 มิติได้อย่างแม่นยำสูง
  * ใช้เวลาเทรนเพียงไม่กี่วินาที
  * ขนาดไฟล์โมเดลเล็กมากเพียง **0.21 MB** (บวกกับ ONNX Graph 13.3 MB)
  * ประหยัด RAM สูงสุด เหมาะสม 100% สำหรับการรันบน Render Cloud Free Tier

### 3.3 ผลการประเมินหลักบนชุดทดสอบ (Evaluation Results)
* **Accuracy บน Test Set (384 รูปภาพจริง):** **82.55%** (เทียบกับค่าสุ่มเดาที่ $\frac{1}{26} = 3.85\%$)
* **Macro Average:** Precision = **0.83**, Recall = **0.82**, F1-Score = **0.82**
* **Weighted Average:** Precision = **0.83**, Recall = **0.83**, F1-Score = **0.82**
* **ผลการทดสอบภาพจริงจากผู้ใช้ (User Real Test Images):**
  * `กล้วยบนพื้นขาว (Banana)` -> **กล้วย (Banana) 99.4% (ถูกต้อง)**
  * `มะม่วงหั่นชิ้นบนเขียงไม้ (Mango)` -> **มะม่วง (Mango) 90.0% (ถูกต้อง)**
  * `แอปเปิ้ลบนโต๊ะไม้ (Apple)` -> **แอปเปิ้ล (Apple) 42.5% (ถูกต้อง)**
  * `เลมอนในตะกร้าหวาย (Lemon)` -> **เลมอน (Lemon) 43.5% (ถูกต้อง)**
* **ผลการทดสอบคลังภาพตัวอย่างในระบบ (Sample Images):** ถูกต้อง **100% (16/16 ภาพ)**

### 3.4 ข้อจำกัดและการวิเคราะห์ข้อผิดพลาด (Model Limitations & Error Analysis)
1. **ภาพที่มีผลไม้ปะปนหลายชนิดในภาพเดียวกัน:** หากในภาพมีผลไม้มากกว่า 1 ชนิด (เช่น ตะกร้าผลไม้รวม) โมเดลจะตรวจจับผลไม้ที่เด่นหรือกินพื้นที่มากที่สุดในบริเวณกึ่งกลางภาพ
2. **ขอบเขตของสายพันธุ์ (Closed-Set Assumption):** โมเดลถูกออกแบบมาเพื่อจำแนกผลไม้ 26 ชนิดหลัก หากนำพืชผลชนิดอื่นที่ไม่ได้อยู่ใน 26 คลาสเข้ามา โมเดลจะจับคู่กับผลไม้ที่มีผิวสัมผัสและรูปทรงใกล้เคียงที่สุดแทน

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
