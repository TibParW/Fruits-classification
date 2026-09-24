# 🍎 โครงการพัฒนาระบบจำแนกชนิดผลไม้ (Fruit Classifier)
## Machine Learning Image Classification & Gradio Web Application (21 Pure Fruit Classes)

![Version](https://img.shields.io/badge/version-2.2.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.12-brightgreen.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange.svg)
![Gradio](https://img.shields.io/badge/Gradio-4%2B-red.svg)
![Classes](https://img.shields.io/badge/Classes-21%20Pure%20Fruits-success.svg)
![Model](https://img.shields.io/badge/Model-HistGradientBoosting-blueviolet.svg)
![Samples](https://img.shields.io/badge/Samples-3%2C517%20Real%20Photos-blue.svg)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-73.1%25-brightgreen.svg)
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

อย่างไรก็ดี ในการจำแนกภาพผลไม้ด้วยแบบจำลอง Classical Machine Learning ดั้งเดิม การใช้ข้อมูลพิกเซลสีดิบ (Raw Pixels) หรือฮิสโตแกรมสีเพียงอย่างเดียว มักเกิดปัญหาเมื่อภาพถ่ายมีฉากหลัง เช่น โต๊ะไม้ ตะกร้าหวาย หรือแสงเงา ซึ่งทำให้ผลไม้สีแดง (เช่น แอปเปิ้ล) ถูกเข้าใจผิดเป็นลิ้นจี่ หรือมะละกอผ่าซีกสีส้มถูกเข้าใจผิดเป็นส้ม การออกแบบกระบวนการวิศวกรรมคุณลักษณะ (Feature Engineering 190 มิติ) ที่ผสานดัชนีสีเชิงเส้น (Linear Color Difference), มุมสีเชิงวงกลม (Circular Hue), ผิวสัมผัส (Local Binary Patterns: LBP), การตรวจจับโพรงเมล็ดกึ่งกลาง (Dark Cavity), และการจัดวางเชิงพื้นที่ (Spatial 3x3 Grid) จึงเป็นหัวใจสำคัญในการแก้ปัญหานี้ให้แม่นยำสูง

### 1.2 ประโยชน์ของระบบ (Benefits)
1. **ตัดผักและสิ่งรบกวนออก 100%:** คัดสรรเฉพาะ **21 ผลไม้ยอดนิยมและผลไม้ไทยเขตร้อนแท้** (มะละกอ สาลี่/ลูกแพร์ ลิ้นจี่ แอปเปิ้ล เลมอน ส้ม ทุเรียน มังคุด เงาะ มะม่วง มะพร้าว กล้วย สตรอว์เบอร์รี แตงโม เสาวรส สับปะรด แก้วมังกร ขนุน น้อยหน่า กระท้อน ลำไย)
2. **การสกัดคุณลักษณะหลายมิติ (Feature Engineering 190 มิติ):** ผสานดัชนีสีเชิงเส้น (ExR, ExG, ExB, ExY), มุมสีเชิงวงกลม $\cos(\theta)/\sin(\theta)$, ฮิสโตแกรมสี RGB & HSV, Center-Crop 60%, ผิวสัมผัส LBP 8-neighbor, และตัวตรวจจับโพรงเมล็ด
3. **เบา รวดเร็ว และเป็น Classical ML แท้:** พัฒนาด้วย Scikit-Learn `HistGradientBoostingClassifier` ขนาดโมเดลเพียง **7.0 MB** รันบน CPU ด้วยความเร็วเพียง 15 ms ต่อภาพ และใช้ RAM ต่ำกว่า 100 MB ปลอดภัย ไม่เสี่ยงโดนตัด RAM บน Render Free Tier

---

## 📦 2. แหล่งที่มา จำนวนข้อมูล และการแบ่งชุดข้อมูล (Dataset & Splitting)

### 2.1 แหล่งที่มาของข้อมูล (Data Sources)
* **Fruit-262 Dataset:** ชุดข้อมูลภาพถ่ายผลไม้จริงจากสภาพแวดล้อมธรรมชาติทั่วโลก (คัดเลือก 21 ชนิดผลไม้แท้ x 120-150 รูปภาพ)
* **Fruits-360 Dataset:** เสริมภาพผลไม้หลายสายพันธุ์เดี่ยวแบบคมชัด เช่น แอปเปิ้ลแดง/เขียว/ทอง, เลมอนเหลือง, สาลี่
* **Sample Images & Augmentation:** ภาพตัวอย่างทดสอบประจำแอป 21 ชนิด พร้อม Data Augmentation (Horizontal Flip, Slight Center Crop)
* **Actual Fruit Photos:** ภาพถ่ายผลไม้จริงจากการทดสอบของผู้ใช้

### 2.2 จำนวนข้อมูลและการแบ่งชุดฝึก/ชุดทดสอบ (Dataset Distribution)
* **จำนวนคลาสทั้งหมด:** **21 คลาสผลไม้ยอดนิยมแท้ (Zero Vegetables)**
* **จำนวนภาพรวมทั้งหมด:** **3,517 รูปภาพ**
* **การแบ่งชุดข้อมูล (Data Splitting):** ใช้วิธี **Stratified Train-Test Split (85:15)**:
  * **ชุดฝึกสอน (Train Set 85%):** **2,989 รูปภาพ**
  * **ชุดทดสอบ (Test Set 15%):** **528 รูปภาพ**

---

## 🔬 3. แบบจำลองที่ใช้ การสกัดคุณลักษณะ ผลการประเมิน และข้อจำกัด

### 3.1 การสกัดคุณลักษณะ (Feature Extraction: 190 มิติ)
แบบจำลองทำการสกัดฟีเจอร์ด้วยหลักการ Digital Image Processing และสถิติ:
1. **Linear Color Indices & Circular Hue (16 มิติ):** Normalized RGB ($r, g, b$), Excess Red ($ExR$), Excess Green ($ExG$), Excess Yellow ($ExY$), มุมสีเชิงวงกลม $\cos(\theta)$ และ $\sin(\theta)$ เพื่อแก้ปัญหา Hue wrap-around สำหรับสีแดง
2. **Color Histograms (56 มิติ):** ฮิสโตแกรมความถี่ RGB (24 ช่อง) และ HSV (32 ช่อง: Hue 16, Sat 8, Val 8)
3. **Center 60% Region (34 มิติ):** ตัดขอบโต๊ะและฉากหลังออก โฟกัสเฉพาะกึ่งกลางผลไม้
4. **Saturated Fruit Body Mask (23 มิติ):** กรองเฉพาะพิกเซลเนื้อผลไม้ที่มีสีสด ($S > 40$) ไม่รวมฉากหลังสีขาวหรือโต๊ะไม้
5. **Surface Texture (22 มิติ):** Local Binary Patterns (LBP 16 bins) ร่วมกับค่าความชัน Sobel Gradient Magnitude เพื่อแยกผิวเรียบ (แอปเปิ้ล, สาลี่) ออกจากผิวขรุขระ (ลิ้นจี่, เงาะ, ทุเรียน)
6. **Dark Center Cavity Detector (3 มิติ):** ตรวจจับเมล็ดสีดำในโพรงกึ่งกลาง (มะละกอ, เสาวรส, แตงโม)
7. **Spatial 3x3 Grid Layout (36 มิติ):** ตาราง 9 ช่องบันทึกการจัดวางสีเชิงพื้นที่

### 3.2 แบบจำลองที่ใช้ (Model Architecture)
* **อัลกอริทึม:** HistGradientBoostingClassifier (Gradient Boosted Decision Trees จาก Scikit-Learn)
* **พารามิเตอร์หลัก:** `max_iter=250`, `learning_rate=0.08`, `l2_regularization=0.5`
* **เหตุผลที่เลือกใช้:**
  * เหมาะสำหรับการเรียนรู้ความสัมพันธ์แบบ Non-Linear ของฟีเจอร์ 190 มิติ
  * ทนทานต่อ Outliers และมี Regularization ในตัว
  * ขนาดไฟล์โมเดลเล็กกะทัดรัด (7.0 MB)
  * ประหยัด RAM สูงสุด เหมาะสม 100% สำหรับการรันบน Render Cloud Free Tier

### 3.3 ผลการประเมินหลักบนชุดทดสอบ (Evaluation Results)
* **Accuracy บน Test Set (528 รูปภาพจริง):** **73.11%** (เทียบกับค่าสุ่มเดาที่ $\frac{1}{21} = 4.76\%$)
* **Macro Average:** Precision = **0.72**, Recall = **0.72**, F1-Score = **0.72**
* **Weighted Average:** Precision = **0.74**, Recall = **0.73**, F1-Score = **0.73**
* **ผลการทดสอบภาพถ่ายผลไม้จริงของผู้ใช้ (Real Test Crops):** ถูกต้อง **100% (5/5 รูป)**
  * มะละกอ (Papaya): ผ่าน 78.5%
  * สาลี่ / ลูกแพร์ (Pear): ผ่าน 96.7%
  * ลิ้นจี่ (Lychee): ผ่าน 100.0%
  * แอปเปิ้ล (Apple): ผ่าน 100.0%
  * เลมอน / มะนาวเหลือง (Lemon): ผ่าน 100.0%

### 3.4 ข้อจำกัดและการวิเคราะห์ข้อผิดพลาด (Model Limitations & Error Analysis)
1. **ผลไม้ที่มีเฉดสีและรูปทรงใกล้เคียงกัน:** เช่น ส้มกับมะละกอ (หากถ่ายเฉพาะเนื้อสีส้มโดยไม่เห็นเมล็ดสีดำ) หรือสาลี่กับแอปเปิ้ลเขียว
2. **ขอบเขตของสายพันธุ์ (Closed-Set Assumption):** โมเดลถูกออกแบบมาเพื่อจำแนกผลไม้ 21 ชนิดหลัก หากนำพืชผลชนิดอื่นที่ไม่ได้อยู่ใน 21 คลาสเข้ามา โมเดลจะจับคู่กับผลไม้ที่มีคุณลักษณะใกล้เคียงที่สุดแทน

---

## 🛠️ 4. ขั้นตอนการติดตั้งและคำสั่งรันระบบบนเครื่อง (Local Setup)

### 4.1 ความต้องการของระบบ (Prerequisites)
* Python 3.10 ขึ้นไป (แนะนำ Python 3.12)
* พื้นที่ว่างบนดิสก์ประมาณ 100 MB

### 4.2 การติดตั้งแพ็กเกจ (Install Dependencies)
```bash
pip install -r requirements.txt
```

### 4.3 การเทรนโมเดลใหม่ (Training)
```bash
python train.py
```
สคริปต์จะทำการโหลดชุดข้อมูล สกัดคุณลักษณะ 190 มิติ ฝึกโมเดล สร้าง `confusion_matrix.png`, `classification_report.txt` และบันทึก `fruit_model.pkl`

### 4.4 การรันเว็บแอปพลิเคชัน (Launch Web App)
```bash
python app.py
```
เปิดเบราว์เซอร์ไปที่ `http://localhost:7860` เพื่อทดลองใช้งาน

---

## 🚀 5. การติดตั้งใช้งานบน Render (Deployment to Render Cloud)

โปรเจกต์นี้ได้รับการคอนฟิกสำหรับ Deploy บน Render.com อย่างสมบูรณ์:
* **Build Command:** `pip install -r requirements.txt`
* **Start Command:** `python app.py --port $PORT`
* **Health Check & Auto-Restart:** รองรับการทำงานต่อเนื่อง 24/7 บน Free Tier
