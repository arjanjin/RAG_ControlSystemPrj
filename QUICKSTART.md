# 🚀 QUICK START GUIDE

## การเริ่มใช้งานแบบด่วน

### 1. 🔧 Setup ระบบ
```bash
# เปิด terminal และไปที่โฟลเดอร์ของโปรเจกต์
cd "/path/to/RAG_ControlSystemPrj"

# รันสคริปต์ setup (แนะนำ)
./setup.sh

# หรือเปิด virtual environment เอง
source venv/bin/activate
```

### 2. 🗣️ ทดสอบการถาม-ตอบ
```bash
python main.py --query
```
จากนั้นพิมพ์คำถาม เช่น:
- ระบบควบคุมแบบวงเปิดคืออะไร
- PID Controller ประกอบด้วยอะไรบ้าง

### 3. 📝 ทดสอบการตรวจข้อสอบ
```bash
python main.py --grade examples/sample_exam.json --output results.json
```

### 4. 🌐 ใช้งาน Web Interface (แนะนำ!)
```bash
streamlit run web_interface.py
```
จากนั้นเปิดบราวเซอร์ไปที่ http://localhost:8501

## 🎯 ฟีเจอร์หลัก

### Q&A Mode
- ถามคำถามเกี่ยวกับระบบควบคุมเป็นภาษาไทย
- ระบบจะตอบพร้อมแหล่งอ้างอิง

### Exam Grading
- อัพโหลดไฟล์ข้อสอบ JSON
- ระบบตรวจและให้คะแนนอัตโนมัติ
- สร้างรายงานผลการตรวจ

### Web Interface
- ใช้งานผ่านเว็บเบราว์เซอร์
- Interface ภาษาไทย
- ครบทุกฟีเจอร์ในหน้าเดียว

## 📋 รูปแบบไฟล์ข้อสอบ
```json
[
  {
    "question": "คำถาม",
    "student_answer": "คำตอบของนักศึกษา", 
    "correct_answer": "เฉลย"
  }
]
```

## 🚨 ข้อกำหนด
- Ollama ต้องรันอยู่ (ollama serve)
- Model llama3:latest ต้องติดตั้งแล้ว
- Python 3.8+ และ dependencies ที่ติดตั้งแล้ว

## 🆘 การแก้ไขปัญหา
- ถ้า Ollama ไม่ทำงาน: `ollama serve`
- ถ้าไม่มี model: `ollama pull llama3:latest` 
- ถ้า ChromaDB error: ลบโฟลเดอร์ `data/vector_db` และรันใหม่

## 📞 ช่วยเหลือ
ดู README.md สำหรับข้อมูลแบบละเอียด