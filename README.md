# RAG Control System Exam Grader

ระบบตรวจข้อสอบวิชาระบบควบคุม (Control System) โดยใช้เทคโนโลยี RAG (Retrieval-Augmented Generation)

## คุณสมบัติ

- ✅ **ตรวจข้อสอบอัตโนมัติ**: ตรวจและให้คะแนนคำตอบโดยอิงจากฐานความรู้
- ✅ **ให้ Feedback**: วิเคราะห์และให้คำแนะนำการปรับปรุงคำตอบ
- ✅ **RAG Pipeline**: ดึงความรู้ที่เกี่ยวข้องจาก Vector Database
- ✅ **รองรับหลายรูปแบบ**: PDF, TXT, DOCX
- ✅ **ภาษาไทย**: รองรับการประมวลผลภาษาไทย
- ✅ **โหมดถาม-ตอบ**: ถามคำถามเกี่ยวกับระบบควบคุมได้โดยตรง

## โครงสร้างโปรเจค

```
RAG_ControlSystemPrj/
├── src/                          # Source code
│   ├── __init__.py
│   ├── document_loader.py        # โหลดและประมวลผลเอกสาร
│   ├── vector_store.py           # จัดการ Vector Database
│   ├── rag_engine.py             # RAG Pipeline หลัก
│   └── exam_grader.py            # ระบบตรวจข้อสอบ
├── data/                         # ข้อมูล
│   ├── knowledge_base/           # ฐานความรู้ (PDF, TXT, DOCX)
│   ├── exams/                    # ข้อสอบ
│   └── vector_db/                # Vector Database (ChromaDB)
├── examples/                     # ตัวอย่าง
│   └── sample_exam.json          # ตัวอย่างข้อสอบ
├── config.py                     # การตั้งค่า
├── requirements.txt              # Dependencies
├── main.py                       # Main script
└── README.md                     # คู่มือ
```

## การติดตั้ง

### ข้อกำหนดระบบ

- Python 3.8+
- Ubuntu / Linux / macOS / Windows
- RAM อย่างน้อย 4GB
- Ollama (สำหรับใช้ Local LLM) หรือ OpenAI API Key

### 1. Clone โปรเจค

```bash
git clone <repository-url>
cd RAG_ControlSystemPrj
```

### 2. สร้าง Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# หรือ
venv\Scripts\activate  # Windows
```

### 3. ติดตั้ง Dependencies

```bash
pip install -r requirements.txt
```

### 4. ติดตั้ง Ollama (ถ้าใช้ Local LLM)

**Ubuntu/Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama2
```

**macOS:**
```bash
brew install ollama
ollama pull llama2
```

**หรือใช้ OpenAI:**
```bash
export OPENAI_API_KEY="your-api-key"
export LLM_PROVIDER="openai"
```

## การใช้งาน

### 1. เตรียมฐานความรู้

วางไฟล์ความรู้เกี่ยวกับระบบควบคุมในโฟลเดอร์ `data/knowledge_base/`

รองรับไฟล์:
- `.pdf` - เอกสาร PDF
- `.txt` - ไฟล์ข้อความ
- `.docx` - Microsoft Word

ตัวอย่างไฟล์ที่มีอยู่แล้ว:
- `control_system_basics.txt`
- `control_theory.txt`

### 2. โหมดถาม-ตอบ (Query Mode)

```bash
python main.py --query
```

ตัวอย่างการใช้งาน:
```
คำถาม: ความแตกต่างระหว่างระบบวงเปิดและวงปิดคือ
คำตอบ: [ระบบจะดึงความรู้จาก vector database และตอบคำถาม]
```

### 3. โหมดตรวจข้อสอบ (Grading Mode)

สร้างไฟล์ข้อสอบในรูปแบบ JSON:

```json
[
  {
    "question": "อธิบายความแตกต่างระหว่างระบบควบคุมแบบวงเปิดและวงปิด",
    "student_answer": "ระบบวงเปิดไม่มี feedback แต่ระบบวงปิดมี feedback",
    "correct_answer": "ระบบควบคุมแบบวงเปิดไม่มีการป้อนกลับ..."
  }
]
```

รันคำสั่ง:
```bash
python main.py --grade examples/sample_exam.json --output results.json
```

ผลลัพธ์:
- `results.json` - ผลคะแนนและรายละเอียดทั้งหมด
- `results_report.txt` - รายงานแบบเต็ม

## การตั้งค่า

แก้ไขไฟล์ `config.py`:

```python
# เลือก LLM Provider
LLM_PROVIDER = "ollama"  # หรือ "openai"

# ตั้งค่า Ollama
OLLAMA_MODEL = "llama2"
OLLAMA_BASE_URL = "http://localhost:11434"

# ตั้งค่า OpenAI
OPENAI_API_KEY = "your-api-key"

# ตั้งค่า Embeddings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ตั้งค่าการแบ่ง Chunk
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ตั้งค่าการค้นหา
TOP_K_RESULTS = 5
```

## ตัวอย่างการใช้งานใน Python

### ตรวจข้อสอบ

```python
from src.exam_grader import ExamGrader
from src.rag_engine import RAGEngine

# สร้าง RAG Engine
rag_engine = RAGEngine()
rag_engine.load_knowledge_base()

# สร้าง Exam Grader
grader = ExamGrader(rag_engine)

# ตรวจข้อเดียว
result = grader.grade_answer(
    question="PID Controller ประกอบด้วยอะไรบ้าง",
    student_answer="P, I, D",
    correct_answer="Proportional, Integral, Derivative"
)

print(f"คะแนน: {result['score']}")
print(f"Feedback: {result['feedback']}")
```

### Query ธรรมดา

```python
from src.rag_engine import RAGEngine

# สร้าง RAG Engine
engine = RAGEngine()
engine.load_knowledge_base()

# ถามคำถาม
result = engine.query("ระบบควบคุมแบบวงปิดคืออะไร")
print(result['answer'])
```

## การพัฒนาต่อ

### เพิ่มฐานความรู้

เพียงวางไฟล์เพิ่มใน `data/knowledge_base/` แล้วรันใหม่

### ปรับแต่ง Prompt

แก้ไข `GRADING_PROMPT_TEMPLATE` ใน `config.py`

### เปลี่ยน Vector Database

แก้ไข `VECTOR_DB_TYPE` ใน `config.py` เป็น `"faiss"` แทน `"chroma"`

## Troubleshooting

### ปัญหา: Ollama ไม่ทำงาน

```bash
# ตรวจสอบว่า Ollama รันอยู่
ollama list

# เริ่ม Ollama
ollama serve
```

### ปัญหา: Memory Error

ลดขนาด `CHUNK_SIZE` และ `TOP_K_RESULTS` ใน `config.py`

### ปัญหา: การโหลดเอกสารช้า

ลด `CHUNK_SIZE` หรือจำนวนเอกสารใน knowledge base

## License

MIT License

## ผู้พัฒนา

RAG Control System Exam Grader Project

## การสนับสนุน

หากมีปัญหาหรือข้อเสนอแนะ:
1. เปิด Issue บน GitHub
2. ติดต่อผู้พัฒนา

---

สร้างด้วย LangChain, ChromaDB, และ Ollama/OpenAI
