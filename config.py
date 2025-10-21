"""
Configuration file for RAG Control System Exam Grader
ไฟล์ตั้งค่าสำหรับระบบตรวจข้อสอบ
"""

import os
from pathlib import Path

# โครงสร้างโฟลเดอร์
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"
EXAMS_DIR = DATA_DIR / "exams"
VECTOR_DB_DIR = DATA_DIR / "vector_db"

# การตั้งค่า LLM
# ใช้ OpenAI หรือ Local LLM (เช่น Ollama)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "openai" or "ollama"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:latest")  # Changed to llama3 for better Thai support
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# การตั้งค่า Embeddings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# การตั้งค่า Vector Database
VECTOR_DB_TYPE = "chroma"  # "chroma" or "faiss"
COLLECTION_NAME = "control_system_knowledge"

# การตั้งค่าการตรวจข้อสอบ
SIMILARITY_THRESHOLD = 0.7
TOP_K_RESULTS = 5

# Template สำหรับ Prompt
GRADING_PROMPT_TEMPLATE = """
คุณเป็นผู้เชี่ยวชาญด้านระบบควบคุม (Control System) และจะตอบเป็นภาษาไทยเท่านั้น

จากความรู้ที่เกี่ยวข้อง:
{context}

คำถาม: {question}

คำตอบของนักศึกษา: {student_answer}

เฉลย/คำตอบที่ถูกต้อง: {correct_answer}

กรุณาตรวจและให้คะแนนคำตอบของนักศึกษา โดย:
1. วิเคราะห์ความถูกต้องของคำตอบ
2. เปรียบเทียบกับเฉลยและความรู้ที่เกี่ยวข้อง
3. ให้คะแนนเป็นเปอร์เซ็นต์ (0-100)
4. ให้คำอธิบายและข้อเสนอแนะเป็นภาษาไทย

สำคัญ: ตอบเป็น JSON เท่านั้น ไม่ต้องมีข้อความอื่น:
{{
  "score": คะแนนเป็นตัวเลข 0-100,
  "is_correct": true หรือ false,
  "feedback": "คำอธิบายและข้อเสนอแนะเป็นภาษาไทย",
  "key_points_covered": ["จุดสำคัญที่ตอบถูก"],
  "missing_points": ["จุดที่ขาดหายไป"]
}}
"""
