#!/usr/bin/env python3
"""
Simple Streamlit Web Interface for RAG Control System
เว็บอินเทอร์เฟซสำหรับระบบตรวจข้อสอบ RAG
"""

import streamlit as st
import json
import tempfile
import os
from pathlib import Path

# Add project root to path for imports
import sys
sys.path.append(str(Path(__file__).parent))

from src.rag_engine import RAGEngine
from src.exam_grader import ExamGrader
from src.vector_store import VectorStoreManager

# Configure Streamlit page
st.set_page_config(
    page_title="RAG Control System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'rag_engine' not in st.session_state:
    st.session_state.rag_engine = None
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

@st.cache_resource
def load_rag_system():
    """Load RAG system components"""
    try:
        vector_store = VectorStoreManager()
        rag_engine = RAGEngine(vector_store_manager=vector_store)
        return rag_engine, vector_store
    except Exception as e:
        st.error(f"Error loading RAG system: {e}")
        return None, None

def main():
    """Main web interface"""
    
    # Title and header
    st.title("🤖 RAG Control System")
    st.subheader("ระบบตรวจข้อสอบวิชาระบบควบคุมด้วย AI")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Menu")
        mode = st.selectbox(
            "เลือกโหมดการใช้งาน:",
            ["🗣️ Q&A Mode", "📝 Exam Grading", "📊 System Status"]
        )
    
    # Load RAG system
    if st.session_state.rag_engine is None:
        with st.spinner("กำลังโหลดระบบ..."):
            rag_engine, vector_store = load_rag_system()
            st.session_state.rag_engine = rag_engine
            st.session_state.vector_store = vector_store
    
    if st.session_state.rag_engine is None:
        st.error("ไม่สามารถโหลดระบบได้ กรุณาตรวจสอบการติดตั้ง")
        return
    
    # Mode selection
    if mode == "🗣️ Q&A Mode":
        qa_mode()
    elif mode == "📝 Exam Grading":
        exam_grading_mode()
    elif mode == "📊 System Status":
        system_status_mode()

def qa_mode():
    """Q&A Mode Interface"""
    st.header("🗣️ Q&A Mode - ถาม-ตอบเกี่ยวกับระบบควบคุม")
    
    # Sample questions
    st.subheader("🔍 คำถามตัวอย่าง:")
    sample_questions = [
        "ระบบควบคุมแบบวงเปิดคืออะไร",
        "PID Controller ประกอบด้วยส่วนใดบ้าง",
        "ฟังก์ชันถ่ายโอน (Transfer Function) คืออะไร",
        "ความแตกต่างระหว่าง Steady State และ Transient Response",
        "Root Locus คืออะไรและใช้ประโยชน์อย่างไร"
    ]
    
    cols = st.columns(2)
    for i, question in enumerate(sample_questions):
        col = cols[i % 2]
        if col.button(f"📝 {question}", key=f"sample_{i}"):
            st.session_state.question_input = question
    
    # Question input
    question = st.text_area(
        "🤔 ใส่คำถามของคุณ:",
        value=getattr(st.session_state, 'question_input', ''),
        height=100,
        key="question_textarea"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔍 ถามคำถาม", type="primary"):
            if question.strip():
                with st.spinner("กำลังค้นหาคำตอบ..."):
                    try:
                        result = st.session_state.rag_engine.query(
                            question, return_context=True
                        )
                        
                        # Display answer
                        st.subheader("💡 คำตอบ:")
                        st.write(result['answer'])
                        
                        # Display sources
                        st.subheader(f"📚 แหล่งอ้างอิง ({result['num_sources']} แหล่ง):")
                        if 'sources' in result:
                            for i, source in enumerate(result['sources'][:3], 1):
                                with st.expander(f"แหล่งที่ {i}"):
                                    st.write(source['content'][:300] + "...")
                    
                    except Exception as e:
                        st.error(f"เกิดข้อผิดพลาด: {e}")
            else:
                st.warning("กรุณาใส่คำถาม")

def exam_grading_mode():
    """Exam Grading Mode Interface"""
    st.header("📝 Exam Grading - ตรวจข้อสอบ")
    
    tab1, tab2 = st.tabs(["📄 Single Question", "📊 Full Exam"])
    
    with tab1:
        st.subheader("ตรวจคำตอบข้อเดียว")
        
        # Input form
        with st.form("single_question"):
            question = st.text_area("คำถาม:", height=100)
            student_answer = st.text_area("คำตอบของนักศึกษา:", height=100)
            correct_answer = st.text_area("เฉลย:", height=100)
            
            submitted = st.form_submit_button("🎯 ตรวจคำตอบ", type="primary")
            
            if submitted and all([question, student_answer, correct_answer]):
                with st.spinner("กำลังตรวจคำตอบ..."):
                    try:
                        grader = ExamGrader(st.session_state.rag_engine)
                        result = grader.grade_answer(
                            question=question,
                            student_answer=student_answer,
                            correct_answer=correct_answer
                        )
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            score = result['score']
                            color = "green" if score >= 70 else "orange" if score >= 50 else "red"
                            st.metric("คะแนน", f"{score}/100", delta=None)
                            
                        with col2:
                            status = "✅ ถูกต้อง" if result['is_correct'] else "❌ ไม่ถูกต้อง"
                            st.metric("สถานะ", status)
                        
                        st.subheader("📝 Feedback:")
                        st.write(result['feedback'])
                        
                        if result.get('key_points_covered'):
                            st.subheader("✅ จุดที่ตอบถูก:")
                            for point in result['key_points_covered']:
                                st.write(f"• {point}")
                        
                        if result.get('missing_points'):
                            st.subheader("❌ จุดที่ขาดหายไป:")
                            for point in result['missing_points']:
                                st.write(f"• {point}")
                    
                    except Exception as e:
                        st.error(f"เกิดข้อผิดพลาด: {e}")
    
    with tab2:
        st.subheader("ตรวจข้อสอบชุด")
        
        # File upload
        uploaded_file = st.file_uploader(
            "อัพโหลดไฟล์ข้อสอบ (JSON):", 
            type=['json']
        )
        
        if uploaded_file is not None:
            try:
                exam_data = json.loads(uploaded_file.getvalue().decode("utf-8"))
                st.success(f"โหลดข้อสอบสำเร็จ - {len(exam_data)} ข้อ")
                
                if st.button("🎯 เริ่มตรวจข้อสอบ", type="primary"):
                    with st.spinner("กำลังตรวจข้อสอบ..."):
                        try:
                            grader = ExamGrader(st.session_state.rag_engine)
                            results = grader.grade_exam(exam_data)
                            
                            # Display summary
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("คะแนนรวม", f"{results['total_score']}/{results['max_score']}")
                            
                            with col2:
                                st.metric("เปอร์เซ็นต์", f"{results['percentage']:.1f}%")
                            
                            with col3:
                                st.metric("ข้อที่ถูก", f"{results['summary']['num_correct']}/{results['num_questions']}")
                            
                            with col4:
                                pass_status = "✅ ผ่าน" if results['summary']['pass'] else "❌ ไม่ผ่าน"
                                st.metric("ผลการสอบ", pass_status)
                            
                            # Detailed results
                            st.subheader("📊 รายละเอียดแต่ละข้อ:")
                            for result in results['results']:
                                with st.expander(f"ข้อที่ {result['question_number']} - คะแนน: {result['score']}/100"):
                                    st.write(f"**คำถาม:** {result['question']}")
                                    st.write(f"**คำตอบนักศึกษา:** {result['student_answer']}")
                                    st.write(f"**เฉลย:** {result['correct_answer']}")
                                    st.write(f"**Feedback:** {result['feedback']}")
                            
                            # Download results
                            results_json = json.dumps(results, ensure_ascii=False, indent=2)
                            st.download_button(
                                label="📥 ดาวน์โหลดผลลัพธ์ (JSON)",
                                data=results_json,
                                file_name="exam_results.json",
                                mime="application/json"
                            )
                        
                        except Exception as e:
                            st.error(f"เกิดข้อผิดพลาดในการตรวจข้อสอบ: {e}")
            
            except Exception as e:
                st.error(f"ไม่สามารถอ่านไฟล์ได้: {e}")

def system_status_mode():
    """System Status Mode Interface"""
    st.header("📊 System Status - สถานะระบบ")
    
    try:
        # System information
        status = st.session_state.rag_engine.get_status()
        vector_info = st.session_state.vector_store.get_collection_info()
        
        # Display status
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("LLM Provider", status['llm_provider'].upper())
        
        with col2:
            st.metric("Vector Store", vector_info.get('name', 'N/A'))
        
        with col3:
            st.metric("Documents", vector_info.get('count', 0))
        
        # Detailed information
        st.subheader("🔧 System Details:")
        
        with st.expander("Vector Store Information"):
            st.json({
                "Collection Name": vector_info.get('name', 'N/A'),
                "Document Count": vector_info.get('count', 0),
                "Persist Directory": vector_info.get('persist_directory', 'N/A')
            })
        
        with st.expander("LLM Configuration"):
            st.json({
                "Provider": status['llm_provider'],
                "Embedding Model": status['embedding_model']
            })
        
        # Test system
        st.subheader("🧪 Quick System Test:")
        if st.button("🔍 ทดสอบระบบ"):
            with st.spinner("กำลังทดสอบ..."):
                try:
                    test_result = st.session_state.rag_engine.query("ทดสอบระบบ")
                    st.success("✅ ระบบทำงานปกติ")
                    st.write("**Test Response:**", test_result['answer'][:100] + "...")
                except Exception as e:
                    st.error(f"❌ พบปัญหา: {e}")
    
    except Exception as e:
        st.error(f"ไม่สามารถแสดงสถานะระบบได้: {e}")

if __name__ == "__main__":
    main()