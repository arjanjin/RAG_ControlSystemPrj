"""
Exam Grader Module
โมดูลสำหรับตรวจข้อสอบโดยใช้ RAG
"""

from typing import Dict, List, Any, Optional
import logging
import json

import config
from src.rag_engine import RAGEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExamGrader:
    """
    คลาสสำหรับตรวจข้อสอบโดยใช้ RAG
    """

    def __init__(self, rag_engine: Optional[RAGEngine] = None):
        """
        Initialize exam grader

        Args:
            rag_engine: RAG Engine (ถ้าไม่ระบุจะสร้างใหม่)
        """
        self.rag_engine = rag_engine or RAGEngine()

    def grade_answer(
        self,
        question: str,
        student_answer: str,
        correct_answer: str,
        max_score: int = 100
    ) -> Dict[str, Any]:
        """
        ตรวจและให้คะแนนคำตอบ

        Args:
            question: คำถาม
            student_answer: คำตอบของนักศึกษา
            correct_answer: เฉลย/คำตอบที่ถูกต้อง
            max_score: คะแนนเต็ม (default: 100)

        Returns:
            Dict: ผลการตรวจ
        """
        # ค้นหา context ที่เกี่ยวข้อง
        context_docs = self.rag_engine.retrieve_context(question)
        context = self.rag_engine.format_context(context_docs)

        # สร้าง prompt
        prompt = config.GRADING_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
            student_answer=student_answer,
            correct_answer=correct_answer
        )

        try:
            # ส่งไปให้ LLM ตรวจ
            response = self.rag_engine.llm.invoke(prompt)

            # พยายาม parse JSON response
            result = self._parse_grading_response(response, max_score)
            result['question'] = question
            result['student_answer'] = student_answer
            result['correct_answer'] = correct_answer

            logger.info(f"Graded question - Score: {result['score']}/{max_score}")
            return result

        except Exception as e:
            logger.error(f"Error grading answer: {e}")
            return {
                'question': question,
                'student_answer': student_answer,
                'correct_answer': correct_answer,
                'score': 0,
                'is_correct': False,
                'feedback': f"เกิดข้อผิดพลาดในการตรวจ: {str(e)}",
                'error': True
            }

    def _parse_grading_response(self, response: str, max_score: int) -> Dict[str, Any]:
        """
        แปลง response จาก LLM เป็น dict

        Args:
            response: response จาก LLM
            max_score: คะแนนเต็ม

        Returns:
            Dict: ผลการตรวจ
        """
        try:
            # ลอง parse JSON
            # ลบ code blocks ถ้ามี
            response = response.strip()
            if response.startswith('```'):
                response = response.split('```')[1]
                if response.startswith('json'):
                    response = response[4:]

            result = json.loads(response)

            # ปรับคะแนนให้เป็น scale ที่ต้องการ
            if 'score' in result:
                result['score'] = min(max_score, int(result['score'] * max_score / 100))

            return result

        except json.JSONDecodeError:
            # ถ้า parse ไม่ได้ ให้แยกข้อมูลจาก text
            logger.warning("Could not parse JSON, using fallback parsing")

            score = 0
            is_correct = False
            feedback = response

            # ลองหาคะแนนจาก text
            if 'score' in response.lower():
                try:
                    import re
                    score_match = re.search(r'score["\s:]+(\d+)', response, re.IGNORECASE)
                    if score_match:
                        score = min(max_score, int(score_match.group(1)))
                except:
                    pass

            return {
                'score': score,
                'is_correct': is_correct,
                'feedback': feedback,
                'key_points_covered': [],
                'missing_points': []
            }

    def grade_exam(
        self,
        exam_data: List[Dict[str, str]],
        max_score_per_question: int = 100
    ) -> Dict[str, Any]:
        """
        ตรวจข้อสอบทั้งชุด

        Args:
            exam_data: รายการข้อสอบ [{'question': ..., 'student_answer': ..., 'correct_answer': ...}]
            max_score_per_question: คะแนนเต็มต่อข้อ

        Returns:
            Dict: ผลการตรวจทั้งหมด
        """
        results = []
        total_score = 0
        max_total_score = len(exam_data) * max_score_per_question

        logger.info(f"Grading exam with {len(exam_data)} questions")

        for i, item in enumerate(exam_data, 1):
            logger.info(f"Grading question {i}/{len(exam_data)}")

            result = self.grade_answer(
                question=item.get('question', ''),
                student_answer=item.get('student_answer', ''),
                correct_answer=item.get('correct_answer', ''),
                max_score=max_score_per_question
            )

            result['question_number'] = i
            results.append(result)
            total_score += result.get('score', 0)

        percentage = (total_score / max_total_score * 100) if max_total_score > 0 else 0

        return {
            'results': results,
            'total_score': total_score,
            'max_score': max_total_score,
            'percentage': round(percentage, 2),
            'num_questions': len(exam_data),
            'summary': self._generate_summary(results)
        }

    def _generate_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """
        สร้างสรุปผลการตรวจ

        Args:
            results: รายการผลการตรวจ

        Returns:
            Dict: สรุปผล
        """
        num_correct = sum(1 for r in results if r.get('is_correct', False))
        num_questions = len(results)
        avg_score = sum(r.get('score', 0) for r in results) / num_questions if num_questions > 0 else 0

        return {
            'num_correct': num_correct,
            'num_incorrect': num_questions - num_correct,
            'average_score': round(avg_score, 2),
            'pass': avg_score >= 50  # ผ่านถ้าเฉลี่ยมากกว่า 50
        }

    def generate_feedback_report(self, grading_results: Dict[str, Any]) -> str:
        """
        สร้างรายงาน feedback แบบละเอียด

        Args:
            grading_results: ผลการตรวจจาก grade_exam

        Returns:
            str: รายงาน feedback
        """
        report = []
        report.append("=" * 60)
        report.append("รายงานผลการตรวจข้อสอบวิชาระบบควบคุม")
        report.append("=" * 60)
        report.append("")

        # สรุปผล
        report.append(f"คะแนนรวม: {grading_results['total_score']}/{grading_results['max_score']}")
        report.append(f"เปอร์เซ็นต์: {grading_results['percentage']}%")
        report.append(f"จำนวนข้อที่ถูก: {grading_results['summary']['num_correct']}/{grading_results['num_questions']}")
        report.append(f"ผลการสอบ: {'ผ่าน' if grading_results['summary']['pass'] else 'ไม่ผ่าน'}")
        report.append("")
        report.append("=" * 60)
        report.append("รายละเอียดแต่ละข้อ")
        report.append("=" * 60)
        report.append("")

        # รายละเอียดแต่ละข้อ
        for result in grading_results['results']:
            report.append(f"ข้อที่ {result['question_number']}")
            report.append(f"คำถาม: {result['question']}")
            report.append(f"คะแนน: {result['score']}")
            report.append(f"สถานะ: {'ถูกต้อง' if result.get('is_correct') else 'ไม่ถูกต้อง'}")
            report.append(f"\nคำตอบของนักศึกษา:\n{result['student_answer']}")
            report.append(f"\nเฉลย:\n{result['correct_answer']}")
            report.append(f"\nคำแนะนำ:\n{result['feedback']}")

            if result.get('key_points_covered'):
                report.append(f"\nจุดที่ตอบถูก:")
                for point in result['key_points_covered']:
                    report.append(f"  - {point}")

            if result.get('missing_points'):
                report.append(f"\nจุดที่ควรเพิ่มเติม:")
                for point in result['missing_points']:
                    report.append(f"  - {point}")

            report.append("\n" + "-" * 60 + "\n")

        return "\n".join(report)


def create_exam_grader() -> ExamGrader:
    """
    สร้าง ExamGrader พร้อมใช้งาน

    Returns:
        ExamGrader: exam grader ที่พร้อมใช้งาน
    """
    rag_engine = RAGEngine()
    rag_engine.load_knowledge_base()
    return ExamGrader(rag_engine)
