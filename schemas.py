from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from enum import Enum

# --- Enums ---
class QuestionType(str, Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    FILL_IN_BLANK = "fill_in_blank"

# --- Request Models ---
class InitialQuizRequest(BaseModel):
    class_id: int
    unit_id: int
    lessons: List[int]
    questions_per_lesson: int = 9

class LessonQuizRequest(BaseModel):
    class_id: int
    session_id: int
    lesson_id: int
    questions_per_lesson: int = 9

class ChatRequest(BaseModel):
    student_id: int
    session_id: int
    lesson_id: int
    class_id: int
    question: str
    conversation_id: Optional[int] = None
    previous_messages: List[dict] = []     
    topics_covered_so_far: List[str] = []

class AdaptiveExplanationRequest(BaseModel):
    student_id: int
    session_id: int
    lesson_id: int
    concept: str
    skill_id: int

# --- Response Models ---
class QuestionGenerated(BaseModel):
    question_text: str
    correct_answer: int 
    options: List[str]
    hint: str
    bottom_hint: str
    difficulty: int
    type: QuestionType
    skill_id: int

    @field_validator('correct_answer')
    @classmethod
    def validate_index(cls, v: int, info):
        if v < 0: return 0
        return v

class QuizResponse(BaseModel):
    success: bool
    questions: List[QuestionGenerated]
    total_questions: int

class ChatResponse(BaseModel):
    answer: str
    topics_covered: List[str]
    all_topics_covered: bool