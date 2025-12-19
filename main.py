import os
import json
import chromadb
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import ValidationError
from typing import List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential

from schemas import (
    InitialQuizRequest, 
    LessonQuizRequest,
    QuestionGenerated, 
    QuizResponse, 
    ChatRequest, 
    ChatResponse,
    AdaptiveExplanationRequest
)

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ChromaDB Setup
chroma_client = chromadb.PersistentClient(path="./my_chroma_db")
collection = chroma_client.get_collection(name="unit1_math_content")

app = FastAPI(title="Edutera RAG V3 (Clean Data)")

# Mock Function for BKT Mastery
def get_student_mastery_mock(student_token: str, skill_id: int):
    # This will be replaced by a real backend call later
    return {"mastery_after": 0.5, "mastery_level": "learning"}

# --- 1. STRONG ENGLISH PROMPT ---
def get_quiz_prompt(content_context, skill_info, num_questions=3, difficulty="medium"):
    diff_map = {"easy": 1, "medium": 2, "hard": 3}
    diff_num = diff_map.get(difficulty, 2)
    
    return f"""You are an expert Math Teacher specialized in creating exam questions.

**Task:** Create {num_questions} questions with '{difficulty}' difficulty.

**Source Material (Context):**
{content_context[:2500]}

**Target Skill:**
- ID: {skill_info['skill_id']}
- Name: {skill_info['skill_name']}

**⚠️ CRITICAL JSON REQUIREMENTS:**
You must output a strictly valid JSON object. 
Each question object MUST have exactly these 8 fields:

{{
  "question_text": "The question text in ARABIC",
  "correct_answer": 0, // Integer index (0-3) of the correct option
  "options": ["Option 1 in Arabic", "Option 2 in Arabic", "Option 3", "Option 4"],
  "hint": "A helpful hint in Arabic",
  "bottom_hint": "The answer revealer in Arabic",
  "difficulty": {diff_num},
  "type": "multiple_choice", // One of: "multiple_choice", "true_false", "fill_in_blank"
  "skill_id": {skill_info['skill_id']}
}}

**Rules:**
1. Mix question types: "multiple_choice", "true_false", "fill_in_blank".
2. JSON keys must be in English.
3. Values (Text) must be in ARABIC.
4. If the context contains examples, try to create similar questions but with different numbers.
5. Return ONLY the JSON object.

**Output Format:**
{{
  "questions": [ ... ]
}}
"""

# --- 2. Retry Logic ---
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_questions_safe(prompt: str) -> List[QuestionGenerated]:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.7
        )
        data = json.loads(response.choices[0].message.content)
        
        valid_questions = []
        for q in data.get("questions", []):
            try:
                valid_q = QuestionGenerated(**q)
                valid_questions.append(valid_q)
            except ValidationError as ve:
                continue
        return valid_questions
    except Exception as e:
        raise e 

# --- 3. API Endpoints ---

@app.get("/rag/health")
def health_check():
    return {"status": "healthy"}

@app.post("/rag/quizzes/generate-initial", response_model=QuizResponse)
def generate_initial_quiz(request: InitialQuizRequest):
    all_questions = []
    
    for lesson_id in request.lessons:
        # Fetch content
        results = collection.get(where={"lesson_id": lesson_id}, include=["metadatas", "documents"])
        if not results['ids']: continue

        # Group by Skill
        skills_map = {}
        for doc, meta in zip(results['documents'], results['metadatas']):
            s_id = meta['skill_id']
            if s_id not in skills_map:
                skills_map[s_id] = {"name": meta['skill_name'], "content": ""}
            skills_map[s_id]["content"] += "\n" + doc

        # Select the richest skill
        if not skills_map: continue
        target_skill_id = list(skills_map.keys())[0]
        skill_data = skills_map[target_skill_id]

        # Generate per difficulty
        for difficulty in ["easy", "medium", "hard"]:
            prompt = get_quiz_prompt(
                content_context=skill_data['content'],
                skill_info={"skill_id": target_skill_id, "skill_name": skill_data['name']},
                num_questions=3,
                difficulty=difficulty
            )
            try:
                new_qs = generate_questions_safe(prompt)
                all_questions.extend(new_qs)
            except Exception:
                pass

    return {
        "success": True,
        "questions": all_questions,
        "total_questions": len(all_questions)
    }

@app.post("/rag/quizzes/generate-lesson", response_model=QuizResponse)
def generate_lesson_quiz(request: LessonQuizRequest):
    temp_req = InitialQuizRequest(
        class_id=request.class_id,
        unit_id=1,
        lessons=[request.lesson_id],
        questions_per_lesson=request.questions_per_lesson
    )
    return generate_initial_quiz(temp_req)

# --- THE MISSING ENDPOINT (FIXED) ---
@app.post("/rag/tutor/answer", response_model=ChatResponse)
def answer_student_question(request: ChatRequest, authorization: str = Header(None)):
    # 1. Simulate Auth Check
    token = authorization.split(" ")[1] if authorization else "mock_token"

    # 2. Retrieve Context
    results = collection.query(
        query_texts=[request.question],
        n_results=3,
        where={"lesson_id": request.lesson_id}
    )
    context = "\n".join(results['documents'][0]) if results['documents'] else "No context available."

    # 3. Prompt
    system_prompt = f"""You are an AI Math Tutor.
    Context from curriculum: {context[:2000]}
    
    Instructions:
    - Answer the student's question in Arabic based ONLY on the context.
    - Be helpful and encouraging.
    """
    
    # 4. Generate
    messages = [{"role": "system", "content": system_prompt}]
    # Add history
    for msg in request.previous_messages[-3:]:
        messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
    messages.append({"role": "user", "content": request.question})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.5
    )
    
    answer_text = response.choices[0].message.content

    return {
        "answer": answer_text,
        "topics_covered": request.topics_covered_so_far, # Placeholder
        "all_topics_covered": False
    }

@app.post("/rag/tutor/explain")
def explain_concept(request: AdaptiveExplanationRequest, authorization: str = Header(None)):
    token = authorization.split(" ")[1] if authorization else "mock"
    bkt = get_student_mastery_mock(token, request.skill_id)
    mastery = bkt['mastery_after']

    style = "Explain simply with daily life examples." if mastery < 0.4 else "Explain normally."
    
    results = collection.query(query_texts=[request.concept], n_results=2, where={"skill_id": request.skill_id})
    context = "\n".join(results['documents'][0]) if results['documents'] else ""

    prompt = f"Role: Tutor. Concept: {request.concept}. Student Level: {mastery}. Style: {style}. Context: {context}. Explain in Arabic."
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return {"explanation": response.choices[0].message.content}