from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
from app.model import MentorAI
from app.train import create_sample_data
import os
import io
import json
import pandas as pd

app = FastAPI(
    title="Mentor AI API",
    description="A similarity-based Q&A evaluation system that can be embedded in other websites",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mentor = MentorAI()
model_loaded = False

class EvaluationRequest(BaseModel):
    question: str
    answer: str

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class EvaluationResponse(BaseModel):
    question_match_score: float
    answer_match_score: float
    overall_score: float
    expected_answer: str
    feedback: str
    topic: str
    matched_question: str

@app.on_event("startup")
async def startup_event():
    global model_loaded
    try:
        if os.path.exists('data/mentor_model.pkl'):
            mentor.load_model()
            model_loaded = True
            print("✓ Mentor AI model loaded successfully")
        else:
            print("⚠ Model not found. Run training first: python -m app.train")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.get("/")
async def root():
    return FileResponse("index.html")

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "version": "1.0.0"
    }

@app.post("/api/evaluate", response_model=EvaluationResponse)
async def evaluate_answer(request: EvaluationRequest):
    if not model_loaded:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        result = mentor.evaluate_answer(request.question, request.answer)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search")
async def search_questions(request: SearchRequest):
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        top_k = request.top_k if request.top_k is not None else 5
        results = mentor.find_similar_questions(request.query, top_k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/topics")
async def get_topics():
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    topics = list(set([qa.get('topic', 'general') for qa in mentor.qa_database]))
    return {"topics": topics}

@app.post("/api/train")
async def train_endpoint():
    global model_loaded
    try:
        qa_data = create_sample_data()
        mentor.train(qa_data)
        mentor.save_model()
        model_loaded = True
        return {
            "status": "trained",
            "items": len(qa_data)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _validate_items(items: List[dict]) -> List[dict]:
    if not isinstance(items, list) or len(items) == 0:
        raise ValueError("Items must be a non-empty list")
    cleaned: List[dict] = []
    for row in items:
        if not isinstance(row, dict):
            continue
        q = str(row.get('question', '')).strip()
        a = str(row.get('answer', '')).strip()
        t = str(row.get('topic', 'general')).strip() or 'general'
        if q and a:
            cleaned.append({"question": q, "answer": a, "topic": t})
    if not cleaned:
        raise ValueError("No valid items with question and answer fields")
    return cleaned

@app.post("/api/data")
async def upload_data_json(items: List[dict]):
    """Accepts JSON body: an array of {question, answer, topic} and retrains the model."""
    global model_loaded
    try:
        cleaned = _validate_items(items)
        mentor.train(cleaned)
        mentor.save_model()
        model_loaded = True
        return {"status": "trained", "items": len(cleaned)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/data/upload")
async def upload_data_file(file: UploadFile = File(...)):
    """Accepts a CSV or JSON file and retrains the model.
    CSV must contain columns: question, answer, [topic]. JSON can be list of objects.
    """
    global model_loaded
    try:
        content = await file.read()
        filename = (file.filename or '').lower()
        items: List[dict] = []
        if filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
            if 'question' not in df.columns or 'answer' not in df.columns:
                raise ValueError("CSV must have 'question' and 'answer' columns")
            if 'topic' not in df.columns:
                df['topic'] = 'general'
            items = df[['question', 'answer', 'topic']].to_dict(orient='records')
        elif filename.endswith('.json') or 'json' in (file.content_type or ''):
            parsed = json.loads(content.decode('utf-8'))
            if isinstance(parsed, dict) and 'items' in parsed:
                parsed = parsed['items']
            items = parsed
        else:
            # try JSON fallback
            try:
                parsed = json.loads(content.decode('utf-8'))
                items = parsed
            except Exception:
                raise ValueError("Unsupported file type. Use CSV or JSON")

        cleaned = _validate_items(items)
        mentor.train(cleaned)
        mentor.save_model()
        model_loaded = True
        return {"status": "trained", "items": len(cleaned)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
