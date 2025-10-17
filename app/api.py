from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
from app.model import MentorAI
import os

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
