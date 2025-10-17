# MENTOR-AI

Simple similarity-based mentor AI (question-answer evaluator)

## Overview

Mentor AI is an intelligent Q&A evaluation system that uses neural network-based similarity matching to help users learn various topics. The system evaluates user answers against a trained knowledge base and provides detailed feedback.

## Features

- 🧠 Similarity-based answer evaluation using sentence transformers
- 📚 Multi-topic support (Python, Mathematics, and more)
- 🔍 Semantic question search
- 🎯 Detailed scoring and feedback
- 🌐 RESTful API for easy integration
- 💻 Clean HTML5 demo interface

## Setup

1. Install dependencies (automatically handled):
```bash
uv sync
```

2. Train the model:
```bash
python -m app.train
```

3. Run the API server:
```bash
uvicorn app.api:app --host 0.0.0.0 --port 5000 --reload
```

## Usage

### Web Interface
Open your browser and navigate to `http://localhost:5000`

### CLI Prediction
```bash
python -m app.predict
```

### API Endpoints

- `GET /api/health` - Check API status
- `POST /api/evaluate` - Evaluate a question-answer pair
- `POST /api/search` - Search for similar questions
- `GET /api/topics` - Get available topics

## API Integration Example

```javascript
// Evaluate an answer
const response = await fetch('http://your-domain/api/evaluate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        question: "What is a list in Python?",
        answer: "A list is a collection of items in Python"
    })
});

const result = await response.json();
console.log(result.overall_score);
```

## Project Structure

```
├── app/
│   ├── __init__.py      # Package initialization
│   ├── model.py         # Core ML model and logic
│   ├── train.py         # Training pipeline
│   ├── predict.py       # CLI prediction tool
│   └── api.py           # FastAPI application
├── data/                # Training data and models
├── tests/               # Test suite
└── index.html          # Demo frontend
```

## Topics Covered

- Python Programming
- Mathematics

More topics can be easily added by extending the training data.

## Technology Stack

- **Backend**: FastAPI, Python 3.11
- **ML**: sentence-transformers, scikit-learn
- **Frontend**: HTML5, Vanilla JavaScript
