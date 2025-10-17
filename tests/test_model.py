import pytest
from app.model import MentorAI

def test_mentor_initialization():
    mentor = MentorAI()
    assert mentor.model is not None
    assert mentor.qa_database == []
    assert mentor.embeddings is None

def test_training():
    mentor = MentorAI()
    qa_data = [
        {
            "topic": "test",
            "question": "What is testing?",
            "answer": "Testing is the process of verifying software works correctly."
        }
    ]
    mentor.train(qa_data)
    assert len(mentor.qa_database) == 1
    assert mentor.embeddings is not None
    assert mentor.embeddings.shape[0] == 1
