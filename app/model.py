import pickle
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os

class MentorAI:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.qa_database = []
        self.embeddings = None
        
    def load_data(self, filepath: str):
        with open(filepath, 'r') as f:
            self.qa_database = json.load(f)
        
    def train(self, qa_data: List[Dict] | None = None):
        if qa_data is not None:
            self.qa_database = qa_data
            
        if not self.qa_database:
            raise ValueError("No training data available")
        
        questions = [item['question'] for item in self.qa_database]
        self.embeddings = self.model.encode(questions)
        
    def save_model(self, filepath: str = 'data/mentor_model.pkl'):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        model_data = {
            'qa_database': self.qa_database,
            'embeddings': self.embeddings
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str = 'data/mentor_model.pkl'):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.qa_database = model_data['qa_database']
        self.embeddings = model_data['embeddings']
    
    def evaluate_answer(self, question: str, user_answer: str, top_k: int = 3) -> Dict:
        question_embedding = self.model.encode([question])
        
        similarities = cosine_similarity(question_embedding, self.embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        best_match = self.qa_database[top_indices[0]]
        similarity_score = float(similarities[top_indices[0]])
        
        answer_similarity = self._compare_answers(user_answer, best_match['answer'])
        
        feedback = self._generate_feedback(
            similarity_score, 
            answer_similarity, 
            best_match['answer'],
            user_answer
        )
        
        return {
            'question_match_score': similarity_score,
            'answer_match_score': answer_similarity,
            'overall_score': (similarity_score + answer_similarity) / 2,
            'expected_answer': best_match['answer'],
            'feedback': feedback,
            'topic': best_match.get('topic', 'general'),
            'matched_question': best_match['question']
        }
    
    def _compare_answers(self, user_answer: str, correct_answer: str) -> float:
        user_embedding = self.model.encode([user_answer])
        correct_embedding = self.model.encode([correct_answer])
        similarity = cosine_similarity(user_embedding, correct_embedding)[0][0]
        return float(similarity)
    
    def _generate_feedback(self, q_score: float, a_score: float, correct_answer: str, user_answer: str) -> str:
        if a_score >= 0.8:
            return f"Excellent! Your answer is very accurate. {correct_answer}"
        elif a_score >= 0.6:
            return f"Good attempt! Your answer is mostly correct. The ideal answer would be: {correct_answer}"
        elif a_score >= 0.4:
            return f"Partial understanding shown. Consider: {correct_answer}"
        else:
            return f"Let's review this topic. The correct answer is: {correct_answer}"
    
    def find_similar_questions(self, query: str, top_k: int = 5) -> List[Dict]:
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'question': self.qa_database[idx]['question'],
                'answer': self.qa_database[idx]['answer'],
                'topic': self.qa_database[idx].get('topic', 'general'),
                'similarity': float(similarities[idx])
            })
        return results
