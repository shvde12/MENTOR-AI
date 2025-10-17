import json
from app.model import MentorAI

def create_sample_data():
    qa_data = [
        {
            "topic": "python",
            "question": "What is a list in Python?",
            "answer": "A list is a built-in data structure in Python that stores ordered collections of items. Lists are mutable, meaning you can modify them after creation. They are created using square brackets []."
        },
        {
            "topic": "python",
            "question": "How do you create a function in Python?",
            "answer": "You create a function in Python using the 'def' keyword, followed by the function name and parentheses containing parameters. Use a colon and indent the function body."
        },
        {
            "topic": "python",
            "question": "What is the difference between a list and a tuple?",
            "answer": "Lists are mutable (can be changed) and use square brackets []. Tuples are immutable (cannot be changed) and use parentheses (). Lists are generally used for homogeneous items, tuples for heterogeneous data."
        },
        {
            "topic": "python",
            "question": "How do you handle exceptions in Python?",
            "answer": "Use try-except blocks to handle exceptions in Python. The code that might raise an exception goes in the try block, and the error handling code goes in the except block."
        },
        {
            "topic": "python",
            "question": "What is a dictionary in Python?",
            "answer": "A dictionary is a built-in data structure that stores key-value pairs. Dictionaries are mutable and unordered. They are created using curly braces {} with key:value pairs."
        },
        {
            "topic": "math",
            "question": "What is the Pythagorean theorem?",
            "answer": "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of squares of the other two sides: a² + b² = c²"
        },
        {
            "topic": "math",
            "question": "What is the derivative of x²?",
            "answer": "The derivative of x² is 2x. This follows the power rule where d/dx(x^n) = n*x^(n-1)."
        },
        {
            "topic": "math",
            "question": "What is a prime number?",
            "answer": "A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. Examples include 2, 3, 5, 7, 11, etc."
        },
        {
            "topic": "math",
            "question": "How do you calculate the area of a circle?",
            "answer": "The area of a circle is calculated using the formula A = πr², where r is the radius of the circle and π (pi) is approximately 3.14159."
        },
        {
            "topic": "math",
            "question": "What is the quadratic formula?",
            "answer": "The quadratic formula is x = (-b ± √(b²-4ac)) / 2a. It's used to solve quadratic equations of the form ax² + bx + c = 0."
        }
    ]
    
    with open('data/training_data.json', 'w') as f:
        json.dump(qa_data, f, indent=2)
    
    return qa_data

def train_model():
    print("Creating sample training data...")
    qa_data = create_sample_data()
    
    print("Initializing Mentor AI model...")
    mentor = MentorAI()
    
    print("Training model...")
    mentor.train(qa_data)
    
    print("Saving model...")
    mentor.save_model()
    
    print(f"Training complete! Model trained on {len(qa_data)} Q&A pairs.")
    print("Topics covered: Python programming, Mathematics")
    
if __name__ == "__main__":
    train_model()
