from app.model import MentorAI

def predict():
    print("Loading Mentor AI model...")
    mentor = MentorAI()
    
    try:
        mentor.load_model()
        print("Model loaded successfully!\n")
    except FileNotFoundError:
        print("Model not found. Please run 'python -m app.train' first.")
        return
    
    print("=" * 60)
    print("MENTOR AI - Interactive Q&A Evaluation")
    print("=" * 60)
    print("\nCommands:")
    print("  - Type 'quit' to exit")
    print("  - Type 'search: <query>' to find similar questions")
    print("  - Or enter a question and answer separated by '|'\n")
    
    while True:
        user_input = input("\nEnter question|answer (or command): ").strip()
        
        if user_input.lower() == 'quit':
            print("Thank you for using Mentor AI!")
            break
        
        if user_input.lower().startswith('search:'):
            query = user_input[7:].strip()
            results = mentor.find_similar_questions(query, top_k=3)
            print(f"\nTop similar questions for: '{query}'")
            print("-" * 60)
            for i, result in enumerate(results, 1):
                print(f"\n{i}. [{result['topic'].upper()}] {result['question']}")
                print(f"   Answer: {result['answer']}")
                print(f"   Similarity: {result['similarity']:.2%}")
            continue
        
        if '|' not in user_input:
            print("Please use format: question|answer")
            continue
        
        parts = user_input.split('|', 1)
        question = parts[0].strip()
        answer = parts[1].strip()
        
        result = mentor.evaluate_answer(question, answer)
        
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Topic: {result['topic'].upper()}")
        print(f"Matched Question: {result['matched_question']}")
        print(f"\nQuestion Match Score: {result['question_match_score']:.2%}")
        print(f"Answer Match Score: {result['answer_match_score']:.2%}")
        print(f"Overall Score: {result['overall_score']:.2%}")
        print(f"\nFeedback: {result['feedback']}")
        print("=" * 60)

if __name__ == "__main__":
    predict()
