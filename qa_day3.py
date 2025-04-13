# qa_day3.py
# Question Answering System Using DistilBERT - Day 3
# Objective: Finalize the QA system with file-based context loading, evaluation, and robust error handling.

# Import required libraries
from transformers import pipeline
import os

# Initialize the QA pipeline with a pre-trained DistilBERT model
# Model: distilbert-base-cased-distilled-squad (fine-tuned for QA on SQuAD)
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Define a default context
default_context = "The Eiffel Tower was completed in 1889 and is located in Paris. It was designed by Gustave Eiffel and is a global cultural icon of France."

# Attempt to load context from file
context_file = "context.txt"
try:
    if os.path.exists(context_file):
        with open(context_file, 'r', encoding='utf-8') as f:
            context = f.read().strip()
        if not context:
            print("Warning: context.txt is empty. Using default context.")
            context = default_context
    else:
        print("Warning: context.txt not found. Using default context.")
        context = default_context
except Exception as e:
    print(f"Error loading context.txt: {e}. Using default context.")
    context = default_context

# Display the loaded context
print(f"\nLoaded context: {context}")

# Evaluation: Test set of questions
test_questions = [
    {"question": "Where is the Eiffel Tower located?", "expected": "Paris"},
    {"question": "When was the Eiffel Tower completed?", "expected": "1889"},
    {"question": "Who designed the Eiffel Tower?", "expected": "Gustave Eiffel"},
    {"question": "What is the Eiffel Tower a cultural icon of?", "expected": "France"},
    {"question": "What is the capital city mentioned?", "expected": "Paris"}
]

# Run evaluation
print("\nRunning evaluation with test questions:")
correct_count = 0
for i, test in enumerate(test_questions, 1):
    try:
        result = qa_pipeline(question=test["question"], context=context)
        answer = result["answer"]
        is_correct = answer.lower() == test["expected"].lower()
        correct_count += 1 if is_correct else 0
        print(f"Test Question {i}: {test['question']}")
        print(f"Answer: {answer} (Expected: {test['expected']}, Correct: {'Yes' if is_correct else 'No'})")
    except Exception as e:
        print(f"Error on Test Question {i}: {e}")
        print(f"Answer: None (Expected: {test['expected']}, Correct: No)")

# Print evaluation summary
print(f"\nEvaluation Summary: {correct_count}/{len(test_questions)} questions answered correctly.")

# Interactive loop for user questions
print("\nInteractive Mode:")
while True:
    # Prompt for a question
    print("\nEnter your question (or 'quit' to exit):")
    question = input().strip()

    # Check for exit condition
    if question.lower() == "quit":
        print("Exiting QA system. Goodbye!")
        break

    # Validate question input
    if not question:
        print("Error: Question cannot be empty. Please try again.")
        continue

    # Get the answer from the pipeline
    try:
        result = qa_pipeline(question=question, context=context)
        print(f"Question: {question}")
        print(f"Answer: {result['answer']}")
    except Exception as e:
        print(f"Error processing question: {e}. Please try a different question.")