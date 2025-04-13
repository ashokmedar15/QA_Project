# qa_day2.py
# Question Answering System Using DistilBERT - Day 2
# Objective: Enhance the QA system to support interactive question input, custom contexts, and error handling.

# Import the required library
from transformers import pipeline

# Initialize the QA pipeline with a pre-trained DistilBERT model
# Model: distilbert-base-cased-distilled-squad (fine-tuned for QA on SQuAD)
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Define a default context
default_context = "The Eiffel Tower was completed in 1889 and is located in Paris."

# Prompt user for a context (with option to use default)
print("Enter a context paragraph (or press Enter to use the default context):")
user_context = input().strip()

# Use default context if user input is empty
context = default_context if not user_context else user_context

# Display the current context
print(f"\nCurrent context: {context}")

# Interactive loop for questions
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
        # Print the question and answer
        print(f"Question: {question}")
        print(f"Answer: {result['answer']}")
    except Exception as e:
        print(f"Error processing question: {e}. Please try a different question.")