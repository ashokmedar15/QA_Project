# qa_day1.py
# Question Answering System Using DistilBERT - Day 1
# Objective: Build a basic QA system using a pre-trained DistilBERT model to answer questions from a context.

# Import the required library
from transformers import pipeline

# Define a sample context (hardcoded paragraph for testing)
context = "The Eiffel Tower was completed in 1889 and is located in Paris."

# Initialize the QA pipeline with a pre-trained DistilBERT model
# Model: distilbert-base-cased-distilled-squad (fine-tuned for QA on SQuAD)
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Define a test question
question = "Where is the Eiffel Tower located?"

# Get the answer from the pipeline
# Input: question and context; Output: dictionary with answer and score
result = qa_pipeline(question=question, context=context)

# Extract and print the answer
print(f"Question: {question}")
print(f"Answer: {result['answer']}")