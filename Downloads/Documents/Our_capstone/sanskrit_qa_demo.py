#!/usr/bin/env python3
"""
Sanskrit QA System Interactive Demo

This script provides an interactive interface to test the Sanskrit Question Answering system.
Users can input Sanskrit text and questions to get contextual answers.

Usage:
    python sanskrit_qa_demo.py

Author: Sanskrit QA Team
Date: August 2025
"""

import sys
import os
import torch
from pathlib import Path

# Add the current directory to Python path to import sanskrit_qa_system
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from sanskrit_qa_system import SanskritQuestionAnsweringSystem
except ImportError as e:
    print(f"Error importing Sanskrit QA system: {e}")
    print("Please ensure sanskrit_qa_system.py is in the same directory.")
    sys.exit(1)


class SanskritQADemo:
    def __init__(self):
        """Initialize the demo with the QA system."""
        self.qa_system = None
        self.initialize_system()
    
    def initialize_system(self):
        """Initialize the Sanskrit QA system."""
        print("🕉️  Sanskrit Question Answering System Demo")
        print("=" * 50)
        print("Initializing the system... This may take a moment.")
        
        try:
            # Initialize with reasonable parameters for demo
            self.qa_system = SanskritQuestionAnsweringSystem(
                vocab_size=5000,
                embedding_dim=512,
                num_layers=8,
                num_heads=8,
                dropout_rate=0.1
            )
            print("✅ System initialized successfully!")
        except Exception as e:
            print(f"❌ Error initializing system: {e}")
            sys.exit(1)
    
    def display_welcome_message(self):
        """Display welcome message and instructions."""
        print("\n" + "=" * 60)
        print("🕉️  WELCOME TO SANSKRIT QA SYSTEM DEMO")
        print("=" * 60)
        print("This system can answer questions about Sanskrit texts,")
        print("particularly from classical literature like Kumārasaṃbhava.")
        print("\nSample text: 'asty uttarasyāṃ diśi devatātmā himālayo nāma nagādhirājaḥ'")
        print("Sample questions:")
        print("- What is the main subject in this verse?")
        print("- Who are the characters mentioned?")
        print("- What action is being described?")
        print("- What is the sentiment (rasa) of this text?")
        print("- Describe the grammatical structure.")
        print("\n" + "=" * 60)
    
    def get_user_input(self, prompt):
        """Get user input with error handling."""
        try:
            return input(prompt).strip()
        except KeyboardInterrupt:
            print("\n\n👋 Thank you for using Sanskrit QA Demo!")
            sys.exit(0)
        except EOFError:
            return ""
    
    def format_answer(self, answer):
        """Format the answer for better display."""
        if not answer:
            return "❌ No answer generated."
        
        # Split long answers into multiple lines for better readability
        formatted_lines = []
        current_line = ""
        words = answer.split()
        
        for word in words:
            if len(current_line + " " + word) <= 80:
                current_line += (" " + word if current_line else word)
            else:
                if current_line:
                    formatted_lines.append(current_line)
                current_line = word
        
        if current_line:
            formatted_lines.append(current_line)
        
        return "\n".join(formatted_lines)
    
    def process_qa_pair(self, context, question):
        """Process a single question-answer pair."""
        try:
            print("\n📝 Processing your question...")
            qa_pairs = self.qa_system.create_qa_pairs(context)
            
            # Find the best matching QA pair or generate a new one
            best_answer = None
            best_score = 0
            
            # Check if any existing QA pair matches the question
            for qa_pair in qa_pairs:
                if self.question_similarity(question, qa_pair['question']) > 0.5:
                    best_answer = qa_pair['answer']
                    break
            
            # If no good match, generate answer using the system's method
            if not best_answer:
                best_answer = self.qa_system._generate_answer(context, question)
            
            return best_answer
            
        except Exception as e:
            return f"Error processing question: {str(e)}"
    
    def question_similarity(self, q1, q2):
        """Simple similarity check between questions."""
        q1_words = set(q1.lower().split())
        q2_words = set(q2.lower().split())
        
        if not q1_words or not q2_words:
            return 0.0
        
        intersection = q1_words & q2_words
        union = q1_words | q2_words
        
        return len(intersection) / len(union) if union else 0.0
    
    def run_demo(self):
        """Run the interactive demo."""
        self.display_welcome_message()
        
        while True:
            print("\n" + "-" * 50)
            print("📖 Enter Sanskrit text (or 'quit' to exit):")
            context = self.get_user_input("Context: ")
            
            if context.lower() in ['quit', 'exit', 'q']:
                break
            
            if not context:
                print("⚠️  Please enter some Sanskrit text.")
                continue
            
            print(f"\n📝 Context received: {context}")
            
            # Allow multiple questions for the same context
            while True:
                print("\n❓ Enter your question (or 'new' for new context, 'quit' to exit):")
                question = self.get_user_input("Question: ")
                
                if question.lower() in ['quit', 'exit', 'q']:
                    return
                
                if question.lower() in ['new', 'n']:
                    break
                
                if not question:
                    print("⚠️  Please enter a question.")
                    continue
                
                print(f"\n❓ Question: {question}")
                print("🤔 Analyzing...")
                
                # Process the QA pair
                answer = self.process_qa_pair(context, question)
                
                print(f"\n✨ Answer:")
                print(f"💡 {self.format_answer(answer)}")
                
                # Show additional analysis
                print(f"\n📊 Additional Analysis:")
                qa_pairs = self.qa_system.create_qa_pairs(context)
                
                if qa_pairs:
                    print(f"   Generated {len(qa_pairs)} QA pairs for this context")
                    
                    # Show sentiment analysis if available
                    for qa_pair in qa_pairs:
                        if 'sentiment' in qa_pair['question'].lower() or 'rasa' in qa_pair['question'].lower():
                            print(f"   🎭 Detected sentiment: {qa_pair['answer']}")
                            break
    
    def run_batch_demo(self):
        """Run a batch demo with predefined examples."""
        print("\n🚀 Running batch demo with predefined examples...")
        
        test_cases = [
            {
                "context": "asty uttarasyāṃ diśi devatātmā himālayo nāma nagādhirājaḥ",
                "questions": [
                    "What is the main subject in this verse?",
                    "Who are the characters mentioned?",
                    "What action is being described?",
                    "What is the sentiment of this text?"
                ]
            },
            {
                "context": "śrī ganeśāya namaḥ",
                "questions": [
                    "Who is being invoked?",
                    "What is the purpose of this verse?"
                ]
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{'='*20} Example {i} {'='*20}")
            print(f"📖 Context: {test_case['context']}")
            
            for question in test_case['questions']:
                print(f"\n❓ Question: {question}")
                answer = self.process_qa_pair(test_case['context'], question)
                print(f"💡 Answer: {self.format_answer(answer)}")
        
        print("\n✅ Batch demo completed!")


def main():
    """Main function to run the demo."""
    try:
        demo = SanskritQADemo()
        
        print("\nChoose demo mode:")
        print("1. Interactive demo (type your own questions)")
        print("2. Batch demo (predefined examples)")
        print("3. Both")
        
        choice = input("\nEnter your choice (1/2/3): ").strip()
        
        if choice == "1":
            demo.run_demo()
        elif choice == "2":
            demo.run_batch_demo()
        elif choice == "3":
            demo.run_batch_demo()
            print("\n" + "="*50)
            print("Now starting interactive demo...")
            demo.run_demo()
        else:
            print("Invalid choice. Starting interactive demo...")
            demo.run_demo()
            
    except KeyboardInterrupt:
        print("\n\n👋 Thank you for using Sanskrit QA Demo!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please check that all dependencies are installed and try again.")


if __name__ == "__main__":
    main()
