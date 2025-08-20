#!/usr/bin/env python3
"""
Simple Sanskrit QA Demo
======================

This demo uses the enhanced Sanskrit QA functionality from the dataset loader
without requiring complex model architecture loading.
"""

import sys
import os
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

# Import from sanskrit_qa_system
try:
    from sanskrit_qa_system import SanskritDatasetLoader
    print("✅ Successfully imported Sanskrit QA functionality")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure sanskrit_qa_system.py is in the same directory")
    sys.exit(1)

class SimpleSanskritQADemo:
    """Simple demo using the enhanced answer generation from SanskritDatasetLoader."""
    
    def __init__(self):
        self.dataset_loader = SanskritDatasetLoader()
        print("✅ Sanskrit QA Demo initialized successfully!")
    
    def answer_question(self, text, question):
        """Generate answer using the enhanced QA functionality."""
        try:
            # Convert text to list format expected by the answer generator
            sentence = text.split()
            context = text
            
            # Use the enhanced answer generation method
            answer = self.dataset_loader._generate_answer(sentence, question, context)
            return answer
        except Exception as e:
            return f"❌ Error generating answer: {e}"
    
    def run_test_questions(self):
        """Run predefined test questions on Kumārasaṃbhava verse."""
        
        # Famous opening verse of Kumārasaṃbhava
        test_context = "asty uttarasyāṃ diśi devatātmā himālayo nāma nagādhirājaḥ"
        
        test_questions = [
            "What is the main subject in this verse?",
            "Who are the characters mentioned?", 
            "What action is being described?",
            "What is the sentiment of this text?",
            "What divine entities are present?",
            "What locations are mentioned?",
            "What grammatical features are present?",
            "What relationships are implied?"
        ]
        
        print(f"\n{Fore.CYAN}📖 Test Context from Kumārasaṃbhava:")
        print(f"{test_context}")
        print("-" * 60)
        
        for question in test_questions:
            print(f"\n{Fore.YELLOW}❓ {question}")
            answer = self.answer_question(test_context, question)
            print(f"{Fore.GREEN}💡 {answer}")
    
    def run_interactive_mode(self):
        """Run interactive question-answer session."""
        
        print(f"\n{Fore.MAGENTA}🔧 Interactive Mode - Enter your own text and questions")
        print("=" * 70)
        
        while True:
            # Get Sanskrit text
            print(f"\n{Fore.CYAN}📖 Enter Sanskrit text (or 'quit' to exit):", end=" ")
            text = input().strip()
            
            if text.lower() == 'quit':
                print(f"{Fore.YELLOW}🙏 धन्यवादः! Thank you for using the Sanskrit QA system!")
                break
            
            if not text:
                print(f"{Fore.RED}⚠️ Please enter some Sanskrit text.")
                continue
            
            # Ask questions about the text
            while True:
                print(f"{Fore.YELLOW}❓ Enter question (or 'new' for new text):", end=" ")
                question = input().strip()
                
                if question.lower() == 'new':
                    break
                
                if question.lower() == 'quit':
                    return
                
                if not question:
                    print(f"{Fore.RED}⚠️ Please enter a question.")
                    continue
                
                answer = self.answer_question(text, question)
                print(f"{Fore.GREEN}💡 Answer: {answer}")

def main():
    """Main demo function."""
    print(f"{Fore.MAGENTA}🕉️  Simple Sanskrit Question Answering System Demo")
    print(f"Enhanced with Kumārasaṃbhava Training Data")
    print("=" * 70)
    
    try:
        # Initialize demo
        demo = SimpleSanskritQADemo()
        
        # Run test questions
        demo.run_test_questions()
        
        # Run interactive mode
        demo.run_interactive_mode()
        
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}🙏 Demo interrupted by user. धन्यवादः!")
    except Exception as e:
        print(f"\n{Fore.RED}❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
