#!/usr/bin/env python3
"""
Simple Sanskrit QA Demo Script
==============================

A simplified demo that tests the Sanskrit QA system with basic functionality.
"""

import torch
import sys
import os
from colorama import init, Fore, Style
init(autoreset=True)

# Import the Sanskrit QA system
try:
    from sanskrit_qa_system import SanskritQATrainer, SanskritDatasetLoader
    print("✅ Successfully imported Sanskrit QA modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure sanskrit_qa_system.py is in the same directory")
    sys.exit(1)

class SimpleSanskritQADemo:
    """Simple demo class for Sanskrit QA"""
    
    def __init__(self):
        self.trainer = None
        self.is_initialized = False
    
    def initialize_system(self):
        """Initialize with basic trainer"""
        try:
            print(f"{Fore.CYAN}🔧 Initializing Sanskrit QA System...")
            self.trainer = SanskritQATrainer()
            self.is_initialized = True
            print(f"{Fore.GREEN}✅ System initialized successfully!")
            return True
        except Exception as e:
            print(f"{Fore.RED}❌ Initialization failed: {e}")
            return False
    
    def answer_question(self, context, question):
        """Generate answer for a given context and question"""
        if not self.is_initialized or not self.trainer:
            return "System not properly initialized"
        
        try:
            # Use the trainer's answer generation method
            answer = self.trainer._generate_contextual_answer(context, question)
            return answer
        except Exception as e:
            return f"Error generating answer: {e}"
    
    def test_sample_questions(self):
        """Test with predefined questions"""
        print(f"\n{Fore.MAGENTA}🧪 Testing Sample Questions")
        print(f"{Fore.MAGENTA}=" * 50)
        
        # Test context
        context = "asty uttarasyāṃ diśi devatātmā himālayo nāma nagādhirājaḥ"
        print(f"{Fore.CYAN}📖 Test Context: {context}")
        print("-" * 60)
        
        # Test questions
        test_questions = [
            "What is the main subject?",
            "Who are the characters mentioned?", 
            "What action is described?",
            "What is the sentiment/emotion?",
            "What divine or mythological elements are present?",
            "What is the grammatical structure?",
            "What literary devices are used?",
            "How does this relate to epic narrative?"
        ]
        
        for question in test_questions:
            print(f"\n{Fore.YELLOW}❓ Question: {question}")
            answer = self.answer_question(context, question)
            print(f"{Fore.GREEN}💡 Answer: {answer}")
    
    def interactive_demo(self):
        """Interactive demo mode"""
        print(f"\n{Fore.MAGENTA}🔄 Interactive Mode - Enter your own text and questions")
        print(f"{Fore.MAGENTA}=" * 60)
        
        while True:
            try:
                # Get context
                context = input(f"\n{Fore.CYAN}📖 Enter Sanskrit text (or 'quit' to exit): {Style.RESET_ALL}")
                if context.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not context.strip():
                    continue
                
                # Get questions for this context
                while True:
                    question = input(f"{Fore.YELLOW}❓ Enter question (or 'new' for new text): {Style.RESET_ALL}")
                    if question.lower() == 'new':
                        break
                    if question.lower() in ['quit', 'exit', 'q']:
                        return
                    
                    if question.strip():
                        answer = self.answer_question(context, question)
                        print(f"{Fore.GREEN}💡 Answer: {answer}")
                        
            except KeyboardInterrupt:
                break
        
        print(f"\n{Fore.YELLOW}👋 Thank you for using the Sanskrit QA Demo!")

def main():
    """Main demo function"""
    print(f"{Fore.MAGENTA}🕉️  Simple Sanskrit Question Answering Demo")
    print(f"{Fore.MAGENTA}=" * 50)
    
    demo = SimpleSanskritQADemo()
    
    # Initialize system
    if demo.initialize_system():
        print(f"{Fore.GREEN}🎉 System ready!")
        
        # Run sample tests first
        demo.test_sample_questions()
        
        # Then run interactive demo
        demo.interactive_demo()
    else:
        print(f"{Fore.RED}❌ Failed to initialize system.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}👋 Demo interrupted by user. Goodbye!")
