#!/usr/bin/env python3
"""
Demo Script for Intensively Trained Sanskrit QA Model
====================================================

This script demonstrates the capabilities of the intensively trained
Kumārasaṃbhava model with enhanced contextual understanding.
"""

import torch
import json
import logging
import os
from pathlib import Path
from colorama import init, Fore, Style
import sys
import warnings
warnings.filterwarnings("ignore")

# Initialize colorama for colored output
init(autoreset=True)

# Import from sanskrit_qa_system
try:
    from sanskrit_qa_system import SanskritQuestionAnsweringSystem, SanskritDatasetLoader
    print("✅ Successfully imported from sanskrit_qa_system")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure sanskrit_qa_system.py is in the same directory")
    sys.exit(1)

class IntensivelyTrainedQADemo:
    """Demo class for the intensively trained Sanskrit QA model."""
    
    def __init__(self, model_path="best_kumarasambhava_model.pth"):
        self.model_path = model_path
        self.qa_system = None
        self.model_loaded = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab_size = 5000  # Default
        self.token_to_idx = {}
        
        print(f"🖥️  Using device: {self.device}")
        
        # Initialize the Sanskrit QA system
        self.initialize_qa_system()
    
    def initialize_qa_system(self):
        """Initialize the Sanskrit QA system with trained model"""
        try:
            # Try to load the trained model
            if os.path.exists('best_kumarasambhava_model.pth'):
                print("🔄 Loading trained model...")
                checkpoint = torch.load('best_kumarasambhava_model.pth', map_location=self.device)
                
                # Get vocabulary size from checkpoint
                vocab_size = checkpoint.get('vocab_size', 5000)
                
                # Initialize QA system with correct parameters
                self.qa_system = SanskritQuestionAnsweringSystem(vocab_size=vocab_size)
                self.qa_system.load_state_dict(checkpoint['model_state_dict'])
                self.qa_system.to(self.device)
                self.qa_system.eval()
                
                # Load vocabulary if available
                self.vocab_size = vocab_size
                if 'token_to_idx' in checkpoint:
                    self.token_to_idx = checkpoint['token_to_idx']
                else:
                    self.token_to_idx = {}
                    
                print("✅ Trained model loaded successfully!")
                print(f"📊 Vocabulary size: {vocab_size}")
                return True
            else:
                print("⚠️ No trained model found, initializing new system...")
                return self._initialize_basic_system()
                
        except Exception as e:
            print(f"❌ Error loading trained model: {e}")
            return self._initialize_basic_system()
    
    def _initialize_basic_system(self):
        """Initialize basic QA system without trained model"""
        try:
            # Initialize with default vocabulary size
            self.vocab_size = 5000  # Default vocabulary size
            self.qa_system = SanskritQuestionAnsweringSystem(vocab_size=self.vocab_size)
            self.qa_system.to(self.device)
            self.token_to_idx = {}
            print("✅ Basic QA system initialized")
            print(f"📊 Using default vocabulary size: {self.vocab_size}")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize basic system: {e}")
            return False
    
    def _initialize_fallback(self):
        """Initialize fallback mode with basic QA system."""
        try:
            # Create basic QA system
            self.qa_system = SanskritQuestionAnsweringSystem()
            sample_text = "asty uttarasyāṃ diśi devatātmā himālayo nāma nagādhirājaḥ"
            self.qa_system.initialize_system([sample_text.split()])
            print(f"{Fore.GREEN}✅ Fallback system initialized")
        except Exception as e:
            print(f"{Fore.RED}❌ Even fallback failed: {e}")
            self.qa_system = None
    
    def answer_question(self, context, question):
        """Generate answer using the QA system."""
        if self.qa_system is None:
            return "❌ QA system not initialized"
        
        try:
            # Use the QA system's answer generation
            answer = self.qa_system.answer_question(context, question)
            
            # Add model status indicator
            status = "🧠 Trained Model" if self.model_loaded else "📖 Base System"
            return f"{answer} [{status}]"
            
        except Exception as e:
            return f"❌ Error generating answer: {e}"

def run_test_scenarios():
    """Run predefined test scenarios."""
    print(f"{Fore.MAGENTA}🕉️  Intensively Trained Sanskrit QA System Demo")
    print(f"{Fore.MAGENTA}Enhanced with 50-epoch Kumārasaṃbhava Training")
    print("=" * 70)
    
    # Initialize demo
    demo = IntensivelyTrainedQADemo()
    
    # Test context from Kumārasaṃbhava
    test_context = "asty uttarasyāṃ diśi devatātmā himālayo nāma nagādhirājaḥ"
    
    print(f"\n{Fore.CYAN}📖 Test Context from Kumārasaṃbhava:")
    print(f"{Fore.WHITE}{test_context}")
    print("-" * 60)
    
    # Test questions
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
    
    for question in test_questions:
        print(f"{Fore.YELLOW}❓ {question}")
        answer = demo.answer_question(test_context, question)
        print(f"{Fore.GREEN}💡 {answer}\n")

def run_interactive_mode():
    """Run interactive question-answering mode."""
    print(f"\n{Fore.CYAN}🔧 Interactive Mode - Enter your own text and questions")
    print("=" * 70)
    
    demo = IntensivelyTrainedQADemo()
    
    while True:
        print(f"\n{Fore.CYAN}📖 Enter Sanskrit text (or 'quit' to exit): ", end="")
        context = input().strip()
        
        if context.lower() in ['quit', 'exit', 'q']:
            break
            
        if not context:
            continue
            
        while True:
            print(f"{Fore.YELLOW}❓ Enter question (or 'new' for new text): ", end="")
            question = input().strip()
            
            if question.lower() == 'new':
                break
            elif question.lower() in ['quit', 'exit', 'q']:
                return
            elif not question:
                continue
                
            answer = demo.answer_question(context, question)
            print(f"{Fore.GREEN}💡 Answer: {answer}")

def main():
    """Main demo function"""
    print(f"{Fore.MAGENTA}🕉️  Enhanced Sanskrit Question Answering System Demo")
    print(f"{Fore.MAGENTA}Enhanced with Intensive Kumārasaṃbhava Training")
    print("=" * 70)
    
    # Run test scenarios first
    run_test_scenarios()
    
    # Then run interactive mode
    run_interactive_mode()
    
    print(f"\n{Fore.CYAN}🙏 Thank you for using the Sanskrit QA System!")

if __name__ == "__main__":
    main()
