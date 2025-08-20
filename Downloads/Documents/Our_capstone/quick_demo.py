#!/usr/bin/env python3
"""
Quick Sanskrit QA Demo - Lightweight Testing Interface

This script provides a quick way to test the Sanskrit QA system without
full model initialization. Perfect for testing the enhanced character recognition.

Usage:
    python quick_demo.py

Author: Sanskrit QA Team
Date: August 2025
"""

import sys
import re
from pathlib import Path

# Simple QA logic extracted for quick testing
class QuickSanskritQA:
    def __init__(self):
        """Initialize with character and divine name dictionaries."""
        # Enhanced character recognition patterns
        self.character_patterns = {
            # Devanagari characters
            'हिमालय': 'Himalaya (King of Mountains)',
            'शिव': 'Shiva',
            'पार्वती': 'Parvati', 
            'गंगा': 'Ganga',
            'विष्णु': 'Vishnu',
            'ब्रह्मा': 'Brahma',
            'इन्द्र': 'Indra',
            'कृष्ण': 'Krishna',
            'राम': 'Rama',
            'सीता': 'Sita',
            
            # IAST/transliterated forms
            'himālaya': 'Himalaya (King of Mountains)',
            'himālayo': 'Himalaya (King of Mountains)', 
            'śiva': 'Shiva',
            'pārvatī': 'Parvati',
            'gaṅgā': 'Ganga', 
            'viṣṇu': 'Vishnu',
            'brahmā': 'Brahma',
            'indra': 'Indra',
            'kṛṣṇa': 'Krishna',
            'rāma': 'Rama',
            'sītā': 'Sita',
            
            # Common epithets
            'nagādhirāja': 'King of Mountains (Himalaya)',
            'nagādhirājaḥ': 'King of Mountains (Himalaya)',
            'devatātmā': 'Divine-souled one',
            'mahādeva': 'Great God (Shiva)',
            'jagannātha': 'Lord of the Universe'
        }
        
        # Subject identification patterns
        self.subject_patterns = [
            r'\b(himālaya[ḥso]?)\b',
            r'\b(हिमालयः?)\b', 
            r'\b(nagādhirāja[ḥs]?)\b',
            r'\b(devatātmā)\b',
            r'\b(\w+ḥ)\s+nāma\b',  # X nāma pattern
        ]
        
        # Action/verb patterns
        self.action_patterns = [
            r'\b(asty|asti|अस्ति)\b',  # exists/is
            r'\b(bhavati|भवति)\b',     # becomes/is
            r'\b(tiṣṭhati|तिष्ठति)\b', # stands
            r'\b(gacchati|गच्छति)\b', # goes
        ]
    
    def extract_characters(self, text):
        """Extract characters from text using enhanced patterns."""
        characters = []
        text_lower = text.lower()
        
        for pattern, description in self.character_patterns.items():
            if pattern.lower() in text_lower:
                characters.append(f"{pattern} → {description}")
        
        return characters
    
    def extract_subjects(self, text):
        """Extract grammatical subjects from text."""
        subjects = []
        
        for pattern in self.subject_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            subjects.extend(matches)
        
        return subjects
    
    def extract_actions(self, text):
        """Extract actions/verbs from text."""
        actions = []
        
        for pattern in self.action_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            actions.extend(matches)
        
        return actions
    
    def analyze_sentiment(self, text):
        """Simple sentiment analysis based on keywords."""
        if any(word in text.lower() for word in ['devatātmā', 'divine', 'sacred']):
            return "Adbhuta (Wonder/Awe) - describing divine/majestic entities"
        elif any(word in text.lower() for word in ['mountain', 'himalaya', 'nagādhirāja']):
            return "Shanta (Peace/Tranquility) - describing natural grandeur"
        else:
            return "Neutral descriptive tone"
    
    def answer_question(self, context, question):
        """Generate answer based on question type."""
        question_lower = question.lower()
        
        if 'character' in question_lower or 'who' in question_lower:
            characters = self.extract_characters(context)
            if characters:
                return f"Characters identified: {', '.join(characters)}"
            else:
                return "No specific named characters detected in this text"
        
        elif 'subject' in question_lower or 'main' in question_lower:
            subjects = self.extract_subjects(context)
            if subjects:
                return f"Main subject(s): {', '.join(subjects)}"
            else:
                return "Subject not clearly identified"
        
        elif 'action' in question_lower or 'verb' in question_lower:
            actions = self.extract_actions(context)
            if actions:
                return f"Action(s) identified: {', '.join(actions)} (state of being/existence)"
            else:
                return "No clear action identified"
        
        elif 'sentiment' in question_lower or 'rasa' in question_lower:
            sentiment = self.analyze_sentiment(context)
            return f"Sentiment analysis: {sentiment}"
        
        elif 'grammar' in question_lower:
            subjects = self.extract_subjects(context)
            actions = self.extract_actions(context)
            return f"Grammatical structure: Subject(s): {subjects}, Verb(s): {actions}"
        
        else:
            # General analysis
            characters = self.extract_characters(context)
            subjects = self.extract_subjects(context)
            actions = self.extract_actions(context)
            
            parts = []
            if characters:
                parts.append(f"Characters: {', '.join(characters)}")
            if subjects:
                parts.append(f"Subjects: {', '.join(subjects)}")
            if actions:
                parts.append(f"Actions: {', '.join(actions)}")
            
            return "; ".join(parts) if parts else "Unable to parse this text"


def main():
    """Main demo function."""
    print("🕉️  Quick Sanskrit QA Demo")
    print("=" * 40)
    print("Enhanced with improved character recognition!")
    print("=" * 40)
    
    qa = QuickSanskritQA()
    
    # Test with the problematic example
    test_context = "asty uttarasyāṃ diśi devatātmā himālayo nāma nagādhirājaḥ"
    test_questions = [
        "What is the main subject in this verse?",
        "Who are the characters mentioned?", 
        "What action is being described?",
        "What is the sentiment of this text?"
    ]
    
    print(f"\n📖 Test Context: {test_context}")
    print("-" * 60)
    
    for question in test_questions:
        answer = qa.answer_question(test_context, question)
        print(f"\n❓ Question: {question}")
        print(f"💡 Answer: {answer}")
    
    print("\n" + "=" * 60)
    print("🔧 Interactive Mode - Enter your own text and questions")
    print("=" * 60)
    
    while True:
        try:
            context = input("\n📖 Enter Sanskrit text (or 'quit' to exit): ").strip()
            
            if context.lower() in ['quit', 'exit', 'q']:
                break
            
            if not context:
                print("⚠️  Please enter some text.")
                continue
            
            while True:
                question = input("❓ Enter question (or 'new' for new text): ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    return
                
                if question.lower() in ['new', 'n']:
                    break
                
                if not question:
                    print("⚠️  Please enter a question.")
                    continue
                
                answer = qa.answer_question(context, question)
                print(f"💡 Answer: {answer}")
        
        except KeyboardInterrupt:
            break
    
    print("\n👋 Thank you for using Quick Sanskrit QA Demo!")


if __name__ == "__main__":
    main()
