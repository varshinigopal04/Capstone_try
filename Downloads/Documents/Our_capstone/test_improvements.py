#!/usr/bin/env python3
"""
Test Suite for Sanskrit QA System

This script tests the improved character recognition and contextual understanding.

Usage:
    python test_improvements.py

Author: Sanskrit QA Team  
Date: August 2025
"""

import sys
from pathlib import Path

# Import the quick demo for testing
try:
    from quick_demo import QuickSanskritQA
except ImportError:
    print("Error: Could not import quick_demo.py")
    sys.exit(1)


def test_character_recognition():
    """Test character recognition improvements."""
    print("🧪 Testing Character Recognition")
    print("=" * 40)
    
    qa = QuickSanskritQA()
    
    test_cases = [
        {
            "context": "asty uttarasyāṃ diśi devatātmā himālayo nāma nagādhirājaḥ",
            "question": "Who are the characters mentioned?",
            "expected_contains": ["Himalaya", "King of Mountains"]
        },
        {
            "context": "शिवस्य पत्नी पार्वती",
            "question": "Who are the characters mentioned?", 
            "expected_contains": ["Shiva", "Parvati"]
        },
        {
            "context": "कृष्णो गोविन्दः",
            "question": "Who are the characters mentioned?",
            "expected_contains": ["Krishna"]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 Test Case {i}:")
        print(f"   Context: {test_case['context']}")
        print(f"   Question: {test_case['question']}")
        
        answer = qa.answer_question(test_case['context'], test_case['question'])
        print(f"   Answer: {answer}")
        
        # Check if expected terms are in the answer
        success = all(term.lower() in answer.lower() for term in test_case['expected_contains'])
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   Status: {status}")
        
        if not success:
            print(f"   Expected to contain: {test_case['expected_contains']}")


def test_subject_identification():
    """Test subject identification."""
    print("\n🧪 Testing Subject Identification")
    print("=" * 40)
    
    qa = QuickSanskritQA()
    
    test_cases = [
        {
            "context": "asty uttarasyāṃ diśi devatātmā himālayo nāma nagādhirājaḥ",
            "question": "What is the main subject?",
            "expected_contains": ["himālayo", "himālaya"]
        },
        {
            "context": "rāmaḥ vanāya gaccati",
            "question": "What is the main subject?",
            "expected_contains": ["rāma"]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 Test Case {i}:")
        print(f"   Context: {test_case['context']}")
        print(f"   Question: {test_case['question']}")
        
        answer = qa.answer_question(test_case['context'], test_case['question'])
        print(f"   Answer: {answer}")
        
        # Check if any expected term is in the answer
        success = any(term.lower() in answer.lower() for term in test_case['expected_contains'])
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   Status: {status}")


def test_action_identification():
    """Test action identification."""
    print("\n🧪 Testing Action Identification")
    print("=" * 40)
    
    qa = QuickSanskritQA()
    
    test_cases = [
        {
            "context": "asty uttarasyāṃ diśi devatātmā himālayo nāma nagādhirājaḥ",
            "question": "What action is described?",
            "expected_contains": ["asty", "exist", "being"]
        },
        {
            "context": "रामो वनं गच्छति",
            "question": "What action is described?", 
            "expected_contains": ["गच्छति", "goes"]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 Test Case {i}:")
        print(f"   Context: {test_case['context']}")
        print(f"   Question: {test_case['question']}")
        
        answer = qa.answer_question(test_case['context'], test_case['question'])
        print(f"   Answer: {answer}")
        
        success = any(term.lower() in answer.lower() for term in test_case['expected_contains'])
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   Status: {status}")


def test_sentiment_analysis():
    """Test sentiment analysis."""
    print("\n🧪 Testing Sentiment Analysis")
    print("=" * 40)
    
    qa = QuickSanskritQA()
    
    test_cases = [
        {
            "context": "asty uttarasyāṃ diśi devatātmā himālayo nāma nagādhirājaḥ",
            "question": "What is the sentiment?",
            "expected_contains": ["Adbhuta", "Wonder", "Awe"]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 Test Case {i}:")
        print(f"   Context: {test_case['context']}")
        print(f"   Question: {test_case['question']}")
        
        answer = qa.answer_question(test_case['context'], test_case['question'])
        print(f"   Answer: {answer}")
        
        success = any(term.lower() in answer.lower() for term in test_case['expected_contains'])
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   Status: {status}")


def run_comprehensive_test():
    """Run comprehensive test on the original problematic case."""
    print("\n🎯 Comprehensive Test - Original Problem Case")
    print("=" * 50)
    
    qa = QuickSanskritQA()
    
    context = "asty uttarasyāṃ diśi devatātmā himālayo nāma nagādhirājaḥ"
    questions = [
        "What is the main subject in this verse?",
        "Who are the characters mentioned?",
        "What action is being described?",
        "What is the sentiment of this text?"
    ]
    
    print(f"📖 Context: {context}")
    print("-" * 50)
    
    for question in questions:
        answer = qa.answer_question(context, question)
        print(f"\n❓ {question}")
        print(f"💡 {answer}")
        
        # Specific validation for the character question
        if "character" in question.lower():
            if "himalaya" in answer.lower():
                print("   ✅ IMPROVED: Now correctly identifies Himalaya!")
            else:
                print("   ❌ Still not detecting Himalaya")


def main():
    """Run all tests."""
    print("🧪 Sanskrit QA System - Improvement Test Suite")
    print("=" * 60)
    print("Testing enhanced character recognition and contextual understanding")
    print("=" * 60)
    
    # Run individual tests
    test_character_recognition()
    test_subject_identification() 
    test_action_identification()
    test_sentiment_analysis()
    
    # Run comprehensive test
    run_comprehensive_test()
    
    print("\n" + "=" * 60)
    print("🏁 Test Suite Completed!")
    print("Check the results above to verify improvements.")
    print("=" * 60)


if __name__ == "__main__":
    main()
