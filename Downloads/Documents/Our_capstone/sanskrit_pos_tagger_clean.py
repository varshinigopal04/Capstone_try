#!/usr/bin/env python3
"""
Sanskrit Part-of-Speech Tagger
=============================

This script performs Part-of-Speech tagging for Sanskrit text using
trained models and linguistic data from Kumārasaṃbhava corpus.
"""

import os
import re
import torch
import json
import logging
from pathlib import Path
from collections import defaultdict, Counter
from colorama import init, Fore, Style
import sys

# Initialize colorama for colored output
init(autoreset=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SanskritPOSTagger:
    """Sanskrit Part-of-Speech Tagger using Kumārasaṃbhava training data"""
    
    def __init__(self, model_path="best_kumarasambhava_model.pth"):
        self.model_path = model_path
        self.word_pos_map = {}
        self.pos_patterns = {}
        self.morphology_data = {}
        self.model_loaded = False
        
        print(f"{Fore.CYAN}🕉️  Sanskrit POS Tagger Initializing...")
        
        # Load linguistic data from CoNLL-U files
        self.load_linguistic_data()
        
        # Set up built-in patterns for unknown words
        self.setup_builtin_patterns()
        
        # Try to load model data if available
        self.load_model_data()
        
        print(f"{Fore.GREEN}✅ POS Tagger initialized with {len(self.word_pos_map)} word mappings")
    
    def load_linguistic_data(self):
        """Load POS and morphological data from Kumārasaṃbhava CoNLL-U files"""
        print(f"{Fore.YELLOW}📚 Loading linguistic data from CoNLL-U files...")
        
        conllu_dir = Path("files/Kumārasaṃbhava")
        if not conllu_dir.exists():
            print(f"{Fore.RED}⚠️ CoNLL-U directory not found: {conllu_dir}")
            return
        
        total_mappings = 0
        for conllu_file in conllu_dir.glob("*.conllu"):
            mappings = self.process_conllu_file(conllu_file)
            total_mappings += mappings
        
        print(f"{Fore.GREEN}✅ Loaded {total_mappings} word-POS mappings")
    
    def process_conllu_file(self, file_path):
        """Process a single CoNLL-U file to extract POS and morphological data"""
        mappings_count = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '\t' in line:
                        parts = line.split('\t')
                        if len(parts) >= 4 and not '-' in parts[0]:  # Skip multi-word tokens
                            token_id = parts[0]
                            word = parts[1].lower()  # Normalize case
                            lemma = parts[2]
                            pos = parts[3]
                            features = parts[5] if len(parts) > 5 else "_"
                            
                            # Store word-POS mapping
                            if word not in self.word_pos_map:
                                self.word_pos_map[word] = defaultdict(int)
                            self.word_pos_map[word][pos] += 1
                            
                            # Parse morphological features
                            morph_features = {}
                            if features != "_":
                                for feature in features.split('|'):
                                    if '=' in feature:
                                        key, value = feature.split('=', 1)
                                        morph_features[key.lower()] = value
                            
                            # Store morphological data
                            if word not in self.morphology_data:
                                self.morphology_data[word] = []
                            self.morphology_data[word].append({
                                'pos': pos,
                                'lemma': lemma,
                                'features': morph_features
                            })
                            
                            mappings_count += 1
                            
        except Exception as e:
            print(f"{Fore.RED}❌ Error processing {file_path}: {e}")
        
        return mappings_count
    
    def setup_builtin_patterns(self):
        """Set up built-in patterns for common Sanskrit word endings"""
        self.pos_patterns = {
            # Verb patterns
            'VERB': [
                r'.*ति$',      # Third person singular
                r'.*तु$',      # Third person singular imperative
                r'.*त्$',      # Past participle ending
                r'.*न्ति$',    # Third person plural
                r'.*स्ति$',    # asti, etc.
                r'.*आस$',     # Past tense
                r'.*त्वा$',    # Gerund
                r'.*य$',      # Gerund
            ],
            # Noun patterns
            'NOUN': [
                r'.*ः$',       # Masculine nominative singular
                r'.*आ$',      # Feminine nominative singular
                r'.*म्$',     # Neuter nominative singular
                r'.*स्य$',    # Genitive singular
                r'.*े$',      # Dual/locative
                r'.*ाः$',     # Masculine nominative plural
                r'.*ानि$',    # Neuter nominative plural
                r'.*येषु$',   # Locative plural
            ],
            # Adjective patterns (similar to nouns)
            'ADJ': [
                r'.*ः$',
                r'.*आ$', 
                r'.*म्$',
                r'.*स्य$',
            ],
            # Adverb patterns
            'ADV': [
                r'.*त्र$',     # Locative adverbs
                r'.*दा$',     # Temporal adverbs
                r'.*था$',     # Manner adverbs
            ]
        }
    
    def load_model_data(self):
        """Load additional data from trained model if available"""
        if os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location='cpu')
                self.model_loaded = True
                # Extract any additional linguistic data from model
                if 'vocab_size' in checkpoint:
                    print(f"{Fore.CYAN}📊 Model vocabulary size: {checkpoint['vocab_size']}")
            except Exception as e:
                print(f"{Fore.YELLOW}⚠️ Could not load model data: {e}")
    
    def analyze_morphology(self, word):
        """Analyze morphological features of a word"""
        word_lower = word.lower()
        
        if word_lower in self.morphology_data:
            analyses = self.morphology_data[word_lower]
            
            # Return the most common analysis
            pos_counts = defaultdict(int)
            for analysis in analyses:
                pos_counts[analysis['pos']] += 1
            
            most_common_pos = max(pos_counts.items(), key=lambda x: x[1])[0]
            
            # Find features for this POS
            features_for_pos = []
            for analysis in analyses:
                if analysis['pos'] == most_common_pos:
                    features_for_pos.append(analysis)
            
            return {
                'pos': most_common_pos,
                'confidence': 'high',
                'possible_analysis': [
                    f"{a['lemma']} ({a['pos']}) - {', '.join([f'{k}:{v}' for k, v in a['features'].items()])}"
                    for a in features_for_pos[:3]  # Top 3 analyses
                ]
            }
        
        return None
    
    def get_most_likely_pos(self, word):
        """Get the most likely POS tag for a word"""
        word_lower = word.lower()
        
        # First check direct mapping from training data
        if word_lower in self.word_pos_map:
            pos_counts = self.word_pos_map[word_lower]
            most_common_pos = max(pos_counts.items(), key=lambda x: x[1])
            return most_common_pos[0], 'high'
        
        # Use pattern matching for unknown words
        for pos, patterns in self.pos_patterns.items():
            for pattern in patterns:
                if re.match(pattern, word):
                    return pos, 'medium'
        
        # Default fallback based on word characteristics
        if word.isdigit():
            return 'NUM', 'low'
        elif len(word) == 1:
            return 'PUNCT', 'medium'
        elif word.endswith('ति') or word.endswith('न्ति'):
            return 'VERB', 'medium'
        elif word.endswith('ः') or word.endswith('म्'):
            return 'NOUN', 'medium'
        else:
            return 'UNKNOWN', 'low'
    
    def tag_text(self, text):
        """Tag a Sanskrit text with POS labels"""
        # Simple tokenization (split on whitespace)
        words = text.strip().split()
        
        results = []
        for word in words:
            # Get morphological analysis if available
            morph_analysis = self.analyze_morphology(word)
            
            if morph_analysis:
                pos = morph_analysis['pos']
                confidence = morph_analysis['confidence']
                morphology = {'possible_analysis': morph_analysis['possible_analysis']}
            else:
                pos, confidence = self.get_most_likely_pos(word)
                morphology = {}
            
            # Determine semantic category
            semantic_category = self.get_semantic_category(word, pos)
            
            analysis = {
                'pos': pos,
                'confidence': confidence,
                'morphology': morphology,
                'semantic_category': semantic_category
            }
            
            results.append((word, analysis))
        
        return results
    
    def get_semantic_category(self, word, pos):
        """Get semantic category for a word"""
        word_lower = word.lower()
        
        # Divine/mythological entities
        if word_lower in ['himālaya', 'himālayo', 'śiva', 'pārvatī', 'indra', 'brahmā']:
            return 'divine/mythological'
        
        # Natural phenomena
        elif word_lower in ['ghana', 'chāyā', 'diś', 'diśi', 'vāyu']:
            return 'natural phenomena'
        
        # Spatial/temporal
        elif word_lower in ['uttara', 'uttarasyāṃ', 'adhaḥ', 'upari']:
            return 'spatial/temporal'
        
        # Actions/states
        elif pos == 'VERB':
            return 'action/state'
        
        return None
    
    def format_results(self, results):
        """Format and display POS tagging results beautifully."""
        print(f"\n📝 POS Tagging Results:")
        print("=" * 60)
        
        for i, (word, analysis) in enumerate(results, 1):
            pos = analysis['pos']
            confidence = analysis['confidence']
            
            # Color coding based on POS
            if pos == 'NOUN':
                color = Fore.BLUE
            elif pos == 'VERB':
                color = Fore.GREEN
            elif pos == 'ADJ':
                color = Fore.YELLOW
            elif pos == 'ADV':
                color = Fore.MAGENTA
            else:
                color = Fore.CYAN
            
            print(f"{i:2}. {word:<15} → {color}{pos:<8}{Style.RESET_ALL} ({confidence})")
            
            # Show morphological analysis if available
            if analysis.get('morphology') and analysis['morphology'].get('possible_analysis'):
                morph_info = analysis['morphology']['possible_analysis'][:2]  # Show top 2
                for morph in morph_info:
                    print(f"    📋 {morph}")
            
            # Show semantic category if available
            if analysis.get('semantic_category'):
                print(f"    🏷️  Category: {analysis['semantic_category']}")
        
        # POS distribution summary
        pos_counts = Counter([analysis['pos'] for _, analysis in results])
        
        print(f"\n{Fore.CYAN}📊 Summary:")
        for pos, count in sorted(pos_counts.items()):
            print(f"  {pos}: {count} words")
        
        print("=" * 60)
    
    def interactive_mode(self):
        """Run interactive POS tagging mode"""
        print(f"\n{Fore.CYAN}🔧 Interactive POS Tagging Mode")
        print("=" * 50)
        
        while True:
            print(f"\n{Fore.YELLOW}📖 Enter Sanskrit text (or 'quit' to exit): ", end="")
            text = input().strip()
            
            if text.lower() == 'quit':
                break
            
            if not text:
                continue
            
            print(f"\n{Fore.CYAN}🔍 Analyzing: {text}")
            results = self.tag_text(text)
            self.format_results(results)

def main():
    """Main function to run the POS tagger"""
    print(f"{Fore.MAGENTA}🕉️  Sanskrit Part-of-Speech Tagger")
    print(f"{Fore.MAGENTA}Based on Kumārasaṃbhava Training Data")
    print("=" * 50)
    
    # Initialize tagger
    tagger = SanskritPOSTagger()
    
    # Test with sample verses
    print(f"\n{Fore.CYAN}📚 Testing with sample Kumārasaṃbhava verses:")
    
    test_verses = [
        "asty uttarasyāṃ diśi devatātmā himālayo nāma nagādhirājaḥ",
        "pūrvāparau toyanidhī vigāhya sthitaḥ pṛthivyā iva mānadaṇḍaḥ",
        "yaś cāpsarovibhramamaṇḍanānāṃ sampādayitrīṃ śikharair bibharti"
    ]
    
    for i, verse in enumerate(test_verses, 1):
        print(f"\n{Fore.YELLOW}📖 Test {i}: {verse}")
        results = tagger.tag_text(verse)
        tagger.format_results(results)
    
    # Run interactive mode
    tagger.interactive_mode()
    
    print(f"\n{Fore.CYAN}🙏 Thank you for using the Sanskrit POS Tagger!")

if __name__ == "__main__":
    main()
