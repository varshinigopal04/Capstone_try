#!/usr/bin/env python3
"""
Sanskrit POS Tagger
==================

This script performs Part-of-Speech tagging for Sanskrit text using the trained
Kumārasaṃbhava model and CoNLL-U linguistic data.
"""

import torch
import json
import re
import os
from pathlib import Path
from collections import defaultdict
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

class SanskritPOSTagger:
    """Sanskrit Part-of-Speech Tagger using trained model and linguistic patterns"""
    
    def __init__(self, model_path="best_kumarasambhava_model.pth"):
        self.model_path = model_path
        self.pos_patterns = {}
        self.word_pos_mapping = {}
        self.morphological_patterns = {}
        
        print(f"{Fore.CYAN}🕉️  Sanskrit POS Tagger Initializing...")
        
        # Load linguistic data from CoNLL-U files
        self.load_linguistic_data()
        
        # Load trained model if available
        self.load_model_data()
        
        print(f"{Fore.GREEN}✅ POS Tagger initialized with {len(self.word_pos_mapping)} word mappings")
    
    def load_linguistic_data(self):
        """Load POS data from Kumārasaṃbhava CoNLL-U files"""
        conllu_dir = Path("files/Kumārasaṃbhava")
        
        if not conllu_dir.exists():
            print(f"{Fore.YELLOW}⚠️  CoNLL-U directory not found, using built-in patterns")
            self.setup_builtin_patterns()
            return
        
        print(f"{Fore.BLUE}📚 Loading linguistic data from CoNLL-U files...")
        
        # Process all .conllu files
        for conllu_file in conllu_dir.glob("*.conllu"):
            self.process_conllu_file(conllu_file)
        
        print(f"{Fore.GREEN}✅ Loaded {len(self.word_pos_mapping)} word-POS mappings")
    
    def process_conllu_file(self, file_path):
        """Process a single CoNLL-U file to extract POS information"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and '\t' in line:
                    columns = line.split('\t')
                    if len(columns) >= 4:
                        # CoNLL-U format: ID, FORM, LEMMA, UPOS, XPOS, FEATS, ...
                        word_form = columns[1].lower()
                        lemma = columns[2].lower() if columns[2] != '_' else word_form
                        pos_tag = columns[3]
                        
                        # Store word-POS mapping
                        if word_form not in self.word_pos_mapping:
                            self.word_pos_mapping[word_form] = []
                        self.word_pos_mapping[word_form].append(pos_tag)
                        
                        # Store lemma-POS mapping
                        if lemma not in self.word_pos_mapping:
                            self.word_pos_mapping[lemma] = []
                        self.word_pos_mapping[lemma].append(pos_tag)
                        
                        # Extract morphological features if available
                        if len(columns) > 5 and columns[5] != '_':
                            features = columns[5]
                            self.morphological_patterns[word_form] = {
                                'pos': pos_tag,
                                'features': features,
                                'lemma': lemma
                            }
        
        except Exception as e:
            print(f"{Fore.RED}❌ Error processing {file_path}: {e}")
    
    def setup_builtin_patterns(self):
        """Setup built-in POS patterns when CoNLL-U files are not available"""
        # Common Sanskrit word patterns based on Kumārasaṃbhava
        builtin_mappings = {
            # Nouns
            'himālaya': ['NOUN'], 'himālayo': ['NOUN'], 'himālayaḥ': ['NOUN'],
            'devatātmā': ['NOUN'], 'devatā': ['NOUN'], 'ātmā': ['NOUN'],
            'nagādhirāja': ['NOUN'], 'nagādhirājaḥ': ['NOUN'], 'naga': ['NOUN'],
            'adhirāja': ['NOUN'], 'adhirājaḥ': ['NOUN'], 'rāja': ['NOUN'],
            'diśi': ['NOUN'], 'pṛthivyā': ['NOUN'], 'toyanidhī': ['NOUN'],
            'mānadaṇḍaḥ': ['NOUN'], 'chāyām': ['NOUN'], 'ghanānāṃ': ['NOUN'],
            'śikharair': ['NOUN'], 'apsaras': ['NOUN'],
            
            # Verbs
            'asty': ['VERB'], 'asti': ['VERB'], 'sthitaḥ': ['VERB'],
            'vigāhya': ['VERB'], 'saṃcaratāṃ': ['VERB'], 'niṣevya': ['VERB'],
            'bibharti': ['VERB'], 'sampādayitrīṃ': ['VERB'],
            
            # Adjectives
            'uttarasyāṃ': ['ADJ'], 'pūrvāparau': ['ADJ'], 'divya': ['ADJ'],
            'mahā': ['ADJ'], 'adhaḥsānugatāṃ': ['ADJ'],
            
            # Pronouns
            'yaś': ['PRON'], 'yā': ['PRON'], 'yat': ['PRON'],
            
            # Adverbs
            'nāma': ['ADV'], 'iva': ['ADV'],
            
            # Particles
            'ca': ['PART'], 'vā': ['PART'], 'tu': ['PART']
        }
        
        self.word_pos_mapping.update(builtin_mappings)
    
    def load_model_data(self):
        """Load additional data from trained model if available"""
        if os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
                if 'vocab_size' in checkpoint:
                    print(f"{Fore.GREEN}📊 Model vocabulary size: {checkpoint['vocab_size']}")
                if 'token_to_idx' in checkpoint:
                    tokens = list(checkpoint['token_to_idx'].keys())
                    print(f"{Fore.GREEN}🔤 Loaded {len(tokens)} tokens from model")
            except Exception as e:
                print(f"{Fore.YELLOW}⚠️  Could not load model data: {e}")
    
    def analyze_morphology(self, word):
        """Analyze morphological features of a word"""
        word_lower = word.lower()
        
        # Check direct mapping first
        if word_lower in self.morphological_patterns:
            return self.morphological_patterns[word_lower]
        
        # Pattern-based analysis for Sanskrit morphology
        morphology = {'word': word, 'possible_analysis': []}
        
        # Noun case endings
        if word.endswith(('aḥ', 'ā', 'am', 'au', 'āḥ', 'āni')):
            morphology['possible_analysis'].append('NOUN (masculine/neuter declension)')
        elif word.endswith(('ā', 'āṃ', 'ayā', 'āyāḥ', 'āsu')):
            morphology['possible_analysis'].append('NOUN (feminine declension)')
        elif word.endswith(('i', 'iḥ', 'im', 'au', 'ayaḥ', 'īni')):
            morphology['possible_analysis'].append('NOUN (i-stem declension)')
        
        # Verb endings
        elif word.endswith(('ti', 'tāṃ', 'anti', 'te', 'nte', 'āte')):
            morphology['possible_analysis'].append('VERB (present tense)')
        elif word.endswith(('ta', 'tā', 'tam', 'tāḥ', 'tāni')):
            morphology['possible_analysis'].append('VERB (past participle)')
        elif word.endswith(('tvā', 'ya', 'itvā')):
            morphology['possible_analysis'].append('VERB (absolutive/gerund)')
        
        # Adjective patterns
        elif word.endswith(('vat', 'mat', 'vān', 'mān')):
            morphology['possible_analysis'].append('ADJ (possessive suffix)')
        
        return morphology
    
    def get_most_likely_pos(self, word):
        """Get the most likely POS tag for a word"""
        word_lower = word.lower()
        
        # Direct lookup
        if word_lower in self.word_pos_mapping:
            pos_tags = self.word_pos_mapping[word_lower]
            # Return most common POS tag
            pos_counts = defaultdict(int)
            for tag in pos_tags:
                pos_counts[tag] += 1
            return max(pos_counts.items(), key=lambda x: x[1])[0]
        
        # Pattern-based inference
        morphology = self.analyze_morphology(word)
        if morphology['possible_analysis']:
            # Extract POS from first analysis
            analysis = morphology['possible_analysis'][0]
            if 'NOUN' in analysis:
                return 'NOUN'
            elif 'VERB' in analysis:
                return 'VERB'
            elif 'ADJ' in analysis:
                return 'ADJ'
        
        # Default classification based on length and patterns
        if len(word) <= 2:
            return 'PART'  # Likely particle
        elif word.endswith(('aḥ', 'ā', 'am')):
            return 'NOUN'  # Common noun endings
        elif word.endswith(('ti', 'te')):
            return 'VERB'  # Common verb endings
        else:
            return 'UNKNOWN'
    
    def tag_text(self, text):
        """Perform POS tagging on Sanskrit text"""
        # Clean and tokenize the text
        words = re.findall(r'\S+', text)
        
        results = []
        for word in words:
            # Clean punctuation
            clean_word = re.sub(r'[^\w]', '', word)
            if not clean_word:
                continue
            
            pos_tag = self.get_most_likely_pos(clean_word)
            morphology = self.analyze_morphology(clean_word)
            
            word_analysis = {
                'word': word,
                'clean_word': clean_word,
                'pos': pos_tag,
                'morphology': morphology,
                'confidence': 'high' if clean_word.lower() in self.word_pos_mapping else 'medium'
            }
            results.append(word_analysis)
        
        return results
    
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
        
        print("=" * 60)
            word = analysis['word']
            pos = analysis['pos']
            confidence = analysis['confidence']
            
            # Color coding for POS tags
            if pos == 'NOUN':
                color = Fore.GREEN
            elif pos == 'VERB':
                color = Fore.BLUE
            elif pos == 'ADJ':
                color = Fore.YELLOW
            elif pos == 'PRON':
                color = Fore.MAGENTA
            elif pos == 'ADV':
                color = Fore.CYAN
            else:
                color = Fore.WHITE
            
            print(f"{i:2d}. {word:<15} → {color}{pos:<8}{Style.RESET_ALL} ({confidence})")
            
            # Show morphological analysis if available
            if analysis['morphology']['possible_analysis']:
                for morph in analysis['morphology']['possible_analysis']:
                    print(f"    {Fore.CYAN}↳ {morph}{Style.RESET_ALL}")
        
        # Summary
        pos_counts = defaultdict(int)
        for analysis in results:
            pos_counts[analysis['pos']] += 1
        
        print(f"\n{Fore.CYAN}📊 Summary:")
        for pos, count in sorted(pos_counts.items()):
            print(f"  {pos}: {count} words")
    
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
            
            # Perform POS tagging
            results = self.tag_text(text)
            self.format_results(results)
            
            # Ask for detailed analysis
            print(f"\n{Fore.YELLOW}🔍 Show detailed morphological analysis? (y/n): ", end="")
            show_details = input().strip().lower() == 'y'
            
            if show_details:
                print(f"\n{Fore.CYAN}🔬 Detailed Morphological Analysis:")
                print("-" * 40)
                for analysis in results:
                    word = analysis['clean_word']
                    print(f"\n{Fore.GREEN}{word}:")
                    if word.lower() in self.morphological_patterns:
                        patterns = self.morphological_patterns[word.lower()]
                        print(f"  Lemma: {patterns.get('lemma', 'N/A')}")
                        print(f"  Features: {patterns.get('features', 'N/A')}")
                    else:
                        print(f"  Pattern-based analysis: {analysis['morphology']['possible_analysis']}")

def main():
    """Main function"""
    print(f"{Fore.MAGENTA}🕉️  Sanskrit Part-of-Speech Tagger")
    print(f"{Fore.MAGENTA}Based on Kumārasaṃbhava Training Data")
    print("=" * 50)
    
    # Initialize POS tagger
    tagger = SanskritPOSTagger()
    
    # Test with sample texts
    test_texts = [
        "asty uttarasyāṃ diśi devatātmā himālayo nāma nagādhirājaḥ",
        "pūrvāparau toyanidhī vigāhya sthitaḥ pṛthivyā iva mānadaṇḍaḥ",
        "yaś cāpsarovibhramamaṇḍanānāṃ sampādayitrīṃ śikharair bibharti"
    ]
    
    print(f"\n{Fore.BLUE}📚 Testing with sample Kumārasaṃbhava verses:")
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{Fore.YELLOW}📖 Test {i}: {text}")
        results = tagger.tag_text(text)
        tagger.format_results(results)
    
    # Run interactive mode
    tagger.interactive_mode()
    
    print(f"\n{Fore.CYAN}🙏 Thank you for using the Sanskrit POS Tagger!")

if __name__ == "__main__":
    main()
