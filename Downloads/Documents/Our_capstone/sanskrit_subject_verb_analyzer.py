#!/usr/bin/env python3
"""
Sanskrit Subject-Verb Analyzer
==============================

This script analyzes Sanskrit text to identify:
1. Main subjects/entities (using POS tagging)
2. Main verbs/actions (using POS tagging)
3. Enhanced analysis using trained Kumārasaṃbhava model

Uses linguistic data from CoNLL-U files for accurate analysis.
"""

import os
import re
import logging
from pathlib import Path
from collections import defaultdict, Counter
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

class SanskritSubjectVerbAnalyzer:
    """Advanced analyzer for identifying subjects and verbs in Sanskrit text"""
    
    def __init__(self):
        self.word_pos_map = {}
        self.word_morphology = {}
        self.compound_words = {}
        self.subject_patterns = {}
        self.verb_patterns = {}
        self.logger = logging.getLogger(__name__)
        
        print(f"{Fore.CYAN}🕉️  Sanskrit Subject-Verb Analyzer Initializing...")
        self.load_linguistic_data()
        self.build_analysis_patterns()
        print(f"{Fore.GREEN}✅ Analyzer ready with {len(self.word_pos_map)} word mappings")
    
    def load_linguistic_data(self):
        """Load POS and morphological data from CoNLL-U files"""
        print(f"{Fore.YELLOW}📚 Loading linguistic data from CoNLL-U files...")
        
        corpus_path = Path("files/Kumārasaṃbhava")
        if not corpus_path.exists():
            print(f"{Fore.RED}❌ Corpus directory not found: {corpus_path}")
            return
        
        conllu_files = list(corpus_path.glob("*.conllu"))
        
        for file_path in conllu_files:
            self._process_conllu_file(file_path)
        
        print(f"{Fore.GREEN}✅ Loaded {len(self.word_pos_map)} word-POS mappings")
        print(f"{Fore.GREEN}✅ Loaded {len(self.word_morphology)} morphological entries")
    
    def _process_conllu_file(self, file_path):
        """Process a single CoNLL-U file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            lines = content.splitlines()
            current_sentence = []
            
            for line in lines:
                line = line.strip()
                
                if line.startswith('# text'):
                    current_sentence = []
                elif line and not line.startswith('#') and '\t' in line:
                    self._process_token_line(line)
                    
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}")
    
    def _process_token_line(self, line):
        """Process a token line from CoNLL-U format"""
        columns = line.split('\t')
        if len(columns) >= 10:
            token_id = columns[0]
            form = columns[1]
            lemma = columns[2]
            pos = columns[3]
            features = columns[5]
            
            # Skip compound token ranges (e.g., "4-5")
            if '-' in token_id:
                return
            
            # Store POS mapping
            self.word_pos_map[form.lower()] = pos
            
            # Store morphological information
            morphology = self._parse_features(features)
            morphology['lemma'] = lemma
            morphology['pos'] = pos
            
            self.word_morphology[form.lower()] = morphology
    
    def _parse_features(self, features_string):
        """Parse morphological features from CoNLL-U format"""
        features = {}
        if features_string and features_string != '_':
            for feature in features_string.split('|'):
                if '=' in feature:
                    key, value = feature.split('=', 1)
                    features[key] = value
        return features
    
    def build_analysis_patterns(self):
        """Build patterns for subject and verb identification"""
        print(f"{Fore.YELLOW}🔍 Building subject and verb analysis patterns...")
        
        # Subject patterns (nominative case indicators)
        self.subject_patterns = {
            'nominative_endings': ['aḥ', 'ā', 'am', 'au', 'āḥ', 'āni'],
            'subject_markers': ['Case=Nom', 'Number=Sing', 'Number=Dual', 'Number=Plur'],
            'entity_types': ['NOUN', 'PROPN', 'PRON'],
            'compound_indicators': ['ātmā', 'rāja', 'deva', 'īśa', 'nātha', 'pati']
        }
        
        # Verb patterns
        self.verb_patterns = {
            'verb_endings': ['ti', 'nti', 'si', 'tha', 'mi', 'ma', 'va', 'ta', 'te'],
            'participle_endings': ['ta', 'na', 'ya', 'tvā', 'itvā'],
            'tense_markers': ['Tense=Pres', 'Tense=Past', 'Tense=Fut'],
            'mood_markers': ['Mood=Ind', 'Mood=Imp', 'Mood=Opt'],
            'action_types': ['VERB', 'AUX']
        }
    
    def analyze_text(self, text):
        """Comprehensive analysis to identify subjects and verbs"""
        words = text.split()
        results = {
            'text': text,
            'subjects': [],
            'verbs': [],
            'word_analysis': [],
            'compound_analysis': [],
            'main_entity': None,
            'main_action': None
        }
        
        # Analyze each word
        for i, word in enumerate(words):
            word_info = self.analyze_word(word, i)
            results['word_analysis'].append(word_info)
            
            # Classify as subject or verb
            if word_info['is_subject']:
                results['subjects'].append(word_info)
            
            if word_info['is_verb']:
                results['verbs'].append(word_info)
        
        # Identify main entity and action
        results['main_entity'] = self._find_main_entity(results['subjects'])
        results['main_action'] = self._find_main_action(results['verbs'])
        
        # Analyze compounds
        results['compound_analysis'] = self._analyze_compounds(words)
        
        return results
    
    def analyze_word(self, word, position=0):
        """Detailed analysis of a single word"""
        word_lower = word.lower()
        
        analysis = {
            'word': word,
            'position': position,
            'pos': self.word_pos_map.get(word_lower, 'UNKNOWN'),
            'morphology': self.word_morphology.get(word_lower, {}),
            'is_subject': False,
            'is_verb': False,
            'subject_confidence': 0,
            'verb_confidence': 0,
            'analysis_details': []
        }
        
        # Check if it's a subject
        analysis['is_subject'], analysis['subject_confidence'] = self._is_subject(word, word_lower, analysis)
        
        # Check if it's a verb
        analysis['is_verb'], analysis['verb_confidence'] = self._is_verb(word, word_lower, analysis)
        
        return analysis
    
    def _is_subject(self, word, word_lower, analysis):
        """Determine if a word is likely a subject"""
        confidence = 0
        reasons = []
        
        pos = analysis['pos']
        morphology = analysis['morphology']
        
        # Strong indicators
        if pos in ['NOUN', 'PROPN', 'PRON']:
            confidence += 40
            reasons.append(f"POS tag: {pos}")
        
        # Nominative case
        if 'Case=Nom' in str(morphology):
            confidence += 30
            reasons.append("Nominative case")
        
        # Subject-like endings
        for ending in self.subject_patterns['nominative_endings']:
            if word.endswith(ending):
                confidence += 15
                reasons.append(f"Nominative ending: -{ending}")
                break
        
        # Compound indicators
        for indicator in self.subject_patterns['compound_indicators']:
            if indicator in word_lower:
                confidence += 20
                reasons.append(f"Entity indicator: {indicator}")
                break
        
        # Known entities
        known_entities = ['himālaya', 'devatā', 'rāja', 'nātha', 'īśvara', 'deva']
        if any(entity in word_lower for entity in known_entities):
            confidence += 25
            reasons.append("Known entity pattern")
        
        analysis['analysis_details'].extend([f"Subject: {r}" for r in reasons])
        
        return confidence > 50, confidence
    
    def _is_verb(self, word, word_lower, analysis):
        """Determine if a word is likely a verb"""
        confidence = 0
        reasons = []
        
        pos = analysis['pos']
        morphology = analysis['morphology']
        
        # Strong indicators
        if pos in ['VERB', 'AUX']:
            confidence += 50
            reasons.append(f"POS tag: {pos}")
        
        # Verb forms
        if 'VerbForm' in str(morphology):
            confidence += 30
            reasons.append("Verbal form")
        
        # Tense markers
        for marker in self.verb_patterns['tense_markers']:
            if marker in str(morphology):
                confidence += 20
                reasons.append(f"Tense: {marker}")
                break
        
        # Verb endings
        for ending in self.verb_patterns['verb_endings']:
            if word.endswith(ending):
                confidence += 15
                reasons.append(f"Verbal ending: -{ending}")
                break
        
        # Participle endings
        for ending in self.verb_patterns['participle_endings']:
            if word.endswith(ending):
                confidence += 10
                reasons.append(f"Participle ending: -{ending}")
                break
        
        # Known verbs
        known_verbs = ['asti', 'asty', 'bhavati', 'gacchati', 'tiṣṭhati', 'karoti']
        if any(verb in word_lower for verb in known_verbs):
            confidence += 25
            reasons.append("Known verb pattern")
        
        analysis['analysis_details'].extend([f"Verb: {r}" for r in reasons])
        
        return confidence > 40, confidence
    
    def _find_main_entity(self, subjects):
        """Find the most likely main entity/subject"""
        if not subjects:
            return None
        
        # Sort by confidence and position
        main_subject = max(subjects, key=lambda x: (x['subject_confidence'], -x['position']))
        return main_subject
    
    def _find_main_action(self, verbs):
        """Find the most likely main action/verb"""
        if not verbs:
            return None
        
        # Sort by confidence, prefer finite verbs
        main_verb = max(verbs, key=lambda x: (x['verb_confidence'], x['pos'] == 'VERB'))
        return main_verb
    
    def _analyze_compounds(self, words):
        """Analyze compound words and relationships"""
        compounds = []
        
        for word in words:
            if len(word) > 8:  # Likely compound
                analysis = {
                    'word': word,
                    'type': 'compound',
                    'possible_parts': self._decompose_compound(word),
                    'semantic_role': self._determine_semantic_role(word)
                }
                compounds.append(analysis)
        
        return compounds
    
    def _decompose_compound(self, word):
        """Attempt to decompose compound words"""
        # Simple decomposition based on known elements
        known_parts = ['deva', 'rāja', 'nātha', 'pati', 'ātmā', 'jana', 'loka', 'kāla']
        parts = []
        
        for part in known_parts:
            if part in word.lower():
                parts.append(part)
        
        return parts
    
    def _determine_semantic_role(self, word):
        """Determine semantic role of word"""
        word_lower = word.lower()
        
        if any(x in word_lower for x in ['rāja', 'nātha', 'pati', 'īśa']):
            return 'ruler/leader'
        elif any(x in word_lower for x in ['deva', 'devatā', 'īśvara']):
            return 'divine entity'
        elif any(x in word_lower for x in ['ātmā', 'jana', 'puruṣa']):
            return 'person/soul'
        else:
            return 'unknown'
    
    def format_analysis(self, results):
        """Format analysis results for display"""
        print(f"\n{Fore.CYAN}📖 Text: {Fore.WHITE}{results['text']}")
        print(f"{Fore.CYAN}{'='*60}")
        
        # Main findings
        if results['main_entity']:
            entity = results['main_entity']
            print(f"{Fore.GREEN}🎯 Main Entity/Subject:")
            print(f"   Word: {Fore.YELLOW}{entity['word']}")
            print(f"   POS: {Fore.BLUE}{entity['pos']}")
            print(f"   Confidence: {Fore.MAGENTA}{entity['subject_confidence']}%")
            if entity['morphology']:
                features = ', '.join([f"{k}={v}" for k, v in entity['morphology'].items() if k != 'pos'])
                if features:
                    print(f"   Features: {Fore.CYAN}{features}")
        
        if results['main_action']:
            action = results['main_action']
            print(f"\n{Fore.GREEN}⚡ Main Action/Verb:")
            print(f"   Word: {Fore.YELLOW}{action['word']}")
            print(f"   POS: {Fore.BLUE}{action['pos']}")
            print(f"   Confidence: {Fore.MAGENTA}{action['verb_confidence']}%")
            if action['morphology']:
                features = ', '.join([f"{k}={v}" for k, v in action['morphology'].items() if k != 'pos'])
                if features:
                    print(f"   Features: {Fore.CYAN}{features}")
        
        # All subjects and verbs
        if results['subjects']:
            print(f"\n{Fore.BLUE}📋 All Subjects Found:")
            for i, subj in enumerate(results['subjects'], 1):
                print(f"   {i}. {Fore.YELLOW}{subj['word']} {Fore.LIGHTBLACK_EX}({subj['pos']}, {subj['subject_confidence']}%)")
        
        if results['verbs']:
            print(f"\n{Fore.BLUE}📋 All Verbs Found:")
            for i, verb in enumerate(results['verbs'], 1):
                print(f"   {i}. {Fore.YELLOW}{verb['word']} {Fore.LIGHTBLACK_EX}({verb['pos']}, {verb['verb_confidence']}%)")
        
        # Compounds
        if results['compound_analysis']:
            print(f"\n{Fore.BLUE}🔗 Compound Analysis:")
            for comp in results['compound_analysis']:
                print(f"   {Fore.YELLOW}{comp['word']} {Fore.LIGHTBLACK_EX}→ {comp['semantic_role']}")
                if comp['possible_parts']:
                    print(f"      Parts: {', '.join(comp['possible_parts'])}")

def run_sample_tests(analyzer):
    """Run tests on sample Sanskrit verses"""
    print(f"\n{Fore.MAGENTA}📚 Testing with sample Kumārasaṃbhava verses:")
    
    test_verses = [
        "asty uttarasyāṃ diśi devatātmā himālayo nāma nagādhirājaḥ",
        "pūrvāparau toyanidhī vigāhya sthitaḥ pṛthivyā iva mānadaṇḍaḥ",
        "yaś cāpsarovibhramamaṇḍanānāṃ sampādayitrīṃ śikharair bibharti"
    ]
    
    for i, verse in enumerate(test_verses, 1):
        print(f"\n{Fore.CYAN}📖 Test {i}: {verse}")
        results = analyzer.analyze_text(verse)
        analyzer.format_analysis(results)

def run_interactive_mode(analyzer):
    """Run interactive analysis mode"""
    print(f"\n{Fore.CYAN}🔧 Interactive Subject-Verb Analysis Mode")
    print(f"{Fore.CYAN}{'='*50}")
    
    while True:
        print(f"\n{Fore.YELLOW}📖 Enter Sanskrit text (or 'quit' to exit): ", end="")
        text = input().strip()
        
        if text.lower() == 'quit':
            break
        
        if not text:
            continue
        
        # Analyze the text
        results = analyzer.analyze_text(text)
        analyzer.format_analysis(results)

def main():
    """Main function"""
    print(f"{Fore.MAGENTA}🕉️  Sanskrit Subject-Verb Analyzer")
    print(f"{Fore.MAGENTA}Based on Kumārasaṃbhava Training Data")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = SanskritSubjectVerbAnalyzer()
    
    # Run sample tests
    run_sample_tests(analyzer)
    
    # Interactive mode
    run_interactive_mode(analyzer)
    
    print(f"\n{Fore.CYAN}🙏 धन्यवादः! Thank you for using the Sanskrit Subject-Verb Analyzer!")

if __name__ == "__main__":
    main()
