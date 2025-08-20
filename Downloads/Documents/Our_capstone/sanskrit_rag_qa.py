#!/usr/bin/env python3
"""
Sanskrit RAG-based Question Answering System
============================================

This script implements a Retrieval-Augmented Generation (RAG) system
for Sanskrit text analysis using the Kumārasaṃbhava corpus.
"""

import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import logging
from pathlib import Path
from collections import defaultdict
import re
from colorama import init, Fore, Style
import sys

# Initialize colorama
init(autoreset=True)

class SanskritRAGSystem:
    """RAG-based Sanskrit Question Answering System"""
    
    def __init__(self, corpus_path="files/Kumārasaṃbhava/"):
        self.corpus_path = Path(corpus_path)
        self.sentences = []
        self.sentence_embeddings = []
        self.pos_data = {}
        self.morphology_data = {}
        self.vectorizer = None
        self.tfidf_matrix = None
        
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        # Initialize the system
        self.load_corpus()
        self.build_retrieval_index()
        self.load_linguistic_data()
    
    def load_corpus(self):
        """Load the Kumārasaṃbhava corpus from CoNLL-U files"""
        print(f"{Fore.CYAN}📚 Loading Kumārasaṃbhava corpus...")
        
        conllu_files = list(self.corpus_path.glob("*.conllu"))
        if not conllu_files:
            print(f"{Fore.RED}❌ No CoNLL-U files found in {self.corpus_path}")
            return
        
        for file_path in conllu_files:
            print(f"  Processing: {file_path.name}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self._parse_conllu_content(content)
            except Exception as e:
                print(f"  ⚠️ Error processing {file_path.name}: {e}")
        
        print(f"✅ Loaded {len(self.sentences)} sentences from corpus")
    
    def _parse_conllu_content(self, content):
        """Parse CoNLL-U content and extract sentences"""
        current_sentence = []
        current_text = ""
        
        for line in content.split('\n'):
            line = line.strip()
            
            if line.startswith('# text = '):
                current_text = line.replace('# text = ', '').strip()
            elif line and not line.startswith('#') and '\t' in line:
                parts = line.split('\t')
                if len(parts) >= 10:
                    token_id, form, lemma, upos, xpos, feats, head, deprel, deps, misc = parts[:10]
                    
                    # Skip multi-word tokens (those with range like 4-5)
                    if '-' not in token_id:
                        current_sentence.append({
                            'form': form,
                            'lemma': lemma,
                            'upos': upos,
                            'xpos': xpos,
                            'feats': feats,
                            'deprel': deprel
                        })
            elif line == '' and current_sentence:
                # End of sentence
                if current_text:
                    self.sentences.append({
                        'text': current_text,
                        'tokens': current_sentence,
                        'words': [token['form'] for token in current_sentence]
                    })
                current_sentence = []
                current_text = ""
    
    def build_retrieval_index(self):
        """Build TF-IDF index for sentence retrieval"""
        print(f"{Fore.CYAN}🔍 Building retrieval index...")
        
        if not self.sentences:
            print(f"{Fore.RED}❌ No sentences to index")
            return
        
        # Extract text for TF-IDF
        texts = [sent['text'] for sent in self.sentences]
        
        # Build TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words=None,  # Keep all words for Sanskrit
            lowercase=True
        )
        
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        print(f"✅ Built retrieval index with {self.tfidf_matrix.shape[0]} sentences")
    
    def load_linguistic_data(self):
        """Load POS and morphological data"""
        print(f"{Fore.CYAN}📖 Loading linguistic data...")
        
        pos_count = 0
        morph_count = 0
        
        for sentence in self.sentences:
            for token in sentence['tokens']:
                word = token['form'].lower()
                
                # Store POS data
                if token['upos'] != '_':
                    self.pos_data[word] = token['upos']
                    pos_count += 1
                
                # Store morphological data
                if token['feats'] != '_':
                    self.morphology_data[word] = {
                        'lemma': token['lemma'],
                        'features': token['feats'],
                        'deprel': token['deprel']
                    }
                    morph_count += 1
        
        print(f"✅ Loaded {pos_count} POS mappings and {morph_count} morphological entries")
    
    def retrieve_relevant_sentences(self, query, top_k=5):
        """Retrieve most relevant sentences for a query"""
        if not self.vectorizer or self.tfidf_matrix is None:
            return []
        
        # Vectorize the query
        query_vector = self.vectorizer.transform([query.lower()])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top-k most similar sentences
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        relevant_sentences = []
        for idx in top_indices:
            if similarities[idx] > 0.01:  # Minimum similarity threshold
                relevant_sentences.append({
                    'sentence': self.sentences[idx],
                    'similarity': similarities[idx],
                    'index': idx
                })
        
        return relevant_sentences
    
    def analyze_pos_and_morphology(self, text):
        """Analyze POS and morphology of input text"""
        words = text.split()
        analysis = []
        
        for word in words:
            word_lower = word.lower()
            word_analysis = {
                'word': word,
                'pos': self.pos_data.get(word_lower, 'UNKNOWN'),
                'morphology': self.morphology_data.get(word_lower, {})
            }
            analysis.append(word_analysis)
        
        return analysis
    
    def generate_answer(self, context, question, retrieved_context=None):
        """Generate answer using RAG approach"""
        context_lower = context.lower()
        question_lower = question.lower()
        
        # Get linguistic analysis
        pos_analysis = self.analyze_pos_and_morphology(context)
        
        # Use retrieved context if available
        if retrieved_context:
            similar_examples = self._find_similar_patterns(context, retrieved_context)
        else:
            similar_examples = []
        
        # Generate answer based on question type
        if 'subject' in question_lower or 'main' in question_lower:
            return self._analyze_subjects_rag(context, pos_analysis, similar_examples)
        elif 'character' in question_lower or 'who' in question_lower:
            return self._analyze_characters_rag(context, pos_analysis, similar_examples)
        elif 'action' in question_lower or 'verb' in question_lower:
            return self._analyze_actions_rag(context, pos_analysis, similar_examples)
        elif 'sentiment' in question_lower or 'emotion' in question_lower:
            return self._analyze_sentiment_rag(context, pos_analysis, similar_examples)
        elif 'location' in question_lower or 'place' in question_lower:
            return self._analyze_locations_rag(context, pos_analysis, similar_examples)
        else:
            return self._general_analysis_rag(context, pos_analysis, similar_examples)
    
    def _find_similar_patterns(self, context, retrieved_context):
        """Find similar patterns in retrieved context"""
        patterns = []
        context_words = set(context.lower().split())
        
        for item in retrieved_context:
            sentence = item['sentence']
            sentence_words = set(sentence['text'].lower().split())
            
            # Find common words
            common_words = context_words.intersection(sentence_words)
            if common_words:
                patterns.append({
                    'sentence': sentence['text'],
                    'common_words': list(common_words),
                    'similarity': item['similarity'],
                    'tokens': sentence['tokens']
                })
        
        return patterns
    
    def _analyze_subjects_rag(self, context, pos_analysis, similar_examples):
        """Analyze subjects using RAG"""
        subjects = []
        
        # Find subjects from POS analysis
        for item in pos_analysis:
            if item['pos'] in ['NOUN', 'PROPN'] and len(item['word']) > 3:
                confidence = 70
                if item['morphology']:
                    feats = item['morphology'].get('features', '')
                    if 'Nom' in feats:  # Nominative case
                        confidence = 90
                
                subjects.append({
                    'word': item['word'],
                    'pos': item['pos'],
                    'confidence': confidence,
                    'features': item['morphology'].get('features', '')
                })
        
        # Add context from similar examples
        context_info = []
        for pattern in similar_examples:
            for token in pattern['tokens']:
                if token['upos'] in ['NOUN', 'PROPN'] and token['form'].lower() in context.lower():
                    context_info.append(f"In similar context: '{token['form']}' ({token['upos']})")
        
        result = f"Main subjects identified: "
        if subjects:
            subject_list = [f"{s['word']} ({s['pos']}, {s['confidence']}%)" for s in subjects[:3]]
            result += "; ".join(subject_list)
        else:
            result += "No clear subjects identified"
        
        if context_info:
            result += f"\n📚 Context from corpus: {'; '.join(context_info[:2])}"
        
        return result
    
    def _analyze_actions_rag(self, context, pos_analysis, similar_examples):
        """Analyze actions using RAG"""
        actions = []
        
        # Find verbs from POS analysis
        for item in pos_analysis:
            if item['pos'] == 'VERB':
                confidence = 80
                features = item['morphology'].get('features', '')
                
                actions.append({
                    'word': item['word'],
                    'pos': item['pos'],
                    'confidence': confidence,
                    'features': features,
                    'lemma': item['morphology'].get('lemma', '')
                })
        
        # Add context from similar examples
        context_info = []
        for pattern in similar_examples:
            for token in pattern['tokens']:
                if token['upos'] == 'VERB' and token['form'].lower() in context.lower():
                    context_info.append(f"'{token['form']}' (lemma: {token['lemma']})")
        
        result = f"Actions identified: "
        if actions:
            action_list = []
            for a in actions:
                action_desc = f"{a['word']} ({a['pos']}"
                if a['lemma']:
                    action_desc += f", lemma: {a['lemma']}"
                action_desc += ")"
                action_list.append(action_desc)
            result += "; ".join(action_list[:2])
        else:
            result += "No clear actions identified"
        
        if context_info:
            result += f"\n📚 Context from corpus: {'; '.join(context_info[:2])}"
        
        return result
    
    def _analyze_characters_rag(self, context, pos_analysis, similar_examples):
        """Analyze characters using RAG"""
        characters = []
        
        # Known character mappings
        character_mappings = {
            'himālaya': 'Himālaya (King of Mountains)',
            'himālayo': 'Himālaya (divine mountain king)',
            'pārvatī': 'Pārvatī (daughter of Himālaya)',
            'umā': 'Umā (another name for Pārvatī)',
            'śiva': 'Śiva (the great god)',
            'rudra': 'Rudra (form of Śiva)',
            'indra': 'Indra (king of gods)'
        }
        
        # Find characters from proper nouns
        for item in pos_analysis:
            if item['pos'] == 'PROPN' or item['word'].lower() in character_mappings:
                char_desc = character_mappings.get(item['word'].lower(), f"{item['word']} (character/entity)")
                characters.append(char_desc)
        
        # Add context from similar examples
        context_info = []
        for pattern in similar_examples:
            for token in pattern['tokens']:
                if token['upos'] == 'PROPN' and token['form'].lower() in character_mappings:
                    context_info.append(character_mappings[token['form'].lower()])
        
        result = f"Characters identified: "
        if characters:
            result += "; ".join(characters[:3])
        else:
            result += "No specific characters identified"
        
        if context_info:
            result += f"\n📚 From corpus: {'; '.join(set(context_info[:2]))}"
        
        return result
    
    def _analyze_sentiment_rag(self, context, pos_analysis, similar_examples):
        """Analyze sentiment using RAG"""
        # Define sentiment patterns
        sentiment_patterns = {
            'Adbhuta (Wonder/Awe)': ['devatātmā', 'himālaya', 'mahā', 'divya'],
            'Shanta (Peace/Tranquility)': ['śānti', 'prasāda', 'śama'],
            'Vira (Heroism/Courage)': ['vīra', 'śūra', 'rāja', 'adhirāja'],
            'Shringara (Love/Beauty)': ['priya', 'sundara', 'kāma', 'rati'],
            'Karuna (Compassion)': ['karuṇā', 'dīna', 'kṛpā', 'dayā']
        }
        
        detected_emotions = []
        context_lower = context.lower()
        
        for emotion, patterns in sentiment_patterns.items():
            for pattern in patterns:
                if pattern in context_lower:
                    detected_emotions.append(f"{emotion} (pattern: {pattern})")
                    break
        
        # Look for sentiment indicators in retrieved context
        corpus_sentiment = []
        for pattern in similar_examples:
            sentence_text = pattern['sentence'].lower()
            for emotion, patterns in sentiment_patterns.items():
                for p in patterns:
                    if p in sentence_text:
                        corpus_sentiment.append(f"{emotion} (from similar text)")
                        break
        
        result = "Sentiment analysis: "
        if detected_emotions:
            result += "; ".join(detected_emotions)
        else:
            result += "Neutral descriptive tone"
        
        if corpus_sentiment:
            result += f"\n📚 Corpus context: {'; '.join(set(corpus_sentiment[:2]))}"
        
        return result
    
    def _analyze_locations_rag(self, context, pos_analysis, similar_examples):
        """Analyze locations using RAG"""
        locations = []
        
        # Location patterns
        location_indicators = ['diśi', 'deśa', 'kṣetra', 'pura', 'nagara', 'uttarasyāṃ']
        
        for word in context.split():
            if word.lower() in location_indicators:
                locations.append(f"{word} (location indicator)")
        
        # Check POS for location-related terms
        for item in pos_analysis:
            if item['morphology']:
                feats = item['morphology'].get('features', '')
                if 'Loc' in feats:  # Locative case
                    locations.append(f"{item['word']} (locative)")
        
        result = "Locations/Directions: "
        if locations:
            result += "; ".join(locations[:3])
        else:
            result += "No specific locations identified"
        
        return result
    
    def _general_analysis_rag(self, context, pos_analysis, similar_examples):
        """General analysis using RAG"""
        pos_counts = defaultdict(int)
        for item in pos_analysis:
            pos_counts[item['pos']] += 1
        
        result = f"Text analysis: {len(context.split())} words; "
        result += f"POS distribution: {dict(pos_counts)}; "
        
        if similar_examples:
            result += f"Found {len(similar_examples)} similar examples in corpus"
        
        return result

def run_interactive_rag_demo():
    """Run interactive RAG-based QA demo"""
    print(f"{Fore.MAGENTA}🕉️  Sanskrit RAG Question Answering System")
    print(f"{Fore.MAGENTA}Enhanced with Retrieval-Augmented Generation")
    print("=" * 60)
    
    # Initialize RAG system
    print(f"{Fore.CYAN}🚀 Initializing RAG system...")
    rag_system = SanskritRAGSystem()
    
    print(f"\n{Fore.GREEN}✅ RAG system ready!")
    print(f"\n{Fore.CYAN}🔧 Interactive Mode - Enter Sanskrit text and questions")
    print("=" * 60)
    
    while True:
        print(f"\n{Fore.YELLOW}📖 Enter Sanskrit text (or 'quit' to exit): ", end="")
        text = input().strip()
        
        if text.lower() == 'quit':
            break
        
        if not text:
            continue
        
        # Retrieve relevant context
        retrieved_context = rag_system.retrieve_relevant_sentences(text, top_k=3)
        
        if retrieved_context:
            print(f"\n{Fore.CYAN}📚 Found {len(retrieved_context)} relevant examples from corpus:")
            for i, item in enumerate(retrieved_context, 1):
                similarity_pct = int(item['similarity'] * 100)
                print(f"  {i}. {item['sentence']['text'][:60]}... (similarity: {similarity_pct}%)")
        
        while True:
            print(f"\n{Fore.YELLOW}❓ Enter question (or 'new' for new text): ", end="")
            question = input().strip()
            
            if question.lower() == 'new':
                break
            
            if question.lower() == 'quit':
                return
            
            if not question:
                continue
            
            # Generate answer using RAG
            answer = rag_system.generate_answer(text, question, retrieved_context)
            print(f"{Fore.GREEN}💡 Answer: {answer}")

def run_sample_tests():
    """Run sample tests with RAG system"""
    print(f"{Fore.MAGENTA}📚 Testing RAG system with sample verses...")
    
    rag_system = SanskritRAGSystem()
    
    test_cases = [
        {
            'text': 'asty uttarasyāṃ diśi devatātmā himālayo nāma nagādhirājaḥ',
            'questions': [
                'What is the main subject?',
                'Who are the characters?',
                'What action is described?',
                'What is the sentiment?'
            ]
        },
        {
            'text': 'pūrvāparau toyanidhī vigāhya sthitaḥ pṛthivyā iva mānadaṇḍaḥ',
            'questions': [
                'What is the main subject?',
                'What action is described?',
                'What locations are mentioned?'
            ]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{Fore.CYAN}📖 Test {i}: {test_case['text']}")
        print("-" * 60)
        
        # Retrieve context
        retrieved_context = rag_system.retrieve_relevant_sentences(test_case['text'], top_k=3)
        
        for question in test_case['questions']:
            answer = rag_system.generate_answer(test_case['text'], question, retrieved_context)
            print(f"{Fore.YELLOW}❓ {question}")
            print(f"{Fore.GREEN}💡 {answer}\n")

def main():
    """Main function"""
    print(f"{Fore.MAGENTA}🕉️  Sanskrit RAG-based Question Answering System")
    print("=" * 60)
    
    # Run sample tests first
    run_sample_tests()
    
    # Then run interactive demo
    run_interactive_rag_demo()
    
    print(f"\n{Fore.CYAN}🙏 Thank you for using the Sanskrit RAG QA system!")

if __name__ == "__main__":
    main()
