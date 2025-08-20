#!/usr/bin/env python3
"""
Simple Sanskrit RAG QA System
============================

A straightforward Retrieval-Augmented Generation system for Sanskrit QA
using the trained Kumārasaṃbhava corpus for context retrieval.
"""

import os
import re
import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from colorama import init, Fore, Style
import logging

# Initialize colorama
init(autoreset=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SimpleSanskritRAG:
    """Simple RAG system for Sanskrit QA using Kumārasaṃbhava corpus."""
    
    def __init__(self):
        self.corpus_sentences = []
        self.corpus_metadata = []
        self.pos_data = {}
        self.morphology_data = {}
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words=None)
        self.sentence_vectors = None
        
        print(f"{Fore.CYAN}🕉️  Simple Sanskrit RAG System Initializing...")
        self.load_corpus_data()
        self.build_retrieval_index()
        print(f"{Fore.GREEN}✅ RAG System ready with {len(self.corpus_sentences)} sentences")
    
    def load_corpus_data(self):
        """Load Kumārasaṃbhava corpus data from CoNLL-U files."""
        files_dir = Path("files/Kumārasaṃbhava")
        if not files_dir.exists():
            print(f"{Fore.RED}❌ Kumārasaṃbhava files directory not found!")
            return
        
        print(f"{Fore.YELLOW}📚 Loading corpus from CoNLL-U files...")
        
        conllu_files = list(files_dir.glob("*.conllu"))
        sentence_id = 0
        
        for file_path in conllu_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    current_sentence = []
                    current_text = ""
                    current_metadata = {}
                    
                    for line in f:
                        line = line.strip()
                        
                        if line.startswith('# text = '):
                            current_text = line[9:].strip()
                        elif line.startswith('# sent_id = '):
                            current_metadata['sent_id'] = line[12:].strip()
                        elif line and not line.startswith('#') and '\t' in line:
                            cols = line.split('\t')
                            if len(cols) >= 10 and not '-' in cols[0]:
                                word = cols[1]
                                lemma = cols[2] if cols[2] != '_' else word
                                pos = cols[3] if cols[3] != '_' else 'UNKNOWN'
                                features = cols[5] if cols[5] != '_' else ''
                                
                                current_sentence.append(word)
                                self.pos_data[word.lower()] = pos
                                self.morphology_data[word.lower()] = {
                                    'lemma': lemma,
                                    'pos': pos,
                                    'features': features
                                }
                        elif line == '' and current_sentence:
                            # End of sentence
                            sentence_text = current_text if current_text else ' '.join(current_sentence)
                            self.corpus_sentences.append(sentence_text)
                            self.corpus_metadata.append({
                                'id': sentence_id,
                                'file': file_path.name,
                                'words': current_sentence,
                                **current_metadata
                            })
                            sentence_id += 1
                            current_sentence = []
                            current_text = ""
                            current_metadata = {}
            except Exception as e:
                print(f"{Fore.YELLOW}⚠️ Error processing {file_path.name}: {e}")
        
        print(f"{Fore.GREEN}✅ Loaded {len(self.corpus_sentences)} sentences")
        print(f"{Fore.GREEN}✅ Loaded {len(self.pos_data)} POS mappings")
    
    def build_retrieval_index(self):
        """Build TF-IDF index for sentence retrieval."""
        if not self.corpus_sentences:
            print(f"{Fore.RED}❌ No corpus data to index!")
            return
        
        print(f"{Fore.YELLOW}🔍 Building retrieval index...")
        
        # Preprocess sentences for better matching
        processed_sentences = []
        for sentence in self.corpus_sentences:
            # Normalize and clean
            processed = re.sub(r'[^\w\s]', ' ', sentence.lower())
            processed = re.sub(r'\s+', ' ', processed).strip()
            processed_sentences.append(processed)
        
        # Build TF-IDF vectors
        self.sentence_vectors = self.vectorizer.fit_transform(processed_sentences)
        print(f"{Fore.GREEN}✅ Retrieval index built")
    
    def retrieve_context(self, query, top_k=3):
        """Retrieve most relevant sentences for the query."""
        if self.sentence_vectors is None:
            return []
        
        # Process query
        query_processed = re.sub(r'[^\w\s]', ' ', query.lower())
        query_processed = re.sub(r'\s+', ' ', query_processed).strip()
        
        # Get query vector
        query_vector = self.vectorizer.transform([query_processed])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.sentence_vectors).flatten()
        
        # Get top-k most similar sentences
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        retrieved_contexts = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum similarity threshold
                retrieved_contexts.append({
                    'sentence': self.corpus_sentences[idx],
                    'similarity': similarities[idx],
                    'metadata': self.corpus_metadata[idx]
                })
        
        return retrieved_contexts
    
    def analyze_with_context(self, text, question, retrieved_contexts):
        """Generate answer using retrieved contexts and local analysis."""
        text_lower = text.lower()
        question_lower = question.lower()
        
        # Combine retrieved contexts
        context_info = ""
        if retrieved_contexts:
            context_info = f"Based on similar verses in Kumārasaṃbhava: "
            context_examples = [ctx['sentence'] for ctx in retrieved_contexts[:2]]
            context_info += f"{'; '.join(context_examples)}. "
        
        # Subject analysis
        if 'subject' in question_lower or 'main' in question_lower or 'who' in question_lower:
            return self._analyze_subjects_with_rag(text, text_lower, retrieved_contexts, context_info)
        
        # Action analysis
        elif 'action' in question_lower or 'verb' in question_lower or 'do' in question_lower:
            return self._analyze_actions_with_rag(text, text_lower, retrieved_contexts, context_info)
        
        # Character analysis
        elif 'character' in question_lower or 'person' in question_lower:
            return self._analyze_characters_with_rag(text, text_lower, retrieved_contexts, context_info)
        
        # Sentiment analysis
        elif 'sentiment' in question_lower or 'emotion' in question_lower or 'feel' in question_lower:
            return self._analyze_sentiment_with_rag(text, text_lower, retrieved_contexts, context_info)
        
        # Location analysis
        elif 'location' in question_lower or 'place' in question_lower or 'where' in question_lower:
            return self._analyze_locations_with_rag(text, text_lower, retrieved_contexts, context_info)
        
        # General analysis
        else:
            return self._general_analysis_with_rag(text, text_lower, retrieved_contexts, context_info)
    
    def _analyze_subjects_with_rag(self, text, text_lower, contexts, context_info):
        """Subject analysis enhanced with RAG and detailed POS tagging."""
        pos_subjects = []
        semantic_subjects = []
        words = text.split()
        
        # Detailed POS analysis for subjects
        print(f"\n{Fore.CYAN}🔍 POS Analysis for Subjects:")
        for i, word in enumerate(words, 1):
            word_lower = word.lower()
            if word_lower in self.pos_data:
                pos = self.pos_data[word_lower]
                morph = self.morphology_data.get(word_lower, {})
                
                if pos in ['NOUN', 'PROPN']:
                    features = morph.get('features', '')
                    confidence = 60
                    case_info = ""
                    
                    if 'Case=Nom' in features:
                        case_info = " (Nominative - likely subject)"
                        confidence = 90
                    elif 'Case=Acc' in features:
                        case_info = " (Accusative - object)"
                        confidence = 40
                    elif 'Case=Loc' in features:
                        case_info = " (Locative - location)"
                        confidence = 30
                    elif 'Case=Gen' in features:
                        case_info = " (Genitive - possessive)"
                        confidence = 35
                    
                    pos_entry = f"{word} → {pos}{case_info}"
                    pos_subjects.append(pos_entry)
                    print(f"   {i}. {Fore.YELLOW}{word:<15} {Fore.WHITE}→ {Fore.GREEN}{pos:<8} {Fore.CYAN}{case_info} {Fore.WHITE}({confidence}%)")
        
        # Known semantic entities
        known_entities = {
            'himālaya': 'Himālaya (King of Mountains)',
            'himālayo': 'Himālaya (divine mountain king)',
            'devatātmā': 'Divine-souled entity',
            'nagādhirāja': 'King of mountains',
            'nagādhirājaḥ': 'King of mountains',
            'brahmā': 'Brahmā (creator god)',
            'śiva': 'Śiva (destroyer god)',
            'indra': 'Indra (king of gods)',
            'pārvatī': 'Pārvatī (daughter of Himālaya)'
        }
        
        for word in words:
            word_lower = word.lower()
            if word_lower in known_entities:
                semantic_subjects.append(known_entities[word_lower])
        
        # Check contexts for additional entities
        context_entities = []
        for ctx in contexts:
            ctx_words = ctx['sentence'].lower().split()
            for word in ctx_words:
                if word in known_entities and word not in [w.lower() for w in words]:
                    context_entities.append(f"{known_entities[word]} (from similar context)")
        
        # Format result
        result = f"{context_info}"
        
        if pos_subjects:
            result += f"📝 POS-tagged subjects: {'; '.join(pos_subjects[:3])}\n"
        
        if semantic_subjects:
            result += f"🎯 Main entities: {', '.join(semantic_subjects[:3])}\n"
        
        if context_entities:
            result += f"📚 From corpus: {'; '.join(context_entities[:2])}\n"
        
        if not pos_subjects and not semantic_subjects:
            result += "No clear subjects identified in this text."
        
        return result.strip()
    
    def _analyze_actions_with_rag(self, text, text_lower, contexts, context_info):
        """Action analysis enhanced with RAG and detailed POS tagging."""
        pos_verbs = []
        semantic_actions = []
        words = text.split()
        
        # Detailed POS analysis for verbs
        print(f"\n{Fore.CYAN}⚡ POS Analysis for Actions/Verbs:")
        for i, word in enumerate(words, 1):
            word_lower = word.lower()
            if word_lower in self.pos_data:
                pos = self.pos_data[word_lower]
                morph = self.morphology_data.get(word_lower, {})
                
                if pos == 'VERB':
                    features = morph.get('features', '')
                    lemma = morph.get('lemma', word)
                    
                    # Extract tense, mood, person info
                    tense = ""
                    if 'Tense=Pres' in features:
                        tense = "Present"
                    elif 'Tense=Past' in features:
                        tense = "Past"
                    elif 'Tense=Fut' in features:
                        tense = "Future"
                    
                    mood = ""
                    if 'Mood=Ind' in features:
                        mood = "Indicative"
                    elif 'Mood=Opt' in features:
                        mood = "Optative"
                    elif 'Mood=Imp' in features:
                        mood = "Imperative"
                    
                    person = ""
                    if 'Person=3' in features:
                        person = "3rd person"
                    elif 'Person=2' in features:
                        person = "2nd person"
                    elif 'Person=1' in features:
                        person = "1st person"
                    
                    verb_details = f"lemma: {lemma}"
                    if tense:
                        verb_details += f", {tense}"
                    if mood:
                        verb_details += f", {mood}"
                    if person:
                        verb_details += f", {person}"
                    
                    pos_entry = f"{word} → VERB ({verb_details})"
                    pos_verbs.append(pos_entry)
                    print(f"   {i}. {Fore.YELLOW}{word:<15} {Fore.WHITE}→ {Fore.RED}VERB {Fore.CYAN}({verb_details})")
        
        # Known action meanings
        action_meanings = {
            'asty': 'exists/there is (state of being)',
            'asti': 'exists/there is (state of being)',
            'bhavati': 'becomes/happens',
            'gacchati': 'goes/moves',
            'tiṣṭhati': 'stands/remains',
            'karoti': 'does/makes',
            'paśyati': 'sees',
            'jānāti': 'knows',
            'vadati': 'speaks/says'
        }
        
        for word in words:
            word_lower = word.lower()
            if word_lower in action_meanings:
                semantic_actions.append(f"{word} ({action_meanings[word_lower]})")
        
        # Check contexts for action patterns
        context_actions = []
        for ctx in contexts:
            ctx_words = ctx['sentence'].lower().split()
            for word in ctx_words:
                if word in action_meanings and word not in [w.lower() for w in words]:
                    context_actions.append(f"{word} ({action_meanings[word]}) - from similar context")
        
        # Format result
        result = f"{context_info}"
        
        if pos_verbs:
            result += f"⚡ POS-tagged verbs: {'; '.join(pos_verbs[:2])}\n"
        
        if semantic_actions:
            result += f"🎯 Main actions: {', '.join(semantic_actions[:3])}\n"
        
        if context_actions:
            result += f"📚 From corpus: {'; '.join(context_actions[:2])}\n"
        
        if not pos_verbs and not semantic_actions:
            result += "No clear actions identified in this text."
        
        return result.strip()
        
        return f"{context_info}No specific actions identified in this text."
    
    def _analyze_characters_with_rag(self, text, text_lower, contexts, context_info):
        """Character analysis enhanced with RAG."""
        characters = []
        
        # Known character patterns
        character_map = {
            'himālaya': 'Himālaya (King of Mountains)',
            'himālayo': 'Himālaya (nominative form)',
            'pārvatī': 'Pārvatī (daughter of Himālaya)',
            'śiva': 'Śiva (the great god)',
            'umā': 'Umā (another name for Pārvatī)'
        }
        
        for char, desc in character_map.items():
            if char in text_lower:
                characters.append(desc)
        
        # Check contexts for additional characters
        for ctx in contexts:
            for char, desc in character_map.items():
                if char in ctx['sentence'].lower() and desc not in characters:
                    characters.append(f"{desc} (from similar verse)")
        
        if characters:
            return f"{context_info}Characters: {', '.join(characters[:3])}"
        
        return f"{context_info}No specific characters identified."
    
    def _analyze_sentiment_with_rag(self, text, text_lower, contexts, context_info):
        """Sentiment analysis enhanced with RAG."""
        # Analyze current text
        sentiment_patterns = {
            'Wonder/Awe': ['devatā', 'divya', 'himālaya', 'mahā'],
            'Peace/Tranquility': ['śānti', 'prasāda', 'śama'],
            'Love/Beauty': ['priya', 'sundara', 'kāma', 'rati'],
            'Heroism': ['vīra', 'śakti', 'bala', 'tejā']
        }
        
        detected_sentiments = []
        for sentiment, patterns in sentiment_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    detected_sentiments.append(f"{sentiment} (pattern: {pattern})")
                    break
        
        # Check contexts for sentiment cues
        for ctx in contexts:
            ctx_lower = ctx['sentence'].lower()
            for sentiment, patterns in sentiment_patterns.items():
                for pattern in patterns:
                    if pattern in ctx_lower and sentiment not in [s.split()[0] for s in detected_sentiments]:
                        detected_sentiments.append(f"{sentiment} (from context)")
                        break
        
        if detected_sentiments:
            return f"{context_info}Emotional tone: {', '.join(detected_sentiments[:2])}"
        
        return f"{context_info}Neutral descriptive tone."
    
    def _analyze_locations_with_rag(self, text, text_lower, contexts, context_info):
        """Location analysis enhanced with RAG."""
        locations = []
        
        location_patterns = {
            'uttarasyāṃ diśi': 'northern direction',
            'parvata': 'mountain',
            'himālaya': 'Himalaya mountains',
            'kailāsa': 'Mount Kailāsa'
        }
        
        for pattern, desc in location_patterns.items():
            if pattern in text_lower:
                locations.append(desc)
        
        # Check contexts
        for ctx in contexts:
            for pattern, desc in location_patterns.items():
                if pattern in ctx['sentence'].lower() and desc not in locations:
                    locations.append(f"{desc} (from context)")
        
        if locations:
            return f"{context_info}Locations: {', '.join(locations[:3])}"
        
        return f"{context_info}No specific locations identified."
    
    def _general_analysis_with_rag(self, text, text_lower, contexts, context_info):
        """General analysis enhanced with RAG."""
        words = text.split()
        word_count = len(words)
        
        # Analyze POS distribution
        pos_counts = Counter()
        for word in words:
            pos = self.pos_data.get(word.lower(), 'UNKNOWN')
            pos_counts[pos] += 1
        
        analysis = f"Text analysis: {word_count} words. "
        if pos_counts:
            top_pos = pos_counts.most_common(2)
            analysis += f"Predominant word types: {', '.join([f'{pos}({count})' for pos, count in top_pos])}. "
        
        if contexts:
            analysis += f"Similar verses found in corpus (similarity: {contexts[0]['similarity']:.2f})"
        
        return f"{context_info}{analysis}"
    
    def answer_question(self, text, question):
        """Main method to answer questions using RAG."""
        try:
            # Retrieve relevant contexts
            retrieved_contexts = self.retrieve_context(text + " " + question, top_k=3)
            
            # Generate answer using contexts
            answer = self.analyze_with_context(text, question, retrieved_contexts)
            
            return answer
            
        except Exception as e:
            return f"❌ Error generating answer: {e}"

def run_interactive_demo():
    """Run interactive demo with the RAG system."""
    rag_system = SimpleSanskritRAG()
    
    print(f"\n{Fore.CYAN}🔧 Interactive Sanskrit RAG QA System")
    print("=" * 50)
    
    while True:
        print(f"\n{Fore.YELLOW}📖 Enter Sanskrit text (or 'quit' to exit): ", end="")
        text = input().strip()
        
        if text.lower() == 'quit':
            break
        
        if not text:
            continue
        
        while True:
            print(f"{Fore.YELLOW}❓ Enter question (or 'new' for new text): ", end="")
            question = input().strip()
            
            if question.lower() == 'new':
                break
            
            if question.lower() == 'quit':
                return
            
            if not question:
                continue
            
            print(f"{Fore.BLUE}🔍 Retrieving context and generating answer...")
            answer = rag_system.answer_question(text, question)
            print(f"{Fore.GREEN}💡 Answer: {answer}")

def run_sample_tests():
    """Run sample tests with known verses."""
    rag_system = SimpleSanskritRAG()
    
    print(f"\n{Fore.MAGENTA}📚 Testing with sample Kumārasaṃbhava verses:")
    
    test_cases = [
        {
            'text': 'asty uttarasyāṃ diśi devatātmā himālayo nāma nagādhirājaḥ',
            'questions': [
                'What is the main subject in this verse?',
                'What characters are mentioned?',
                'What is the sentiment?'
            ]
        },
        {
            'text': 'pūrvāparau toyanidhī vigāhya sthitaḥ pṛthivyā iva mānadaṇḍaḥ',
            'questions': [
                'What is the main subject?',
                'What action is described?'
            ]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{Fore.CYAN}📖 Test {i}: {test_case['text']}")
        print("-" * 60)
        
        for question in test_case['questions']:
            print(f"\n{Fore.YELLOW}❓ {question}")
            answer = rag_system.answer_question(test_case['text'], question)
            print(f"{Fore.GREEN}💡 {answer}")

def main():
    """Main function."""
    print(f"{Fore.MAGENTA}🕉️  Simple Sanskrit RAG QA System")
    print(f"{Fore.MAGENTA}Enhanced with Kumārasaṃbhava Corpus Retrieval")
    print("=" * 60)
    
    # Run sample tests first
    run_sample_tests()
    
    # Then run interactive demo
    run_interactive_demo()
    
    print(f"\n{Fore.CYAN}🙏 Thank you for using the Sanskrit RAG QA System!")

if __name__ == "__main__":
    main()
