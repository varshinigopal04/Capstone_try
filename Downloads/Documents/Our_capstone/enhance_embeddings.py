#!/usr/bin/env python3
"""
Sanskrit Embedding Enhancement Script
====================================

This script focuses on improving the embedding quality and contextual understanding
for better QA performance through enhanced Word2Vec and transformer training.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import logging
from tqdm import tqdm

# Import the Sanskrit QA system
from sanskrit_qa_system import *

class EnhancedSanskritEmbeddings:
    """Enhanced embedding system with better contextual understanding"""
    
    def __init__(self, dimension=512, window_size=8, min_count=2, epochs=25):
        self.dimension = dimension
        self.window_size = window_size
        self.min_count = min_count
        self.epochs = epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Enhanced vocabulary with semantic groupings
        self.semantic_groups = {
            'divine_beings': ['शिव', 'पार्वती', 'विष्णु', 'कृष्ण', 'राम', 'ब्रह्मा', 'गणेश'],
            'natural_elements': ['हिमालय', 'गंगा', 'वृक्ष', 'पर्वत', 'नदी', 'वन'],
            'actions': ['गम्', 'दृश्', 'श्रु', 'वद्', 'कृ', 'भू', 'अस्'],
            'emotions': ['प्रेम', 'क्रोध', 'शान्ति', 'आनन्द', 'दुःख', 'भय', 'वीर'],
            'qualities': ['सुन्दर', 'महान्', 'दिव्य', 'शुभ', 'पवित्र', 'गुरु']
        }
        
    def create_enhanced_vocabulary(self, sentences):
        """Create vocabulary with semantic awareness"""
        vocab = defaultdict(int)
        
        # Count word frequencies
        for sentence in sentences:
            for word in sentence:
                vocab[word] += 1
        
        # Filter by minimum count
        filtered_vocab = {word: count for word, count in vocab.items() 
                         if count >= self.min_count}
        
        # Add semantic relationships
        self.semantic_similarities = self._compute_semantic_similarities(filtered_vocab)
        
        return filtered_vocab
    
    def _compute_semantic_similarities(self, vocab):
        """Compute semantic similarity scores for vocabulary"""
        similarities = defaultdict(dict)
        
        for group_name, words in self.semantic_groups.items():
            for word1 in words:
                if word1 in vocab:
                    for word2 in words:
                        if word2 in vocab and word1 != word2:
                            similarities[word1][word2] = 0.8  # High semantic similarity
        
        return similarities
    
    def train_enhanced_embeddings(self, sentences, vocab):
        """Train embeddings with enhanced context and semantic awareness"""
        print("🧠 Training Enhanced Sanskrit Embeddings...")
        
        # Create token mappings
        token_to_idx = {token: idx for idx, token in enumerate(vocab.keys())}
        vocab_size = len(vocab)
        
        # Initialize embeddings with Xavier initialization
        embeddings = nn.Parameter(torch.randn(vocab_size, self.dimension, device=self.device))
        nn.init.xavier_uniform_(embeddings)
        
        # Create enhanced context pairs with semantic awareness
        context_pairs = self._create_semantic_context_pairs(sentences, token_to_idx)
        
        # Optimizer
        optimizer = torch.optim.Adam([embeddings], lr=0.01)
        
        # Training loop
        for epoch in tqdm(range(self.epochs), desc="Training Embeddings"):
            total_loss = 0
            
            # Shuffle context pairs
            np.random.shuffle(context_pairs)
            
            for i in range(0, len(context_pairs), 64):  # Batch size 64
                batch_pairs = context_pairs[i:i+64]
                
                if not batch_pairs:
                    continue
                
                # Prepare batch
                center_words = []
                context_words = []
                
                for center, context in batch_pairs:
                    if center in token_to_idx and context in token_to_idx:
                        center_words.append(token_to_idx[center])
                        context_words.append(token_to_idx[context])
                
                if not center_words:
                    continue
                
                center_tensor = torch.tensor(center_words, device=self.device)
                context_tensor = torch.tensor(context_words, device=self.device)
                
                # Forward pass
                center_embeds = embeddings[center_tensor]
                context_embeds = embeddings[context_tensor]
                
                # Compute similarity scores
                scores = torch.sum(center_embeds * context_embeds, dim=1)
                
                # Generate negative samples
                neg_samples = torch.randint(0, vocab_size, (len(center_words), 5), device=self.device)
                neg_embeds = embeddings[neg_samples]
                neg_scores = torch.bmm(neg_embeds, center_embeds.unsqueeze(2)).squeeze(2)
                
                # Compute loss (skip-gram with negative sampling)
                pos_loss = -torch.log(torch.sigmoid(scores) + 1e-10).mean()
                neg_loss = -torch.log(1 - torch.sigmoid(neg_scores) + 1e-10).mean()
                
                loss = pos_loss + neg_loss
                
                # Add semantic regularization
                semantic_loss = self._compute_semantic_loss(embeddings, token_to_idx)
                total_batch_loss = loss + 0.1 * semantic_loss
                
                # Backward pass
                optimizer.zero_grad()
                total_batch_loss.backward()
                optimizer.step()
                
                total_loss += total_batch_loss.item()
            
            if epoch % 5 == 0:
                avg_loss = total_loss / max(len(context_pairs) // 64, 1)
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
        
        return embeddings.detach(), token_to_idx
    
    def _create_semantic_context_pairs(self, sentences, token_to_idx):
        """Create context pairs with semantic awareness"""
        pairs = []
        
        for sentence in sentences:
            for i, center_word in enumerate(sentence):
                if center_word not in token_to_idx:
                    continue
                
                # Standard context window
                start = max(0, i - self.window_size)
                end = min(len(sentence), i + self.window_size + 1)
                
                for j in range(start, end):
                    if j != i and j < len(sentence):
                        context_word = sentence[j]
                        if context_word in token_to_idx:
                            pairs.append((center_word, context_word))
                
                # Add semantic relationships
                if center_word in self.semantic_similarities:
                    for similar_word in self.semantic_similarities[center_word]:
                        if similar_word in token_to_idx:
                            pairs.append((center_word, similar_word))
        
        return pairs
    
    def _compute_semantic_loss(self, embeddings, token_to_idx):
        """Compute semantic regularization loss"""
        semantic_loss = 0
        count = 0
        
        for word1, similarities in self.semantic_similarities.items():
            if word1 not in token_to_idx:
                continue
                
            word1_idx = token_to_idx[word1]
            word1_embed = embeddings[word1_idx]
            
            for word2, similarity_score in similarities.items():
                if word2 not in token_to_idx:
                    continue
                
                word2_idx = token_to_idx[word2]
                word2_embed = embeddings[word2_idx]
                
                # Encourage similar embeddings for semantically related words
                cosine_sim = torch.cosine_similarity(word1_embed, word2_embed, dim=0)
                target_sim = torch.tensor(similarity_score, device=self.device)
                
                semantic_loss += (cosine_sim - target_sim) ** 2
                count += 1
        
        return semantic_loss / max(count, 1)

def enhanced_training_pipeline():
    """Enhanced training pipeline for better contextual understanding"""
    print("🕉️  Enhanced Sanskrit QA Training Pipeline")
    print("=" * 50)
    
    # Step 1: Load data with preprocessing
    print("\n📚 Step 1: Loading and preprocessing data...")
    sentences = fetch_kumarasambhava_conllu_files()
    
    if not sentences or len(sentences) < 10:
        # Use enhanced sample data
        sentences = [
            ['asty', 'uttarasyāṃ', 'diśi', 'devatātmā', 'himālayo', 'nāma', 'nagādhirājaḥ'],
            ['pūrvāparau', 'toyanidhī', 'vagāhya', 'madhye', 'kṣitīḍīśa', 'sa', 'śailah'],
            ['śailendrasya', 'sutā', 'devī', 'hemācalasya', 'pārvatī'],
            ['tapasā', 'śivasya', 'priyā', 'bhāryā', 'umā', 'nāma', 'gauryā'],
            ['kailāse', 'parvatarāje', 'śiva', 'eva', 'tapas', 'cakāra', 'mahat'],
            ['gaṅgā', 'himavataḥ', 'tanayā', 'snāna', 'puṇya', 'jala', 'pavitra'],
            ['brahma', 'viṣṇu', 'maheśa', 'trimūrti', 'sādhana', 'mokṣa', 'dāna'],
            ['ānanda', 'śānti', 'prema', 'karuna', 'dayā', 'kṣamā', 'satya'],
            ['vīrya', 'teja', 'bala', 'śakti', 'yoga', 'dhyāna', 'samādhi'],
            ['kavitā', 'śloka', 'chandas', 'rāga', 'tāla', 'saṅgīta', 'nṛtya']
        ] * 100  # Repeat for more training data
    
    # Preprocess sentences
    processed_sentences = []
    for sentence in sentences:
        # Clean and normalize
        clean_sentence = []
        for word in sentence:
            if len(word) > 1 and word.isalpha():
                clean_sentence.append(word.lower())
        if len(clean_sentence) >= 3:  # Keep sentences with at least 3 words
            processed_sentences.append(clean_sentence)
    
    print(f"📖 Processed {len(processed_sentences)} sentences")
    
    # Step 2: Enhanced embedding training
    print("\n🧠 Step 2: Training enhanced embeddings...")
    enhanced_embedder = EnhancedSanskritEmbeddings(
        dimension=512,
        window_size=8,
        min_count=2,
        epochs=30
    )
    
    vocab = enhanced_embedder.create_enhanced_vocabulary(processed_sentences)
    print(f"📝 Enhanced vocabulary size: {len(vocab)}")
    
    embeddings, token_to_idx = enhanced_embedder.train_enhanced_embeddings(processed_sentences, vocab)
    
    # Step 3: Analyze embedding quality
    print("\n📊 Step 3: Analyzing embedding quality...")
    analyze_embedding_quality(embeddings, token_to_idx, vocab)
    
    # Step 4: Initialize QA model with enhanced embeddings
    print("\n🏗️  Step 4: Initializing QA model with enhanced embeddings...")
    model = SanskritQuestionAnsweringSystem(
        vocab_size=len(vocab),
        d_model=512,
        num_layers=8,
        num_heads=8,
        d_ff=2048,
        dropout_rate=0.1
    )
    
    # Transfer embeddings to model
    with torch.no_grad():
        model.encoder.embedding.weight.copy_(embeddings)
        if hasattr(model, 'decoder'):
            model.decoder.embedding.weight.copy_(embeddings)
    
    print("✅ Enhanced embeddings transferred to model")
    
    # Step 5: Create enhanced QA trainer
    print("\n🎯 Step 5: Setting up enhanced QA trainer...")
    class SimpleTokenizer:
        def __init__(self, vocab, token_to_idx):
            self.vocab = vocab
            self.token_to_idx = token_to_idx
            self.vocab_size = len(vocab)
    
    tokenizer = SimpleTokenizer(vocab, token_to_idx)
    
    from train_sanskrit_qa import SanskritQATrainer
    trainer = SanskritQATrainer(model, tokenizer)
    
    # Step 6: Test enhanced understanding
    print("\n🧪 Step 6: Testing enhanced contextual understanding...")
    test_enhanced_understanding(trainer, embeddings, token_to_idx)
    
    return model, trainer, embeddings, token_to_idx

def analyze_embedding_quality(embeddings, token_to_idx, vocab):
    """Analyze the quality of trained embeddings"""
    print("📈 Analyzing Embedding Quality...")
    
    # Convert to numpy for analysis
    embeddings_np = embeddings.cpu().numpy()
    
    # Find most frequent words for analysis
    frequent_words = sorted(vocab.items(), key=lambda x: x[1], reverse=True)[:20]
    
    print("\n🔍 Most Similar Word Pairs:")
    print("-" * 40)
    
    for word, freq in frequent_words[:10]:
        if word in token_to_idx:
            word_idx = token_to_idx[word]
            word_embed = embeddings_np[word_idx:word_idx+1]
            
            # Compute similarities with all words
            similarities = cosine_similarity(word_embed, embeddings_np)[0]
            
            # Get top 5 similar words (excluding self)
            top_indices = np.argsort(similarities)[::-1][1:6]
            
            print(f"\n'{word}' is similar to:")
            for idx in top_indices:
                similar_word = list(token_to_idx.keys())[list(token_to_idx.values()).index(idx)]
                similarity_score = similarities[idx]
                print(f"  → {similar_word} ({similarity_score:.3f})")

def test_enhanced_understanding(trainer, embeddings, token_to_idx):
    """Test the enhanced contextual understanding"""
    test_cases = [
        {
            'context': 'asty uttarasyāṃ diśi devatātmā himālayo nāma nagādhirājaḥ',
            'questions': [
                'What is the main subject?',
                'Who are the characters mentioned?',
                'What divine elements are present?',
                'What is the sentiment?'
            ]
        },
        {
            'context': 'śailendrasya sutā devī hemācalasya pārvatī',
            'questions': [
                'Who is being described?',
                'What is her relationship?',
                'What divine aspects are mentioned?',
                'What emotions are conveyed?'
            ]
        },
        {
            'context': 'tapasā śivasya priyā bhāryā umā nāma gauryā',
            'questions': [
                'What spiritual practice is mentioned?',
                'Who are the divine characters?',
                'What relationship is described?',
                'What names are given?'
            ]
        }
    ]
    
    print("\n🎭 Testing Enhanced Understanding:")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        context = test_case['context']
        questions = test_case['questions']
        
        print(f"\n📖 Test Case {i}: {context}")
        print("-" * 60)
        
        for question in questions:
            answer = trainer._generate_contextual_answer(context, question, "enhanced")
            print(f"❓ {question}")
            print(f"💡 {answer}")
            print()

if __name__ == "__main__":
    enhanced_training_pipeline()
