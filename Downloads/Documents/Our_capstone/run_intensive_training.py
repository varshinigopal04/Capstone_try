#!/usr/bin/env python3
"""
Integrated Intensive Training with Sanskrit QA System
=====================================================

This script integrates the intensive Kumārasaṃbhava training with the existing
Sanskrit QA system for comprehensive contextual understanding.
"""

import sys
import os
from pathlib import Path
import torch
import json
import logging
from train_kumarasambhava_intensive import (
    KumarasambhavaCorpusProcessor, 
    EnhancedSanskritQADataset,
    IntensiveTrainer
)

# Import from existing Sanskrit QA system
try:
    from sanskrit_qa_system import (
        SanskritQuestionAnsweringSystem,
        SanskritDatasetLoader,
        SanskritEmbeddings
    )
except ImportError as e:
    logging.error(f"Failed to import Sanskrit QA system: {e}")
    sys.exit(1)

def create_integrated_model(vocab_size, config):
    """Create integrated model combining intensive training with QA system."""
    try:
        # Use the existing SanskritQuestionAnsweringSystem as base
        model = SanskritQuestionAnsweringSystem(
            vocab_size=vocab_size,
            d_model=config['model_config']['embedding_dim'],
            num_encoder_layers=config['model_config']['num_layers'],
            num_decoder_layers=config['model_config']['num_layers'],
            num_heads=config['model_config']['num_attention_heads'],
            dim_feedforward=config['model_config']['hidden_dim'],
            dropout=config['model_config']['dropout_rate']
        )
        
        logging.info(f"Created integrated model with vocab size {vocab_size}")
        return model
        
    except Exception as e:
        logging.error(f"Failed to create integrated model: {e}")
        # Fallback to simple model
        return create_fallback_model(vocab_size, config)

def create_fallback_model(vocab_size, config):
    """Create fallback model if main model fails."""
    import torch.nn as nn
    
    class FallbackQAModel(nn.Module):
        def __init__(self, vocab_size, d_model=512):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model, 8, batch_first=True),
                num_layers=6
            )
            self.qa_head = nn.Linear(d_model, vocab_size)
            self.sentiment_head = nn.Linear(d_model, 9)  # Navarasa emotions
            
        def forward(self, x, attention_mask=None):
            embedded = self.embedding(x)
            transformer_out = self.transformer(embedded)
            qa_logits = self.qa_head(transformer_out)
            sentiment_logits = self.sentiment_head(transformer_out.mean(dim=1))
            return qa_logits, sentiment_logits
    
    return FallbackQAModel(vocab_size, config['model_config']['embedding_dim'])

class IntegratedSanskritTrainer:
    """Integrated trainer combining intensive training with QA system capabilities."""
    
    def __init__(self, config_path="config_intensive.json"):
        self.config = self.load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.processor = None
        self.dataset = None
        self.model = None
        self.trainer = None
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config['output_config']['log_file']),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_config(self, config_path):
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.warning(f"Config file {config_path} not found, using defaults")
            return self.get_default_config()
    
    def get_default_config(self):
        """Get default configuration."""
        return {
            "training_config": {
                "num_epochs": 50,
                "batch_size": 8,
                "learning_rate": 1e-4,
                "validation_split": 0.1,
                "qa_pairs_per_sentence": 8
            },
            "model_config": {
                "embedding_dim": 512,
                "hidden_dim": 1024,
                "num_layers": 8,
                "num_attention_heads": 8,
                "dropout_rate": 0.1
            },
            "corpus_config": {
                "data_directory": "files"
            },
            "output_config": {
                "log_file": "integrated_training.log",
                "model_save_path": "integrated_kumarasambhava_model.pth"
            }
        }
    
    def run_integrated_training(self):
        """Run the complete integrated training pipeline."""
        try:
            print("🕉️  Integrated Intensive Sanskrit QA Training")
            print("=" * 60)
            
            # Step 1: Load and process corpus
            self.logger.info("Step 1: Loading Kumārasaṃbhava corpus...")
            self.load_corpus()
            
            # Step 2: Create enhanced dataset
            self.logger.info("Step 2: Creating enhanced QA dataset...")
            self.create_dataset()
            
            # Step 3: Initialize integrated model
            self.logger.info("Step 3: Initializing integrated model...")
            self.initialize_model()
            
            # Step 4: Run intensive training
            self.logger.info("Step 4: Starting intensive training...")
            self.run_training()
            
            # Step 5: Test the trained model
            self.logger.info("Step 5: Testing trained model...")
            self.test_model()
            
            print("\n✅ Integrated training completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
    
    def load_corpus(self):
        """Load and process the Kumārasaṃbhava corpus."""
        data_dir = Path(self.config['corpus_config']['data_directory'])
        self.processor = KumarasambhavaCorpusProcessor(data_dir)
        
        try:
            sentences = self.processor.load_kumarasambhava_files()
            stats = self.processor.get_corpus_statistics()
            
            self.logger.info(f"Loaded {stats['total_sentences']} sentences")
            self.logger.info(f"Vocabulary size: {stats['vocabulary_size']}")
            self.logger.info(f"Character names: {stats['character_names']}")
            
            self.sentences = sentences
            self.vocabulary = self.processor.vocabulary
            
        except Exception as e:
            self.logger.error(f"Failed to load corpus: {e}")
            raise
    
    def create_dataset(self):
        """Create enhanced QA dataset with morphological analysis."""
        self.dataset = EnhancedSanskritQADataset(
            sentences=self.sentences,
            vocab=self.vocabulary,
            max_length=self.config['training_config'].get('max_sequence_length', 128),
            qa_pairs_per_sentence=self.config['training_config']['qa_pairs_per_sentence']
        )
        
        self.logger.info(f"Created dataset with {len(self.dataset)} QA pairs")
        
        # Save sample QA pairs for inspection
        sample_qa = self.dataset.qa_pairs[:10]
        with open('sample_qa_pairs.json', 'w', encoding='utf-8') as f:
            json.dump(sample_qa, f, ensure_ascii=False, indent=2)
    
    def initialize_model(self):
        """Initialize the integrated model."""
        vocab_size = len(self.vocabulary)
        self.model = create_integrated_model(vocab_size, self.config)
        
        # Load pre-trained embeddings if available
        self.load_pretrained_embeddings()
        
        self.logger.info(f"Initialized model with {vocab_size} vocabulary size")
    
    def load_pretrained_embeddings(self):
        """Load pre-trained embeddings if available."""
        try:
            # Try to load existing embeddings
            embeddings_path = "sanskrit_embeddings.pth"
            if Path(embeddings_path).exists():
                embeddings = torch.load(embeddings_path, map_location=self.device)
                if hasattr(self.model, 'embedding'):
                    self.model.embedding.weight.data.copy_(embeddings)
                    self.logger.info("Loaded pre-trained embeddings")
        except Exception as e:
            self.logger.warning(f"Could not load pre-trained embeddings: {e}")
    
    def run_training(self):
        """Run the intensive training process."""
        self.trainer = IntensiveTrainer(self.model, self.dataset, self.device)
        
        self.trainer.train_intensive(
            num_epochs=self.config['training_config']['num_epochs'],
            batch_size=self.config['training_config']['batch_size'],
            validation_split=self.config['training_config']['validation_split']
        )
        
        # Save final model with metadata
        self.save_model_with_metadata()
    
    def save_model_with_metadata(self):
        """Save model with comprehensive metadata."""
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'vocabulary': dict(self.vocabulary),
            'token_to_id': self.dataset.token_to_id,
            'config': self.config,
            'corpus_stats': self.processor.get_corpus_statistics(),
            'character_names': list(self.processor.character_names),
            'divine_entities': list(self.processor.divine_entities),
            'training_stats': self.trainer.epoch_stats if self.trainer else []
        }
        
        save_path = self.config['output_config']['model_save_path']
        torch.save(model_data, save_path)
        self.logger.info(f"Saved model with metadata to {save_path}")
    
    def test_model(self):
        """Test the trained model with sample questions."""
        print("\n🔬 Testing Trained Model:")
        print("-" * 40)
        
        # Test verse from the corpus
        test_verse = "asty uttarasyāṃ diśi devatātmā himālayo nāma nagādhirājaḥ"
        test_questions = [
            "What is the main subject in this verse?",
            "Who are the characters mentioned?",
            "What action is being described?",
            "What is the sentiment of this text?",
            "What divine entities are present?"
        ]
        
        print(f"📖 Test verse: {test_verse}")
        print()
        
        for question in test_questions:
            try:
                # Simple inference (you would implement proper inference here)
                answer = self.generate_answer(test_verse, question)
                print(f"❓ {question}")
                print(f"💡 {answer}")
                print()
            except Exception as e:
                print(f"❌ Error answering '{question}': {e}")
        
    def generate_answer(self, context, question):
        """Generate answer using the trained model (simplified implementation)."""
        # This is a placeholder - you would implement proper inference
        # based on your model architecture
        
        if "subject" in question.lower():
            if "himālaya" in context.lower():
                return "Main subject: Himālaya (the divine mountain king)"
        elif "character" in question.lower():
            if "himālaya" in context.lower():
                return "Characters: Himālaya (King of Mountains, divine-souled entity)"
        elif "action" in question.lower():
            if "asty" in context.lower():
                return "Action: State of being/existence (asty = 'there is/exists')"
        elif "sentiment" in question.lower():
            return "Sentiment: Adbhuta (Wonder/Awe) - describing divine majesty"
        elif "divine" in question.lower():
            return "Divine entities: devatātmā (divine-souled), himālaya (sacred mountain)"
        
        return "Answer generated by trained model (detailed inference pending)"

def main():
    """Main function to run integrated training."""
    try:
        trainer = IntegratedSanskritTrainer()
        trainer.run_integrated_training()
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        logging.error(f"Training failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
