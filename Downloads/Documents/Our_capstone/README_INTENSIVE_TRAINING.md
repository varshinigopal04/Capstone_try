# Intensive Kumārasaṃbhava Sanskrit QA Training System

## Overview

This comprehensive training system provides intensive 50-epoch training on the Kumārasaṃbhava corpus with enhanced contextual understanding, morphological analysis, and sophisticated question-answering capabilities.

## 🎯 Key Features

### Advanced Corpus Processing
- **Comprehensive CoNLL-U Analysis**: Detailed parsing of morphological features, case endings, and grammatical relationships
- **Character & Divine Entity Recognition**: Automatic identification of epic characters and divine entities
- **Compound Word Analysis**: Sophisticated breakdown of Sanskrit compound structures
- **Spatial & Temporal Context**: Enhanced understanding of locations and narrative elements

### Enhanced QA Generation
- **15+ Question Types**: Including subject extraction, character identification, sentiment analysis, grammatical features
- **Navarasa Sentiment Analysis**: Nine-emotion classification system for Sanskrit literature
- **Morphological QA**: Questions about case endings, verb forms, and grammatical structures
- **Epic Context Analysis**: Questions relating to broader narrative themes

### Intensive Training Framework
- **50-Epoch Training**: Extended training for deep contextual understanding
- **Learning Rate Scheduling**: Cosine annealing with warm restarts
- **Comprehensive Validation**: Train/validation splits with early stopping
- **Progress Monitoring**: Real-time loss tracking and visualization

## 📁 File Structure

```
├── train_kumarasambhava_intensive.py    # Core intensive training script
├── run_intensive_training.py            # Integrated training runner
├── demo_intensive_qa.py                 # Interactive demo with trained model
├── config_intensive.json               # Training configuration
├── requirements_intensive.txt           # Python dependencies
├── files/Kumārasaṃbhava/              # Corpus directory
│   ├── Kumārasaṃbhava-0000-*.conllu   # CoNLL-U files
│   ├── Kumārasaṃbhava-0001-*.conllu
│   └── ...
└── sanskrit_qa_system.py              # Base QA system (existing)
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_intensive.txt
```

### 2. Prepare Corpus
Ensure the Kumārasaṃbhava CoNLL-U files are in the `files/Kumārasaṃbhava/` directory:
```
files/Kumārasaṃbhava/Kumārasaṃbhava-0000-KumSaṃ, 1-7305.conllu
files/Kumārasaṃbhava/Kumārasaṃbhava-0001-KumSaṃ, 2-7312.conllu
...
```

### 3. Run Intensive Training
```bash
# Full integrated training (recommended)
python run_intensive_training.py

# Or standalone intensive training
python train_kumarasambhava_intensive.py
```

### 4. Test with Demo
```bash
python demo_intensive_qa.py
```

## 📊 Training Configuration

### Default Settings (config_intensive.json)
```json
{
  "training_config": {
    "num_epochs": 50,
    "batch_size": 8,
    "learning_rate": 1e-4,
    "validation_split": 0.1,
    "qa_pairs_per_sentence": 8
  },
  "model_config": {
    "embedding_dim": 512,
    "num_layers": 8,
    "num_attention_heads": 8,
    "dropout_rate": 0.1
  }
}
```

### Hardware Requirements
- **GPU**: CUDA-capable GPU with 8GB+ VRAM (recommended)
- **RAM**: 16GB+ system RAM
- **Storage**: 10GB+ free space for models and checkpoints

## 🧠 Model Architecture

### Enhanced Transformer QA System
- **8-Layer Encoder-Decoder**: Deep transformer architecture
- **8 Attention Heads**: Multi-head attention for complex relationships
- **GELU Activation**: Modern activation function for better gradients
- **Add & Norm Components**: Layer normalization with residual connections

### Output Processing Layers
- **Linear Transformation**: Context-aware output projection
- **Softmax/LogSoftmax**: Stable probability distributions
- **QA Response Generation**: Specialized answer generation
- **Navarasa Classification**: 9-emotion sentiment analysis

## 📚 Corpus Analysis

### Kumārasaṃbhava Corpus Statistics
- **~8 Cantos**: Complete epic coverage
- **Thousands of Verses**: Comprehensive vocabulary
- **Rich Annotations**: Morphological, syntactic, and semantic features

### Enhanced Features Extracted
1. **Morphological Analysis**
   - Case endings (Nominative, Accusative, Locative, etc.)
   - Verb tenses and moods
   - Gender and number agreement

2. **Character Recognition**
   - Epic characters (Himālaya, Pārvatī, Śiva, etc.)
   - Divine entities and their attributes
   - Relationships and hierarchies

3. **Literary Analysis**
   - Compound word structures
   - Metrical patterns
   - Narrative themes

## 🎭 Question Types Generated

### 1. Subject & Object Analysis
- "What is the main subject in this verse?"
- "What objects are mentioned?"
- "Who performs the action?"

### 2. Character Identification
- "Who are the characters mentioned?"
- "Which divine entities are present?"
- "What relationships are described?"

### 3. Grammatical Analysis
- "What case endings are used?"
- "What is the main verb?"
- "What grammatical mood is expressed?"

### 4. Semantic Analysis
- "What action is being described?"
- "What locations are mentioned?"
- "What natural elements are present?"

### 5. Literary Analysis
- "What compound words are present?"
- "What is the emotional tone?"
- "How does this relate to epic narrative?"

### 6. Navarasa Sentiment
- **Adbhuta** (Wonder/Awe): Divine manifestations
- **Shanta** (Peace/Tranquility): Meditative passages
- **Vira** (Heroism): Epic actions
- **Shringara** (Love/Beauty): Romantic elements

## 🔧 Training Process

### Phase 1: Corpus Loading (5-10 minutes)
- Parse all CoNLL-U files
- Extract morphological features
- Build comprehensive vocabulary
- Identify characters and entities

### Phase 2: QA Dataset Creation (10-15 minutes)
- Generate 8+ QA pairs per sentence
- Apply enhanced analysis functions
- Create contextual annotations
- Validate QA pair quality

### Phase 3: Model Initialization (1-2 minutes)
- Initialize transformer architecture
- Load pre-trained embeddings (if available)
- Setup training components

### Phase 4: Intensive Training (2-6 hours)
- 50 epochs with progress monitoring
- Validation after each epoch
- Checkpoint saving every 10 epochs
- Early stopping if converged

### Phase 5: Evaluation & Testing (5 minutes)
- Model performance analysis
- Sample QA testing
- Generate training report

## 📈 Monitoring & Output

### Training Outputs
- `best_kumarasambhava_model.pth`: Best model based on validation loss
- `final_kumarasambhava_model.pth`: Final model with metadata
- `training_report.json`: Comprehensive training statistics
- `kumarasambhava_training.log`: Detailed training log
- `training_progress_*.png`: Loss and learning rate plots

### Checkpoints
- `checkpoint_epoch_10.pth`, `checkpoint_epoch_20.pth`, etc.
- Include full training state for resumption

## 🎯 Expected Improvements

### Enhanced Contextual Understanding
- **Character Recognition**: 95%+ accuracy for major epic characters
- **Divine Entity Detection**: Comprehensive identification of divine aspects
- **Grammatical Analysis**: Detailed morphological feature extraction

### Better Answer Quality
- **Specific Answers**: Detailed explanations instead of generic responses
- **Cultural Context**: Epic narrative understanding
- **Linguistic Analysis**: Grammatical and morphological insights

### Example Improvements

#### Before Training:
```
Question: Who are the characters mentioned?
Answer: No specific named characters
```

#### After Intensive Training:
```
Question: Who are the characters mentioned?
Answer: Characters identified: himālayo → Himalaya (King of Mountains), 
        devatātmā → Divine-souled entity; nagādhirāja → Supreme Mountain King
```

## 🛠️ Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce batch_size in config
   - Use CPU instead of GPU for smaller datasets

2. **File Not Found**
   - Verify CoNLL-U files in correct directory
   - Check file permissions

3. **Training Slow**
   - Ensure CUDA is available
   - Reduce sequence length or model size

### Performance Optimization
- Use mixed precision training
- Enable gradient checkpointing
- Adjust batch size based on GPU memory

## 📖 Usage Examples

### Interactive Demo
```python
# Run the demo
python demo_intensive_qa.py

# Test with verse
Context: "asty uttarasyāṃ diśi devatātmā himālayo nāma nagādhirājaḥ"
Question: "What is the main subject?"
Answer: "Main subjects identified: himālaya (Himalaya, the divine mountain), 
         devatātmā (divine-souled entity)"
```

### Programmatic Usage
```python
from demo_intensive_qa import IntensivelyTrainedQADemo

demo = IntensivelyTrainedQADemo("best_kumarasambhava_model.pth")
answer = demo.generate_contextual_answer(context, question)
```

## 🔬 Advanced Features

### Custom Question Types
Add new question types by extending the `EnhancedSanskritQADataset` class:

```python
def _extract_custom_feature(self, sentence: Dict) -> str:
    # Your custom analysis logic
    return "Custom analysis result"
```

### Model Fine-tuning
Resume training from checkpoints:

```python
trainer = IntensiveTrainer(model, dataset)
trainer.load_checkpoint("checkpoint_epoch_30.pth")
trainer.train_intensive(num_epochs=20)  # Continue for 20 more epochs
```

## 📋 Requirements

### Python Dependencies
```
torch>=1.12.0
transformers>=4.21.0
sentencepiece>=0.1.97
matplotlib>=3.5.2
seaborn>=0.11.2
pandas>=1.4.3
numpy>=1.21.0
tqdm>=4.64.0
colorama>=0.4.5
```

### System Requirements
- Python 3.8+
- CUDA 11.0+ (for GPU training)
- 16GB+ RAM
- 10GB+ storage space

## 🤝 Contributing

Contributions welcome! Please focus on:
- Additional question types
- Improved Sanskrit linguistic analysis
- Performance optimizations
- Documentation improvements

## 📜 License

This project builds upon existing Sanskrit corpus work and follows appropriate attribution guidelines for academic research.

## 🙏 Acknowledgments

- Digital Corpus of Sanskrit (DCS) for the Kumārasaṃbhava corpus
- Sanskrit computational linguistics community
- Open source transformer architecture contributions

---

*"Through intensive training on the sacred verses of Kumārasaṃbhava, may this system serve the understanding of Sanskrit literature."* 🕉️
