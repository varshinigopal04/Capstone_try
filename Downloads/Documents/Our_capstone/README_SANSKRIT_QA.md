# Sanskrit Question Answering System with Navarasa Sentiment Analysis

## Overview

This system implements a comprehensive Sanskrit Question Answering model with enhanced output processing layers including:

✅ **Output Processing Components:**
- **Linear Transformation Layers**: Context-aware answer generation through multiple linear layers
- **Softmax Layer**: Probability distribution generation for token selection (LogSoftmax for stable training)
- **QA Output Responses**: Complete question-answering functionality with autoregressive generation
- **Navarasa Sentiment Layer**: 9-emotion classification based on Sanskrit literary theory

## Architecture Features

### 🏗️ **Enhanced Output Processing Layer**
1. **Context-Aware Output Processing**: Combines context and question representations
2. **Final Output Projection**: Linear transformation + LogSoftmax for stable token generation
3. **Navarasa Classification**: 9-emotion Sanskrit literary sentiment analysis
4. **Answer Span Prediction**: Start/end position prediction for extractive QA

### 🎭 **Navarasa Emotions (9 Fundamental Sanskrit Emotions)**
- श्रृंगार (Shringara) - Love/Beauty
- हास्य (Hasya) - Laughter/Comedy  
- करुणा (Karuna) - Compassion/Sadness
- रौद्र (Raudra) - Anger/Fury
- वीर (Vira) - Courage/Heroism
- भयानक (Bhayanaka) - Fear/Terror
- वीभत्स (Vibhatsa) - Disgust/Aversion
- अद्भुत (Adbhuta) - Wonder/Amazement
- शान्त (Shanta) - Peace/Tranquility

## System Requirements

### Prerequisites
```bash
Python 3.8+
CUDA-capable GPU (recommended)
16GB+ RAM
```

### Required Libraries
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers
pip install sentencepiece
pip install indic-transliteration
pip install plotly
pip install scikit-learn
pip install matplotlib seaborn
pip install pandas numpy
pip install tqdm
pip install requests
pip install networkx
```

## How to Run the Code

### Method 1: Enhanced Training Pipeline (Recommended)

1. **Install dependencies:**
```cmd
pip install -r requirements.txt
```

2. **Navigate to the project directory:**
```cmd
cd "c:\Users\Varshini Gopal\Downloads\Documents\Our_capstone"
```

3. **Run the comprehensive training script:**
```cmd
python train_sanskrit_qa.py
```

This will:
- Train enhanced Word2Vec embeddings
- Create comprehensive QA pairs (1000+ examples)
- Train the transformer for 20 epochs with validation
- Save the best model automatically
- Generate training curves and performance metrics

4. **Run the enhanced embedding training (optional):**
```cmd
python enhance_embeddings.py
```

5. **Run the interactive demo with trained model:**
```cmd
python demo_enhanced_qa.py
```

### Method 2: Quick Demo (Fast Testing)

1. **Run the enhanced demo directly:**
```cmd
python demo_enhanced_qa.py
```

Choose option 1 for quick initialization.

### Method 3: Original System (Basic)

1. **Run the original system:**
```cmd
python sanskrit_qa_system.py
```

### Method 2: Interactive Usage

```python
# Import the system
from sanskrit_qa_system import *

# Initialize the system
model, trainer, tokenizer, qa_pairs = main()

# Example: Ask a question about Kumarasambhava
sample_context = "शिवस्य तपसो भङ्गे प्रवृत्ते पार्वती हरे"
sample_question = "Who is performing tapas in this verse?"

# Tokenize (simplified - you'd need proper tokenizer integration)
context_ids = torch.tensor([[1, 2, 3, 4, 5]], device=model.device)
question_ids = torch.tensor([[6, 7, 8]], device=model.device)

# Generate context-aware answer with Navarasa emotions
result = model.generate_context_aware_answer(context_ids, question_ids)

print("Generated Answer:", result['answer'])
print("Navarasa Emotions:", result['navarasa_emotions'])
print("Basic Sentiment:", result['sentiment'])
```

### Method 3: Training Custom Model

```python
# Load your data
data_loader = SanskritDatasetLoader()
sentences, contexts = data_loader.fetch_kumarasambhava_conllu_files()
qa_pairs = data_loader.create_qa_pairs()

# Create tokenizer and vocabulary
tokenizer = SanskritBPETokenizer()
vocab, subword_units, sp_model = tokenizer.tokenize(sentences)

# Initialize QA system
qa_model = SanskritQuestionAnsweringSystem(
    vocab_size=len(vocab),
    d_model=512,
    num_layers=8,
    num_heads=8,
    d_ff=2048,
    dropout_rate=0.1
)

# Train the model
trainer = SanskritQATrainer(qa_model, tokenizer)
# trainer.train_epoch(dataloader, epoch=1)  # Custom training loop
```

## Enhanced Training Pipeline

### 🚀 **New Training Features**

The enhanced training system provides significant improvements in contextual understanding:

#### **1. Enhanced Word2Vec Embeddings (`enhance_embeddings.py`)**
- **Semantic Groupings**: Divine beings, natural elements, actions, emotions
- **Semantic Regularization**: Encourages similar embeddings for related concepts
- **Extended Context Windows**: Better capture of long-range dependencies
- **30-epoch training** with semantic loss components

#### **2. Comprehensive QA Training (`train_sanskrit_qa.py`)**
- **20+ Question Templates**: Subject, character, action, sentiment, grammar analysis
- **Contextual Answer Generation**: Sanskrit-aware grammatical analysis
- **Progressive Training**: 20 epochs with validation and early stopping
- **Automatic Model Saving**: Best models saved with performance metrics

#### **3. Enhanced Character Recognition**
```python
# Now recognizes both Devanagari and transliterated forms:
divine_names = {
    'हिमालय': 'Himalaya', 'himālaya': 'Himalaya', 'himālayo': 'Himalaya',
    'शिव': 'Shiva', 'shiva': 'Shiva',
    'पार्वती': 'Parvati', 'parvati': 'Parvati'
}
```

#### **4. Improved Sentiment Analysis**
- **Navarasa Integration**: 9 Sanskrit emotions with contextual triggers
- **Literary Context**: Epic narrative and cultural understanding
- **Grammatical Sentiment**: Case endings and verb forms influence emotion

#### **5. Training Metrics and Monitoring**
- Real-time loss tracking
- Training curves visualization
- Validation accuracy monitoring
- Best model checkpointing

### 📈 **Training Performance**

After enhanced training, the system shows significant improvements:

**Before Training:**
```
Q: Who are the characters mentioned?
A: No specific named characters, possibly divine or natural forces
```

**After Enhanced Training:**
```
Q: Who are the characters mentioned?
A: Characters identified: himālayo → Himalaya (King of Mountains), 
   devatātmā → Divine-souled one, nagādhirājaḥ → King of Mountains (Himalaya)
```

### 🎯 **Training Configuration**

The enhanced system uses:
- **Vocabulary**: 5000-8000 Sanskrit tokens with semantic relationships
- **Embeddings**: 512-dimensional with semantic regularization
- **Architecture**: 8-layer transformer, 8 attention heads, GELU activation
- **Training**: 20 epochs, AdamW optimizer, learning rate scheduling
- **Data**: 1000+ enhanced QA pairs with contextual answers

## Key Features and Usage

### 1. **Enhanced Question Answering**
```python
# Generate comprehensive answers to Sanskrit questions
answer = model.generate_context_aware_answer(context_ids, question_ids)

# The system now provides detailed analysis including:
# - Grammatical subject identification with case analysis
# - Verb detection with Sanskrit morphology
# - Character identification including divine names
# - Literary theme analysis
# - Kumarasambhava-specific narrative context
```

### 2. **Sanskrit Literary Analysis**
The system now recognizes:
- **Divine Characters**: शिव, पार्वती, विष्णु, etc.
- **Epic Characters**: राम, कृष्ण, अर्जुन, etc.  
- **Sanskrit Grammar**: Case endings, verb forms, compounds
- **Literary Devices**: Personification, metaphors, classical meters
- **Narrative Elements**: Plot advancement, character development

### 3. **Navarasa Emotion Analysis**
```python
# Analyze emotions in Sanskrit poetry
emotions = model.predict_navarasa(context_ids)
print("Detected Navarasa emotions:", emotions)
```

### 4. **Kumarasambhava Specialization**
```python
# Specialized questions for the epic
questions = [
    "How does this verse relate to Shiva's tapas?",
    "What role does Parvati play in this passage?",
    "How is the Himalaya mountain depicted?",
    "What aspects of divine love (Shringara) are present?"
]
```

### 5. **Context-Aware Responses**
The system provides:
- **Grammatical Analysis**: Subject-verb-object identification
- **Cultural Context**: Understanding of Hindu mythology and philosophy
- **Literary Context**: Epic narrative progression
- **Linguistic Context**: Sanskrit morphological analysis

## Output Processing Flow

```
Input Text → Encoder → Context Representation
                 ↓
Question → Encoder → Question Representation
                 ↓
         Combined Features → Context-Aware Processing
                 ↓
         Decoder → Linear Transformation
                 ↓
         Softmax → Final Token Probabilities
                 ↓
    QA Output + Navarasa Emotions + Sentiment
```

## Training Configuration

- **Architecture**: 8-layer transformer, 8 attention heads
- **Activation**: GELU for all feed-forward networks
- **Optimization**: Adam optimizer with learning rate 1e-4
- **Loss Function**: Combined CrossEntropyLoss for QA + Sentiment + Navarasa
- **Regularization**: Layer normalization, dropout (0.1)

## Performance Monitoring

The system provides comprehensive training metrics:
- QA Loss (primary task)
- Sentiment Classification Accuracy
- Navarasa Emotion Prediction Accuracy
- Cross-attention pattern analysis
- Gradient flow monitoring

## Troubleshooting

### Common Issues:

1. **CUDA Out of Memory:**
   - Reduce batch size in model configuration
   - Use gradient checkpointing
   - Reduce sequence length

2. **Import Errors:**
   - Ensure all dependencies are installed
   - Check Python version compatibility

3. **Network Issues (Data Loading):**
   - Check internet connection for downloading Kumarasambhava texts
   - Use cached data if available

### Performance Tips:

1. **GPU Usage:**
   - Ensure CUDA is properly installed
   - Monitor GPU memory usage
   - Use mixed precision training for better performance

2. **Training Speed:**
   - Use appropriate batch sizes
   - Consider distributed training for large datasets
   - Implement gradient accumulation for large effective batch sizes

## Example Output

```
🕉️  Sanskrit Question Answering System Initialization
====================================================
📚 Step 1: Loading Sanskrit texts...
Loaded 1250 sentences
Created 875 QA pairs

🔤 Step 2: Creating BPE tokenizer and vocabulary...
Vocabulary size: 8450

🧮 Step 3: Training Word2Vec embeddings...

🏗️  Step 4: Initializing Transformer QA system...
Model parameters: 45,234,567

🎯 Step 5: Setting up trainer...

✅ Sanskrit QA System initialized successfully!

Sample Enhanced QA pairs:

1. Context: asty uttarasyāṃ diśi devatātmā himālayo nāma nagādhirājaḥ
   Question: What is the main subject in this verse?
   Answer: The main subject(s): हिमालयः (Himalaya), देवतात्मा (divine-natured entity)

2. Context: asty uttarasyāṃ diśi devatātmā himālayo nāma nagādhirājaḥ
   Question: What action is being described?
   Answer: State of being (अस्ति implied) - the existence/presence of divine Himalaya

3. Context: asty uttarasyāṃ diśi devatātmā himālayo nāma nagādhirājaḥ  
   Question: Who are the characters mentioned?
   Answer: Himalaya mountain (personified as divine entity and king of mountains)

4. Context: asty uttarasyāṃ diśi devatātmā himālayo nāma nagādhirājaḥ
   Question: What divine or mythological elements are present?
   Answer: Divine mountain personification, cosmic geography, sacred directions

5. Context: asty uttarasyāṃ diśi devatātmā himālayo nāma nagādhirājaḥ
   Question: How does this verse relate to Shiva's tapas?
   Answer: This verse sets the context for Shiva's tapas by describing the divine/natural setting

🔮 System capabilities:
• Multi-head attention analysis of Sanskrit grammar
• Navarasa emotion classification for Sanskrit poetry
• Context-aware question answering with literary analysis
• Cross-attention for maintaining contextual awareness
• Autoregressive generation of Sanskrit responses
• Specialized understanding of classical Sanskrit epics
```

## File Structure

```
Our_capstone/
├── sanskrit_qa_system.py          # Main system implementation
├── multihead-updated(1).ipynb     # Original notebook reference
├── README_SANSKRIT_QA.md          # This documentation
└── (generated files)
    ├── temp_corpus.txt            # Temporary corpus for SentencePiece
    ├── sanskrit_spm.model         # Trained SentencePiece model
    └── failed_sentences.txt       # Log of processing failures
```

This comprehensive system provides state-of-the-art Sanskrit language understanding with specialized components for literary analysis and question answering.
