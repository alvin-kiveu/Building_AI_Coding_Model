# Building_AI_Coding_Model

# Building an AI Coding Model from Scratch: A Beginner's Guide

## Table of Contents
1. [Prerequisites & Setup](#prerequisites--setup)
2. [Understanding Language Models](#understanding-language-models)
3. [Architecture Design](#architecture-design)
4. [Data Preparation](#data-preparation)
5. [Building the Model](#building-the-model)
6. [Training Pipeline](#training-pipeline)
7. [Evaluation & Testing](#evaluation--testing)
8. [Deployment](#deployment)
9. [Next Steps](#next-steps)

---

## Prerequisites & Setup

### What You Need to Know
Before starting, familiarize yourself with:
- **Python basics** (variables, functions, loops, libraries)
- **Basic math** (nothing beyond high school algebra)
- **Command line usage** (navigating folders, running scripts)
- **Git** (cloning repos, basic commands)

### Required Tools & Installation

#### 1. Python 3.9+
```bash
# Check if installed
python3 --version

# Install from python.org if needed
```

#### 2. Core Libraries
```bash
# Create a virtual environment (recommended)
python3 -m venv coding_model_env
source coding_model_env/bin/activate  # On Windows: coding_model_env\Scripts\activate

# Install essential packages
pip install torch transformers datasets numpy pandas jupyter
pip install accelerate tensorboard wandb
```

**Why each library:**
- `torch` – Deep learning framework (easier than alternatives)
- `transformers` – Pre-built model architectures from Hugging Face
- `datasets` – Easy dataset loading & preprocessing
- `accelerate` – Simplifies multi-GPU training
- `wandb` – Tracks training progress and metrics

#### 3. Hardware Considerations
- **Minimum**: GPU with 6GB VRAM (NVIDIA GTX 1660 or similar)
- **Better**: 12GB+ VRAM (RTX 3060 or better)
- **No GPU?** Start with smaller models; CPU training is slow but possible

---

## Understanding Language Models

### The Big Picture
An AI coding model is a **transformer-based language model** trained on code. Think of it as:
- **Input**: A few lines of code or a comment
- **Process**: Predicts the most likely next tokens (pieces of code)
- **Output**: Generates the next line(s) of code

### How It Works (Simplified)
1. **Tokenization** – Convert code text into numbers the model understands
2. **Embeddings** – Map tokens to meaning vectors
3. **Attention** – Model learns which parts are important (context)
4. **Prediction** – Generate likely next tokens one at a time

### Key Concepts
- **Token** – A small piece of text (word, symbol, or code element)
- **Transformer** – Architecture using "attention" to understand context
- **Fine-tuning** – Adapting a pre-trained model for specific tasks
- **Loss** – A measure of how wrong the model is (lower is better)

---

## Architecture Design

### Option 1: Start Small (Recommended for Beginners)
**Model Size**: ~125 Million Parameters
- Training time: 1-3 weeks on single GPU
- Inference speed: Fast (~50-100 tokens/second)
- Memory: ~1.5GB to run
- Good for: Learning, experimentation, prototyping

```
Input Code → Tokenizer → Embeddings (768 dim) → 
12 Transformer Layers → Attention Heads (12) → 
Output Logits → Softmax → Generated Code
```

### Option 2: Medium (If You Have Resources)
**Model Size**: ~350 Million Parameters
- Training time: 2-4 weeks on single GPU
- Better code understanding
- Requires more VRAM (6GB+)

### Option 3: Larger (Advanced)
**Model Size**: ~1-3 Billion Parameters
- Training time: Weeks on multiple GPUs
- Much better at complex coding tasks
- Requires 24GB+ VRAM or multiple GPUs

### Recommended: Use Hugging Face Transformers
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load a small pre-trained model
model_name = "gpt2"  # Start with GPT-2 (124M parameters)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
```

---

## Data Preparation

### Where to Find Code Datasets

#### 1. **The Stack** (Recommended)
- 3.3 billion files in 258 programming languages
- Free and open-source
- Available on Hugging Face Datasets
```python
from datasets import load_dataset

# Load a subset (example: Python only, 10GB)
dataset = load_dataset("bigcode/the-stack", 
                       data_dir="data/python", 
                       split="train",
                       streaming=True)
```

#### 2. **CodeSearchNet**
- Code + documentation from GitHub
- 6M+ functions in multiple languages
- Good for code + comments

#### 3. **GitHub Data**
- Directly from GitHub API (requires crawling)
- License considerations (many open-source projects)

#### 4. **Other Options**
- Hugging Face Hub: Pre-processed code datasets
- Kaggle: Various code competition datasets
- Academic sources: Papers often release datasets

### Data Cleaning & Preprocessing

```python
import json
from datasets import load_dataset

def clean_code(code):
    """Remove duplicates, fix encoding, filter noise"""
    # Remove very long files (likely generated/minified)
    if len(code) > 100_000:
        return None
    
    # Remove files that are mostly comments or empty
    lines = code.split('\n')
    non_comment_ratio = len([l for l in lines if l.strip() and not l.strip().startswith('#')]) / max(len(lines), 1)
    if non_comment_ratio < 0.3:
        return None
    
    return code

# Load and clean
dataset = load_dataset("bigcode/the-stack", 
                      data_dir="data/python", 
                      split="train[:5%]")  # Start with 5%

cleaned_dataset = dataset.filter(lambda x: clean_code(x['content']) is not None)
cleaned_dataset = cleaned_dataset.map(lambda x: {'text': clean_code(x['content'])})
```

### Tokenization Strategy

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Special tokens for code
tokenizer.add_tokens(['<|endoffile|>', '<|startoftext|>'])

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        max_length=1024,  # Context window
        truncation=True,
        return_tensors='pt'
    )

tokenized_dataset = cleaned_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=['content', 'repo_name']  # Remove unnecessary columns
)
```

---

## Building the Model

### Quick Start: Using Hugging Face

```python
from transformers import (
    GPT2Config, 
    GPT2LMHeadModel, 
    AutoTokenizer
)
import torch

# Configuration for a small model
config = GPT2Config(
    vocab_size=50257,
    n_positions=1024,  # Context window (max input length)
    n_embd=768,        # Embedding dimension
    n_layer=12,        # Number of transformer layers
    n_head=12,         # Number of attention heads
    intermediate_size=3072,  # Feed-forward network size
)

# Create model
model = GPT2LMHeadModel(config)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
```

### Custom Training Script

```python
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

class CodeModelTrainer:
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
    
    def setup_optimizer(self, learning_rate=5e-5, warmup_steps=1000, total_steps=10000):
        """Initialize optimizer and learning rate scheduler"""
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        return optimizer, scheduler
    
    def train_epoch(self, train_dataloader, optimizer, scheduler):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc="Training")
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids  # Causal language modeling
            )
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Optimize
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(train_dataloader)

# Example usage
trainer = CodeModelTrainer(model, tokenizer)
optimizer, scheduler = trainer.setup_optimizer()

for epoch in range(3):
    avg_loss = trainer.train_epoch(train_dataloader, optimizer, scheduler)
    print(f"Epoch {epoch}: Avg Loss = {avg_loss:.4f}")
```

---

## Training Pipeline

### Full Training Setup

```python
from transformers import Trainer, TrainingArguments
import torch

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./code_model_checkpoint",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,  # Adjust based on GPU memory
    per_device_eval_batch_size=8,
    save_steps=500,
    eval_steps=500,
    logging_steps=100,
    learning_rate=5e-5,
    warmup_steps=1000,
    weight_decay=0.01,
    save_total_limit=3,
    load_best_model_at_end=True,
    evaluation_strategy="steps",
    fp16=True,  # Mixed precision training (faster)
    report_to=["wandb"],  # Track on Weights & Biases
    remove_unused_columns=False,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
)

# Train!
trainer.train()
```

### Important Training Tips

1. **Batch Size**: Start with 8, increase if you have VRAM
2. **Learning Rate**: 1e-5 to 5e-5 works well for code
3. **Epochs**: 2-3 epochs is usually enough
4. **Checkpoints**: Save frequently (every 500-1000 steps)
5. **Mixed Precision**: Use `fp16=True` to train faster with less memory
6. **Gradient Accumulation**: If batch size is too small:
   ```python
   gradient_accumulation_steps=4  # Accumulate 4 batches before update
   ```

### Monitor Training
```python
# Install and login to Weights & Biases
pip install wandb
wandb login

# Then training automatically uploads metrics
# View at https://wandb.ai/your-username/your-project
```

---

## Evaluation & Testing

### Generate Code (Inference)

```python
def generate_code(prompt, model, tokenizer, max_length=100):
    """Generate code given a prompt"""
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            temperature=0.7,  # Lower = more deterministic
            top_p=0.9,  # Nucleus sampling
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Test it
prompt = "def fibonacci(n):\n    "
result = generate_code(prompt, model, tokenizer)
print(result)
```

### Evaluation Metrics

```python
# Install evaluation library
pip install evaluate

from evaluate import load

# BLEU score (how similar to human code)
bleu = load("bleu")
predictions = ["the cat sat on the mat"]
references = [["the cat sat on the mat"]]
results = bleu.compute(predictions=predictions, references=references)
print(results)

# Perplexity (how surprised the model is by test data)
import numpy as np

def calculate_perplexity(model, tokenizer, text):
    input_ids = tokenizer.encode(text, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    return torch.exp(loss).item()

perplexity = calculate_perplexity(model, tokenizer, test_code)
print(f"Perplexity: {perplexity:.2f}")
```

### Testing on Real Code Tasks

```python
test_cases = [
    ("def add(a, b):\n    ", "return a + b"),
    ("for i in range(", "10):\n    print(i)"),
    ("import ", "numpy as np"),
]

for prompt, expected in test_cases:
    generated = generate_code(prompt, model, tokenizer, max_length=50)
    print(f"Prompt: {prompt}")
    print(f"Expected: {expected}")
    print(f"Generated: {generated}")
    print("---")
```

---

## Deployment

### Option 1: Local CLI Tool

```python
# save as code_generator.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class CodeGenerator:
    def __init__(self, model_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
    
    def generate(self, prompt, max_length=100):
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        output = self.model.generate(
            input_ids,
            max_length=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    import sys
    
    gen = CodeGenerator("./code_model_checkpoint")
    prompt = sys.argv[1] if len(sys.argv) > 1 else "def hello():\n    "
    result = gen.generate(prompt)
    print(result)

# Usage: python code_generator.py "def fibonacci(n):\n    "
```

### Option 2: Flask Web API

```python
# save as app.py
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)

# Load model once on startup
tokenizer = AutoTokenizer.from_pretrained("./code_model_checkpoint")
model = AutoModelForCausalLM.from_pretrained("./code_model_checkpoint")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')
    max_length = data.get('max_length', 100)
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    output = model.generate(input_ids, max_length=max_length, temperature=0.7)
    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return jsonify({'generated_code': generated})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

# Usage:
# curl -X POST http://localhost:5000/generate -H "Content-Type: application/json" -d '{"prompt":"def add(a, b):\n    "}'
```

### Option 3: Hugging Face Model Hub

```bash
# Install transformers-cli
pip install huggingface-hub

# Push your model
huggingface-cli login
huggingface-cli upload your-username/your-code-model ./code_model_checkpoint --repo-type model

# Anyone can now load it:
# from transformers import AutoModelForCausalLM
# model = AutoModelForCausalLM.from_pretrained("your-username/your-code-model")
```

---

## Next Steps

### Phase 1: Get Started (Week 1)
- [ ] Set up Python environment and libraries
- [ ] Download a small dataset (10-100MB)
- [ ] Fine-tune a small pre-trained model (GPT-2)
- [ ] Generate code and test basic functionality

### Phase 2: Improve (Week 2-3)
- [ ] Expand dataset size
- [ ] Experiment with hyperparameters
- [ ] Evaluate on test set
- [ ] Deploy as a CLI tool

### Phase 3: Optimize (Week 4+)
- [ ] Increase model size
- [ ] Train from scratch (advanced)
- [ ] Create web interface
- [ ] Publish on Hugging Face

### Advanced Topics (Later)
- Multi-GPU training with DistributedDataParallel
- Quantization for faster inference
- Fine-tuning on specific domains (Python, Web, etc.)
- Adding instruction fine-tuning
- Reinforcement learning from human feedback (RLHF)

---

## Common Issues & Solutions

### Out of Memory (OOM)
```python
# Reduce batch size
per_device_train_batch_size=4

# Use gradient accumulation
gradient_accumulation_steps=4

# Enable mixed precision
fp16=True

# Use smaller model
n_layer=6  # Instead of 12
n_embd=512  # Instead of 768
```

### Training is Too Slow
- Use GPU (CUDA required)
- Reduce dataset size for first experiments
- Use mixed precision training
- Increase batch size (if VRAM allows)

### Poor Code Generation
- Train longer (more epochs)
- Use larger model
- Better data preprocessing
- Adjust temperature (0.5 = more conservative, 1.0 = more creative)

### Model Won't Train
- Check dataset format (should have 'text' column)
- Verify tokenizer is working
- Check GPU availability: `torch.cuda.is_available()`
- Look at loss—should decrease over time

---

## Useful Resources

### Learning
- Hugging Face Course: https://huggingface.co/course
- PyTorch Tutorials: https://pytorch.org/tutorials
- Jay Alammar's Blog: https://jalammar.github.io (Great explanations!)

### Data
- The Stack Dataset: https://huggingface.co/datasets/bigcode/the-stack
- CodeSearchNet: https://github.com/github/CodeSearchNet

### Communities
- Hugging Face Forums: https://discuss.huggingface.co
- Papers with Code: https://paperswithcode.com
- Reddit r/MachineLearning, r/LocalLLMs

### Papers (For Deep Dives)
- "Attention is All You Need" (Transformers)
- "Language Models are Unsupervised Multitask Learners" (GPT-2)
- "Codex" (OpenAI's code model)

---

## Summary

Building an AI coding model involves:
1. **Preparation**: Set up Python, install libraries, gather code data
2. **Design**: Choose model size and architecture
3. **Data**: Clean and tokenize code datasets
4. **Building**: Create transformer model using Hugging Face
5. **Training**: Train on GPU using standard training pipelines
6. **Evaluation**: Test generation quality with benchmarks
7. **Deployment**: Share as CLI, API, or on Hugging Face Hub

Start small, learn fast, and iterate. Good luck! 🚀
