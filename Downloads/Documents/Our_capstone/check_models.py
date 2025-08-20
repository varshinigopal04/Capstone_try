#!/usr/bin/env python3
"""
Model Verification Script
========================

This script checks if the trained models exist and shows their metadata.
"""

import torch
from pathlib import Path
import json

def check_model_files():
    """Check for trained model files and display their metadata."""
    
    model_files = [
        "best_kumarasambhava_model.pth",
        "final_kumarasambhava_model.pth", 
        "integrated_kumarasambhava_model.pth"
    ]
    
    print("🔍 Checking for trained model files...")
    print("=" * 60)
    
    found_models = []
    
    for model_file in model_files:
        if Path(model_file).exists():
            print(f"✅ Found: {model_file}")
            found_models.append(model_file)
            
            try:
                # Load and display metadata
                checkpoint = torch.load(model_file, map_location='cpu')
                
                print(f"   📊 Model Statistics:")
                print(f"   - Vocabulary size: {len(checkpoint.get('vocabulary', {}))}")
                print(f"   - Token mappings: {len(checkpoint.get('token_to_id', {}))}")
                print(f"   - Character names: {len(checkpoint.get('character_names', []))}")
                print(f"   - Divine entities: {len(checkpoint.get('divine_entities', []))}")
                
                if 'training_stats' in checkpoint:
                    stats = checkpoint['training_stats']
                    print(f"   - Final training loss: {stats.get('final_loss', 'N/A')}")
                    print(f"   - Best validation loss: {stats.get('best_val_loss', 'N/A')}")
                    print(f"   - Training epochs: {stats.get('epochs_completed', 'N/A')}")
                
                print()
                
            except Exception as e:
                print(f"   ❌ Error loading metadata: {e}")
                print()
        else:
            print(f"❌ Not found: {model_file}")
    
    if not found_models:
        print("\n⚠️  No trained models found!")
        print("Please run train_kumarasambhava_intensive.py first.")
    else:
        print(f"\n✅ Found {len(found_models)} trained model(s)")
    
    # Check for training report
    if Path("training_report.json").exists():
        print("\n📋 Training Report:")
        print("-" * 30)
        with open("training_report.json", 'r', encoding='utf-8') as f:
            report = json.load(f)
            print(f"Training completed: {report.get('training_completed', False)}")
            print(f"Final epoch: {report.get('final_epoch', 'N/A')}")
            print(f"Best loss: {report.get('best_loss', 'N/A')}")
            print(f"Total training time: {report.get('total_training_time', 'N/A')}")

if __name__ == "__main__":
    check_model_files()
