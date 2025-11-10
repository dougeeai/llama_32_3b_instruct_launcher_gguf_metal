#!/usr/bin/env python3
# %% [0.0] Launcher Script Info
"""
Llama 3.2 3B Instruct GGUF Launcher for macOS - BARE BONES VERSION
Minimal code needed to run the model with Metal acceleration
Author: DougeeAI
Date: November 2025
Python: 3.13
Metal: Enabled
"""

# %% [0.1] Model Card & Summary
"""
Model: Llama 3.2 3B Instruct
Quantization: Q8_0
File Size: ~3.4 GB
This bare-bones version runs the model with minimal features
"""

# %% [1.0] Core Imports
"""
Essential imports only - no fancy UI or monitoring
"""
import os
import sys
from pathlib import Path

# Core requirement
try:
    from llama_cpp import Llama
except ImportError:
    print("Error: llama-cpp-python not installed with Metal support!")
    print("Please run: CMAKE_ARGS='-DLLAMA_METAL=on' pip install llama-cpp-python --no-cache-dir")
    sys.exit(1)

# %% [1.1] Utility Imports
"""
Bare version - no utility imports needed
"""
# Basic version only - this cell not needed

# %% [2.0] User Configuration - All Settings
"""
Minimal settings for bare-bones operation
"""
# Model Path - UPDATE THIS IF YOUR FILENAME IS DIFFERENT
MODEL_PATH = os.path.expanduser("~/Documents/ai/models/llama_32_3b_instruct_gguf/llama_32_3b_instruct_q8_0.gguf") #Update with model location

# Essential Settings Only
N_GPU_LAYERS = -1    # Use Metal GPU
N_CTX = 8192         # Context window
MAX_TOKENS = 2048    # Max response length
TEMPERATURE = 0.7    # Randomness
VERBOSE = False      # Set True for debug

# %% [2.1] Model Configuration Dataclass
"""
Bare version - no dataclass needed, using simple variables
"""
# Basic version only - this cell not needed

# %% [2.2] Model Path Validation
"""
Simple check that model file exists
"""
def validate_model():
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        print(f"Error: Model not found at: {model_path}")
        print("Please check the path and filename")
        return False
    print(f"Found model: {model_path.name}")
    return True

# %% [2.3] Model Paths - HF Download (Optional)
"""
Bare version - manual download only
"""
# Basic version only - this cell not needed
# Download manually from: https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF

# %% [3.0] Hardware Auto-Detection
"""
Bare version - using defaults, no auto-detection
"""
# Basic version only - this cell not needed

# %% [3.1] Hardware Detection
"""
Bare version - no hardware detection
"""
# Basic version only - this cell not needed

# %% [3.2] Environment Validation
"""
Simple check that llama-cpp-python is available
"""
def check_environment():
    try:
        from llama_cpp import Llama
        print("llama-cpp-python is installed")
        return True
    except ImportError:
        print("llama-cpp-python not found")
        return False

# %% [4.0] Model Loader
"""
Simple model loading - no class wrapper
"""
def load_model():
    print("Loading model...")
    try:
        model = Llama(
            model_path=MODEL_PATH,
            n_ctx=N_CTX,
            n_gpu_layers=N_GPU_LAYERS,
            verbose=VERBOSE,
            chat_format="llama-3"
        )
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# %% [4.1] Model Validation
"""
Bare version - validation done in load_model
"""
# Basic version only - this cell not needed

# %% [5.0] Model Initialization
"""
Bare version - initialization done in load_model
"""
# Basic version only - this cell not needed

# %% [6.0] Inference Test
"""
Quick test to verify model works
"""
def test_model(model):
    print("\nTesting model...")
    try:
        response = model("Hello! Please respond with a brief greeting.", 
                        max_tokens=50)
        text = response['choices'][0]['text']
        print(f"Test response: {text}")
        return True
    except Exception as e:
        print(f"Test failed: {e}")
        return False

# %% [6.1] Terminal Chat Interface
"""
Minimal chat loop
"""
def chat_loop(model):
    print("\n=== Simple Chat (type 'quit' to exit) ===\n")
    
    messages = []
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
            
        if not user_input:
            continue
        
        # Add user message
        messages.append({"role": "user", "content": user_input})
        
        # Generate response
        print("Assistant: ", end="", flush=True)
        try:
            response = model.create_chat_completion(
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                stream=True
            )
            
            # Stream the response
            full_response = ""
            for chunk in response:
                if chunk['choices'][0]['delta'].get('content'):
                    text = chunk['choices'][0]['delta']['content']
                    print(text, end="", flush=True)
                    full_response += text
            print()  # New line after response
            
            # Add to conversation history
            messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            print(f"\nError: {e}")

# %% [7.0] Optional Features
"""
Bare version - no optional features
"""
# Basic version only - this cell not needed

# %% [8.0] Main Entry Point
"""
Minimal startup sequence
"""
def main():
    print("=== Llama 3.2 3B Bare Bones Launcher ===\n")
    
    # Check environment
    if not check_environment():
        return 1
    
    # Validate model path
    if not validate_model():
        return 1
    
    # Load model
    model = load_model()
    if not model:
        return 1
    
    # Test model
    if not test_model(model):
        print("Warning: Test failed but continuing anyway...")
    
    # Start chat
    chat_loop(model)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())