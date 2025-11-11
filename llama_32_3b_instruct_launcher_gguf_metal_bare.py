#!/usr/bin/env python3
# %% [0.0] Launcher Script Info
# Minimal launcher script metadata
# Llama 3.2 3B Instruct GGUF Launcher for macOS - BARE BONES VERSION
# Description: Minimal code needed to run the model with Metal acceleration
# Author: dougeeai
# Created: 2025-11-09
# Last Updated: 2025-11-11

# %% [0.1] Model Card & Summary
# Bare bones version - minimal code, no checks
# MODEL: Llama-3.2-3B-Instruct
# Architecture: Llama 3.2 (3.21B parameters)

# %% [1.0] Core Imports
# Essential imports only
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
# Bare Version: Utility imports skipped

# %% [2.0] Base Directory Configuration
# Set base directory for portability
BASE_DIR = os.path.expanduser("~/Documents/ai")  # <-- CHANGE THIS to your AI folder location
MODELS_DIR = os.path.join(BASE_DIR, "models")

# %% [2.1] Model Source Configuration
# Simplified model path configuration
MODEL_NAME = "llama_32_3b_instruct_gguf"
MODEL_FILENAME = "llama_32_3b_instruct_q8_0.gguf"  # Change for different quants
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_NAME, MODEL_FILENAME)

# %% [2.2] User Configuration - All Settings
# Core settings for model operation
N_GPU_LAYERS = -1    # Use Metal GPU
N_CTX = 8192         # Context window
MAX_TOKENS = 2048    # Max response length
TEMPERATURE = 0.7    # Randomness
VERBOSE = False      # Set True for debug

# %% [2.3] Model Configuration Dataclass
# Bare Version: Dataclass skipped - using direct variables

# %% [3.0] Hardware Auto-Detection
# Bare Version: Auto-detection skipped

# %% [3.1] Hardware Detection
# Bare Version: Hardware detection skipped

# %% [3.2] Environment Validation
# Bare Version: Environment validation skipped

# %% [4.0] Model Loader
# Direct model loading function
def load_model():
    """Simple model loading"""
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
# Bare Version: Model validation skipped

# %% [5.0] Model Initialization
# Bare Version: Using direct load_model() instead

# %% [6.0] Inference Test
# Bare Version: Inference test skipped

# %% [6.1] Terminal Chat Interface
# Minimal chat loop with streaming
def chat_loop(model):
    """Minimal chat interface"""
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
            
            # Keep history manageable
            if len(messages) > 20:
                messages = messages[-20:]
            
        except Exception as e:
            print(f"\nError: {e}")

# %% [7.0] Optional Features
# Bare Version: Optional features skipped

# %% [8.0] Main Entry Point
# Simple main function - load and chat
def main():
    """Bare bones main - just load and chat"""
    print("=== Llama 3.2 3B Bare Bones Launcher ===\n")
    
    # Load model
    model = load_model()
    if not model:
        print("Failed to load model")
        return 1
    
    # Start chat
    chat_loop(model)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
