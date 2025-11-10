# Llama 3.2 3B Instruct GGUF Launcher for macOS (Metal)

Quick and dirty launcher scripts for running Llama-3.2-3B-Instruct GGUF models on macOS with Metal GPU acceleration.

## Description

Python scripts optimized for Apple Silicon Macs (M1/M2/M3/M4) to run Llama 3.2 3B Instruct GGUF models with Metal GPU acceleration. Includes a full version with hardware detection and a bare-bones version for minimal setup.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4) or Intel Mac with Metal support
- Python 3.13
- ~4-6GB unified memory
- GGUF model file (Q8_0 recommended)

## Setup

1. Create conda environment:
```bash
conda env create -f environment.yml
conda activate llama_32_3b_instruct_launcher_gguf_metal
```

2. Install llama-cpp-python with Metal support:
```bash
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --no-cache-dir
```

3. **IMPORTANT: Update model path in the script**
```python
MODEL_PATH = "~/Documents/ai/models/llama_32_3b_instruct_gguf/llama_32_3b_instruct_q8_0.gguf"  # Change to your model location
```

4. Run:
```bash
python llama_32_3b_instruct_launcher_gguf_metal.py  # Full version
# or
python llama_32_3b_instruct_launcher_gguf_metal_bare.py  # Minimal version
```

## Files

- `llama_32_3b_instruct_launcher_gguf_metal.py` - Full launcher with hardware detection, Rich UI, conversation saving
- `llama_32_3b_instruct_launcher_gguf_metal_bare.py` - Bare minimum code to load and chat
- `environment.yml` - Conda environment for macOS with Metal dependencies

## Configuration

Key settings in the script:
- `N_GPU_LAYERS` - Set to -1 for full Metal offload
- `N_CTX` - Context window (8192 default, up to 131072)
- `N_THREADS` - CPU threads (auto-detects by default)

## Usage

Type messages at the prompt. Commands:
- `quit` - Exit the chat
- `/clear` - Reset conversation history (full version)
- `/save` - Save conversation to JSON (full version)
- `/help` - Show commands (full version)

## macOS Specific Notes

- Metal acceleration works on all Apple Silicon chips
- Unified memory architecture means no separate VRAM limit
- Compilation of llama-cpp-python takes 5-10 minutes
- Performance scales with chip: M4 > M3 > M2 > M1
- Intel Macs with AMD GPUs also support Metal acceleration
