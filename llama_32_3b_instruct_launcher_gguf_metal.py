#!/usr/bin/env python3
# %% [0.0] Launcher Script Info
# Script metadata and documentation for version tracking
# Llama 3.2 3B Instruct GGUF Launcher for macOS with Metal Acceleration
# Description: Optimized for Apple Silicon (M1/M2/M3/M4) with Metal GPU acceleration
# Author: dougeeai
# Created: 2025-11-09
# Last Updated: 2025-11-11
# Optimized for: Python 3.13 + Metal

# %% [0.1] Model Card & Summary
# Quick reference for model capabilities and requirements
# MODEL: Llama-3.2-3B-Instruct
# Architecture: Llama 3.2 (3.21B parameters)
# Context: 128K capable (trained on 8k)
# Best For: General chat, instruction following, creative writing, coding assistance
# Memory Requirements: ~4-6 GB unified memory on Apple Silicon
# Performance: Excellent on M1/M2/M3/M4 with Metal acceleration

# %% [1.0] Core Imports
# Essential Python libraries required for GGUF operation
import os
import sys
import json
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager

# Suppress warnings
warnings.filterwarnings("ignore")

# Llama-cpp-python with Metal support
try:
    from llama_cpp import Llama
except ImportError:
    print("Error: llama-cpp-python not installed with Metal support!")
    print("Please run: CMAKE_ARGS='-DLLAMA_METAL=on' pip install llama-cpp-python --no-cache-dir")
    sys.exit(1)

# %% [1.1] Utility Imports
# Supporting libraries for hardware monitoring and performance metrics
import time
import psutil
import platform
import subprocess
from datetime import datetime

# Rich imports for better terminal output
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich import print as rprint

# Initialize Rich console
console = Console()

# %% [2.0] Base Directory Configuration
# Set base AI directory - all paths will be relative to this location

# Set your base directory (change this for your system)
# macOS standard location in Documents
BASE_DIR = os.path.expanduser("~/Documents/ai")  # <-- CHANGE THIS to your AI folder location

# Directory structure (automatically created from BASE_DIR)
MODELS_DIR = os.path.join(BASE_DIR, "models")
HF_DOWNLOADS_DIR = os.path.join(MODELS_DIR, "huggingface_downloads")  # For HF downloads
HF_CACHE_DIR = os.path.join(HF_DOWNLOADS_DIR, "cache")  # HF cache

# Create directories if they don't exist
for dir_path in [MODELS_DIR, HF_DOWNLOADS_DIR, HF_CACHE_DIR]:
    Path(dir_path).mkdir(parents=True, exist_ok=True)

# %% [2.1] Model Source Configuration
# Configure where to load the model from - local file or HuggingFace download

# Choose model source: "local" or "huggingface"
MODEL_SOURCE = "local"  # Options: "local" or "huggingface"

# Model identification
MODEL_NAME = "llama_32_3b_instruct_gguf"  # Folder name for this model
MODEL_FILENAME = "llama_32_3b_instruct_q8_0.gguf"  # Actual GGUF filename

# Local file configuration (for manually downloaded models)
# Local models go directly in: BASE_DIR/models/MODEL_NAME/
LOCAL_MODEL_PATH = os.path.join(MODELS_DIR, MODEL_NAME, MODEL_FILENAME)

# HuggingFace configuration
HF_REPO_ID = "bartowski/Llama-3.2-3B-Instruct-GGUF"
HF_FILENAME = "Llama-3.2-3B-Instruct-Q8_0.gguf"  # Filename on HuggingFace
# HuggingFace models will be saved to: models/huggingface_downloads/MODEL_NAME/
HF_MODEL_DIR = os.path.join(HF_DOWNLOADS_DIR, MODEL_NAME)

# Resolve actual model path based on source
if MODEL_SOURCE == "huggingface":
    try:
        from huggingface_hub import hf_hub_download
        console.print(f"[cyan]Downloading model from HuggingFace: {HF_REPO_ID}/{HF_FILENAME}[/cyan]")
        console.print(f"[cyan]Download location: {HF_MODEL_DIR}[/cyan]")
        console.print(f"[cyan]Cache location: {HF_CACHE_DIR}[/cyan]")
        
        MODEL_PATH = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            cache_dir=HF_CACHE_DIR,
            local_dir=HF_MODEL_DIR,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        console.print(f"[green]Model downloaded to: {MODEL_PATH}[/green]")
    except ImportError:
        console.print("[red]ERROR: huggingface-hub not installed. Run: pip install huggingface-hub[/red]")
        console.print("[yellow]Falling back to local path...[/yellow]")
        MODEL_PATH = LOCAL_MODEL_PATH
    except Exception as e:
        console.print(f"[red]ERROR downloading from HuggingFace: {e}[/red]")
        console.print("[yellow]Falling back to local path...[/yellow]")
        MODEL_PATH = LOCAL_MODEL_PATH
else:
    MODEL_PATH = LOCAL_MODEL_PATH
    if not Path(MODEL_PATH).exists():
        console.print(f"[yellow]WARNING: Local model not found at: {MODEL_PATH}[/yellow]")
        console.print(f"[yellow]Expected location: {LOCAL_MODEL_PATH}[/yellow]")
        console.print(f"[cyan]To download from HuggingFace, set MODEL_SOURCE = 'huggingface'[/cyan]")

# %% [2.2] User Configuration - All Settings
# Central location for all user-modifiable model and generation settings

# Hardware Settings
N_GPU_LAYERS = -1        # -1 = offload all layers to Metal GPU
N_THREADS = None         # None = auto-detect based on CPU cores
N_BATCH = 512           # Batch size for prompt processing
N_CTX = 8192            # Context window size

# Memory Settings
USE_MMAP = True         # Memory-mapped file I/O (efficient on macOS)
USE_MLOCK = False       # Don't lock memory (not needed on Apple Silicon)
F16_KV = True          # Use 16-bit key/value cache

# Generation Parameters
MAX_TOKENS = 2048
TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K = 40
REPEAT_PENALTY = 1.1
SEED = -1              # -1 for random seed

# Performance Flags
OFFLOAD_KQV = True     # Offload K, Q, V matrices to GPU
MUL_MAT_Q = True       # Use quantized matrix multiplication

# Chat Configuration
CHAT_FORMAT = "llama-3"
SYSTEM_MESSAGE = "You are a helpful AI assistant."

# Debug Options
VERBOSE = False        # Set to True for debugging output
SHOW_TIMINGS = True    # Show generation timings

# Generation Presets (alternative to manual settings above)
GENERATION_PRESETS = {
    "precise": {
        "temperature": 0.1,
        "top_p": 0.95,
        "top_k": 40,
        "repeat_penalty": 1.1
    },
    "balanced": {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.1
    },
    "creative": {
        "temperature": 1.2,
        "top_p": 0.95,
        "top_k": 100,
        "repeat_penalty": 1.0
    }
}

# Select a preset (None = use manual settings above)
USE_PRESET = None  # Options: None, "precise", "balanced", "creative"

# %% [2.3] Model Configuration Dataclass
# Structured container for passing configuration to model loader

@dataclass
class ModelConfig:
    """Configuration for Llama model with Metal acceleration"""
    # Model
    model_path: str = MODEL_PATH
    
    # Hardware
    n_gpu_layers: int = N_GPU_LAYERS
    n_threads: Optional[int] = N_THREADS
    n_batch: int = N_BATCH
    n_ctx: int = N_CTX
    
    # Memory
    use_mmap: bool = USE_MMAP
    use_mlock: bool = USE_MLOCK
    f16_kv: bool = F16_KV
    
    # Generation
    max_tokens: int = MAX_TOKENS
    temperature: float = TEMPERATURE
    top_p: float = TOP_P
    top_k: int = TOP_K
    repeat_penalty: float = REPEAT_PENALTY
    seed: int = SEED
    
    # Performance
    offload_kqv: bool = OFFLOAD_KQV
    mul_mat_q: bool = MUL_MAT_Q
    
    # Chat
    chat_format: str = CHAT_FORMAT
    system_message: str = SYSTEM_MESSAGE
    
    # Debug
    verbose: bool = VERBOSE
    show_timings: bool = SHOW_TIMINGS
    
    def __post_init__(self):
        """Auto-configure settings based on hardware if needed"""
        if self.n_threads is None:
            self.n_threads = psutil.cpu_count(logical=False)

# %% [3.0] Hardware Auto-Detection
# Automatically determine optimal settings based on available hardware

def auto_detect_settings() -> Dict[str, Any]:
    """Detect hardware and suggest optimal settings"""
    settings = {}
    
    # CPU detection
    cpu_count = psutil.cpu_count(logical=False)
    settings['n_threads'] = cpu_count
    
    # Memory detection
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Suggest batch size based on memory
    if memory_gb >= 32:
        settings['n_batch'] = 512
    elif memory_gb >= 16:
        settings['n_batch'] = 256
    else:
        settings['n_batch'] = 128
    
    # Always use Metal on Apple Silicon
    settings['n_gpu_layers'] = -1  # All layers to GPU
    
    return settings

# %% [3.1] Hardware Detection
# Gather detailed hardware information for optimization decisions

class HardwareDetector:
    """Detect and display hardware information"""
    
    @staticmethod
    def get_cpu_info() -> Dict[str, Any]:
        """Get CPU information"""
        return {
            "name": platform.processor() or "Apple Silicon",
            "cores_physical": psutil.cpu_count(logical=False),
            "cores_logical": psutil.cpu_count(logical=True),
            "frequency": psutil.cpu_freq().current if psutil.cpu_freq() else 0,
        }
    
    @staticmethod
    def get_memory_info() -> Dict[str, Any]:
        """Get memory information"""
        mem = psutil.virtual_memory()
        return {
            "total_gb": mem.total / (1024**3),
            "available_gb": mem.available / (1024**3),
            "used_gb": mem.used / (1024**3),
            "percent": mem.percent,
        }
    
    @staticmethod
    def get_apple_silicon_info() -> Dict[str, Any]:
        """Get Apple Silicon specific information"""
        info = {}
        
        try:
            # Use system_profiler to get chip info
            result = subprocess.run(
                ["system_profiler", "SPHardwareDataType", "-json"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                hw_data = json.loads(result.stdout)
                hardware = hw_data.get("SPHardwareDataType", [{}])[0]
                
                info["chip"] = hardware.get("chip_type", "Unknown")
                info["cores_total"] = hardware.get("number_processors", "Unknown")
                info["memory"] = hardware.get("physical_memory", "Unknown")
                
                # Determine GPU cores based on chip
                chip_lower = str(info["chip"]).lower()
                if "m4" in chip_lower:
                    info["gpu_cores"] = "10-core GPU"  # M4 base
                elif "m3" in chip_lower:
                    info["gpu_cores"] = "10-core GPU"  # M3 base
                elif "m2" in chip_lower:
                    info["gpu_cores"] = "10-core GPU"  # M2 base
                elif "m1" in chip_lower:
                    info["gpu_cores"] = "8-core GPU"   # M1 base
                else:
                    info["gpu_cores"] = "Metal GPU"
                
        except Exception as e:
            if VERBOSE:
                console.print(f"[yellow]Could not get chip details: {e}[/yellow]")
        
        return info
    
    @staticmethod
    def display_info():
        """Display all hardware information"""
        table = Table(title="System Information", show_header=True, header_style="bold magenta")
        table.add_column("Component", style="cyan", no_wrap=True)
        table.add_column("Details", style="green")
        
        # CPU info
        cpu = HardwareDetector.get_cpu_info()
        table.add_row("CPU", f"{cpu['cores_physical']} cores ({cpu['cores_logical']} threads)")
        
        # Memory info
        mem = HardwareDetector.get_memory_info()
        table.add_row("Memory", f"{mem['total_gb']:.1f} GB total, {mem['available_gb']:.1f} GB available")
        
        # Apple Silicon info
        apple = HardwareDetector.get_apple_silicon_info()
        if apple.get("chip"):
            table.add_row("Chip", apple["chip"])
            table.add_row("GPU", apple.get("gpu_cores", "Metal GPU"))
        
        table.add_row("Platform", platform.platform())
        table.add_row("Python", platform.python_version())
        table.add_row("Metal", "Available" if "arm" in platform.machine().lower() else "Not Available")
        
        console.print(table)

# %% [3.2] Environment Validation
# Verify Python version and required packages before proceeding

def validate_environment() -> bool:
    """Check environment setup"""
    issues = []
    
    # Python version check
    py_version = sys.version_info
    if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 10):
        issues.append(f"Python 3.10+ required, found {py_version.major}.{py_version.minor}")
    
    # Check llama-cpp-python
    try:
        import llama_cpp
    except ImportError:
        issues.append("llama-cpp-python not installed")
    
    # Check for Metal support (Apple Silicon)
    if "arm" not in platform.machine().lower() and "apple" in platform.platform().lower():
        console.print("[yellow]Warning: Running on Intel Mac - Metal acceleration may be limited[/yellow]")
    
    if issues:
        console.print("[red]Environment issues found:[/red]")
        for issue in issues:
            console.print(f"  - {issue}")
        return False
    
    return True

# %% [4.0] Model Loader
# Class to handle GGUF model loading with optimal settings

class GGUFModelLoader:
    """Load and manage GGUF models"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.load_time = 0
    
    def load(self) -> bool:
        """Load the model with configuration"""
        try:
            console.print(Panel.fit(
                f"Loading: {Path(self.config.model_path).name}\n"
                f"Context: {self.config.n_ctx} tokens\n"
                f"GPU Layers: {'All' if self.config.n_gpu_layers == -1 else self.config.n_gpu_layers}\n"
                f"Threads: {self.config.n_threads}",
                title="Model Configuration",
                border_style="blue"
            ))
            
            start_time = time.time()
            
            # Load with progress indicator
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("[cyan]Loading model with Metal acceleration...", total=None)
                
                self.model = Llama(
                    model_path=self.config.model_path,
                    n_ctx=self.config.n_ctx,
                    n_batch=self.config.n_batch,
                    n_threads=self.config.n_threads,
                    n_gpu_layers=self.config.n_gpu_layers,
                    f16_kv=self.config.f16_kv,
                    use_mmap=self.config.use_mmap,
                    use_mlock=self.config.use_mlock,
                    verbose=self.config.verbose,
                    seed=self.config.seed,
                    offload_kqv=self.config.offload_kqv,
                    mul_mat_q=self.config.mul_mat_q,
                    chat_format=self.config.chat_format,
                )
                
                progress.update(task, completed=True)
            
            self.load_time = time.time() - start_time
            console.print(f"[green]Model loaded successfully in {self.load_time:.2f} seconds[/green]")
            
            return True
            
        except Exception as e:
            console.print(f"[red]Error loading model: {str(e)}[/red]")
            if "metal" in str(e).lower():
                console.print("[yellow]Ensure llama-cpp-python was compiled with Metal:[/yellow]")
                console.print("[yellow]CMAKE_ARGS='-DLLAMA_METAL=on' pip install llama-cpp-python --no-cache-dir[/yellow]")
            return False

# %% [4.1] Model Validation
# Verify GGUF file integrity before attempting to load

def validate_gguf_file(path: str) -> bool:
    """Check if file is valid GGUF format"""
    path = Path(path)
    
    if not path.exists():
        console.print(f"[red]Model file not found: {path}[/red]")
        return False
    
    # Check file size
    size_gb = path.stat().st_size / (1024**3)
    console.print(f"[green]Found model: {path.name} ({size_gb:.1f} GB)[/green]")
    
    try:
        with open(path, 'rb') as f:
            # GGUF files start with 'GGUF' magic bytes
            magic = f.read(4)
            if magic == b'GGUF':
                console.print("[green]Valid GGUF file detected[/green]")
                return True
            else:
                console.print(f"[red]Invalid GGUF file (magic bytes: {magic})[/red]")
                return False
    except Exception as e:
        console.print(f"[red]Error validating file: {e}[/red]")
        return False

# %% [5.0] Model Initialization
# Create and configure model instance with optional preset support

def initialize_model(config: Optional[ModelConfig] = None) -> Optional[GGUFModelLoader]:
    """Initialize the model with configuration"""
    
    if config is None:
        # Apply preset if selected
        gen_settings = {}
        if USE_PRESET and USE_PRESET in GENERATION_PRESETS:
            gen_settings = GENERATION_PRESETS[USE_PRESET]
            console.print(f"[cyan]Using generation preset: {USE_PRESET}[/cyan]")
        
        config = ModelConfig(
            model_path=MODEL_PATH,
            temperature=gen_settings.get('temperature', TEMPERATURE),
            top_p=gen_settings.get('top_p', TOP_P),
            top_k=gen_settings.get('top_k', TOP_K),
            repeat_penalty=gen_settings.get('repeat_penalty', REPEAT_PENALTY)
        )
    
    # Validate model file
    if not validate_gguf_file(config.model_path):
        return None
    
    # Create loader
    loader = GGUFModelLoader(config)
    
    # Load model
    if not loader.load():
        return None
    
    return loader

# %% [6.0] Inference Test
# Quick test to verify model works and measure performance

def test_inference(loader: GGUFModelLoader) -> bool:
    """Run a test inference"""
    console.print("\n[cyan]Running inference test...[/cyan]")
    
    test_prompt = "Hello! Please respond with a brief greeting."
    
    try:
        start_time = time.time()
        
        response = loader.model(
            test_prompt,
            max_tokens=50,
            temperature=0.7,
            stream=False
        )
        
        elapsed = time.time() - start_time
        
        text = response['choices'][0]['text']
        tokens_generated = response['usage']['completion_tokens']
        tokens_per_second = tokens_generated / elapsed if elapsed > 0 else 0
        
        console.print(f"[green]Test passed![/green]")
        console.print(f"Response: {text.strip()}")
        console.print(f"[dim]Generated {tokens_generated} tokens in {elapsed:.2f}s ({tokens_per_second:.1f} tok/s)[/dim]")
        
        return True
        
    except Exception as e:
        console.print(f"[red]Test failed: {e}[/red]")
        return False

# %% [6.1] Terminal Chat Interface
# Interactive chat loop with conversation history and streaming responses

class ChatInterface:
    """Interactive chat with conversation history"""
    
    def __init__(self, loader: GGUFModelLoader):
        self.loader = loader
        self.messages = []
        self.config = loader.config
        
        # Add system message
        if self.config.system_message:
            self.messages.append({
                "role": "system",
                "content": self.config.system_message
            })
    
    def chat(self, user_input: str) -> str:
        """Process a chat message"""
        # Add user message
        self.messages.append({"role": "user", "content": user_input})
        
        # Generate response
        response = self.loader.model.create_chat_completion(
            messages=self.messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            repeat_penalty=self.config.repeat_penalty,
            stream=True
        )
        
        # Stream response
        full_response = ""
        console.print("[bold green]Assistant:[/bold green] ", end="")
        
        for chunk in response:
            if chunk['choices'][0]['delta'].get('content'):
                text = chunk['choices'][0]['delta']['content']
                console.print(text, end="")
                full_response += text
        
        console.print()  # New line after response
        
        # Add assistant message to history
        self.messages.append({"role": "assistant", "content": full_response})
        
        # Keep conversation history manageable
        if len(self.messages) > 20:
            # Keep system message and last 18 messages
            self.messages = [self.messages[0]] + self.messages[-18:]
        
        return full_response
    
    def run(self):
        """Run the chat interface"""
        console.print(Panel.fit(
            "Chat Interface Started\n"
            "Commands: /clear (clear history), /save (save chat), /help, /quit",
            title="Llama 3.2 3B Chat",
            border_style="green"
        ))
        
        while True:
            try:
                # Get user input
                user_input = console.input("\n[bold blue]You:[/bold blue] ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['/quit', 'quit', 'exit']:
                    console.print("[yellow]Goodbye![/yellow]")
                    break
                elif user_input.lower() == '/clear':
                    self.messages = [self.messages[0]] if self.messages and self.messages[0]['role'] == 'system' else []
                    console.print("[yellow]Conversation cleared[/yellow]")
                    continue
                elif user_input.lower() == '/save':
                    self.save_conversation()
                    continue
                elif user_input.lower() == '/help':
                    self.show_help()
                    continue
                
                # Process chat
                self.chat(user_input)
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Use /quit to exit[/yellow]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
    
    def save_conversation(self):
        """Save conversation to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.messages, f, indent=2)
        
        console.print(f"[green]Conversation saved to {filename}[/green]")
    
    def show_help(self):
        """Show help message"""
        help_text = """
Commands:
  /help   - Show this help message
  /clear  - Clear conversation history  
  /save   - Save conversation to file
  /quit   - Exit the chat
        """
        console.print(Panel(help_text, title="Help", border_style="blue"))

# %% [7.0] Optional Features
# Additional capabilities like JSON mode and grammar constraints

# def create_json_grammar():
#     """Create grammar for JSON output"""
#     return {
#         "type": "json",
#         "schema": {
#             "type": "object",
#             "properties": {
#                 "response": {"type": "string"},
#                 "confidence": {"type": "number"},
#                 "reasoning": {"type": "string"}
#             },
#             "required": ["response", "confidence"]
#         }
#     }

# %% [8.0] Main Entry Point
# Orchestrate the entire launch sequence from validation to chat interface

def main():
    """Main entry point - runs all steps in order"""
    
    # Welcome message
    console.print(Panel.fit(
        "Llama 3.2 3B Instruct Launcher\n"
        "Optimized for Apple Silicon with Metal",
        title="Welcome",
        border_style="bold green"
    ))
    
    # 1. Validate environment
    console.print("\n[cyan]Step 1: Validating environment...[/cyan]")
    if not validate_environment():
        return 1
    
    # 2. Detect hardware
    console.print("\n[cyan]Step 2: Detecting hardware...[/cyan]")
    HardwareDetector.display_info()
    
    # 3. Create configuration
    console.print("\n[cyan]Step 3: Loading configuration...[/cyan]")
    config = ModelConfig()
    
    # 4. Initialize model
    console.print("\n[cyan]Step 4: Initializing model...[/cyan]")
    loader = initialize_model(config)
    if not loader:
        console.print("[red]Failed to initialize model[/red]")
        return 1
    
    # 5. Run test
    console.print("\n[cyan]Step 5: Testing inference...[/cyan]")
    if not test_inference(loader):
        console.print("[red]Inference test failed[/red]")
        return 1
    
    # 6. Start chat interface
    console.print("\n[cyan]Step 6: Starting chat interface...[/cyan]")
    chat = ChatInterface(loader)
    chat.run()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
