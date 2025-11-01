"""
Standalone Ultravox Speech-to-Text Script
Transcribes audio from ul.mp3 file using Ultravox model
"""

import json
import os
import time
import numpy as np
from huggingface_hub import login
from pydub import AudioSegment
from dotenv import load_dotenv

try:
    from transformers import AutoTokenizer
    from vllm import AsyncLLMEngine, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
except ModuleNotFoundError as e:
    print(f"Error: {e}")
    print("Install required packages: pip install -r requirements.txt")
    exit(1)


class UltravoxSTT:
    """Standalone Ultravox Speech-to-Text processor"""
    
    def __init__(self, model_name="fixie-ai/ultravox-v0_6-gemma-3-27b"):
        """Initialize the Ultravox STT processor
        
        Args:
            model_name: Ultravox model to use
        """
        print(f"Initializing Ultravox STT with model: {model_name}")
        
        # Load environment variables from .env file
        load_dotenv()
        
        # Get HF token from environment
        hf_token = os.getenv("HF_TOKEN")
        
        if not hf_token:
            print("ERROR: HF_TOKEN not found in environment variables!")
            print("Please create a .env file with your HuggingFace token:")
            print("HF_TOKEN=your_token_here")
            exit(1)
        
        # Authenticate with Hugging Face
        print("Authenticating with Hugging Face...")
        try:
            login(token=hf_token)
            print("Authentication successful!")
        except Exception as e:
            print(f"Authentication failed: {e}")
            exit(1)
        
        self.model_name = model_name
        self._initialize_engine()
        self._initialize_tokenizer()
        print("Initialization complete!")
    
    def _initialize_engine(self):
        """Initialize the vLLM engine"""
        print("Loading vLLM engine (this may take a few minutes)...")
        engine_args = AsyncEngineArgs(
            model=self.model_name,
            gpu_memory_utilization=0.9,
            max_model_len=8192,
            trust_remote_code=True,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        print("vLLM engine loaded successfully!")
    
    def _initialize_tokenizer(self):
        """Initialize the tokenizer"""
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        print("Tokenizer loaded successfully!")
    
    def load_audio(self, file_path):
        """Load audio file and convert to required format
        
        Args:
            file_path: Path to audio file (mp3, wav, etc.)
            
        Returns:
            np.ndarray: Audio as float32 array normalized to [-1.0, 1.0]
        """
        print(f"\nLoading audio from: {file_path}")
        
        # Load audio file using pydub
        audio = AudioSegment.from_file(file_path)
        
        # Convert to mono if stereo
        if audio.channels > 1:
            print("Converting stereo to mono...")
            audio = audio.set_channels(1)
        
        # Convert to 16kHz sample rate (required by Ultravox)
        if audio.frame_rate != 16000:
            print(f"Resampling from {audio.frame_rate}Hz to 16000Hz...")
            audio = audio.set_frame_rate(16000)
        
        # Get raw audio data as int16
        audio_int16 = np.array(audio.get_array_of_samples(), dtype=np.int16)
        
        # Convert to float32 and normalize to [-1.0, 1.0]
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        
        duration = len(audio_float32) / 16000
        print(f"Audio loaded: {duration:.2f} seconds, {len(audio_float32)} samples")
        
        return audio_float32
    
    async def transcribe(self, audio_data, temperature=0.7, max_tokens=500):
        """Transcribe audio to text
        
        Args:
            audio_data: Audio as float32 numpy array
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            str: Transcribed text
        """
        print("\nStarting transcription...")
        
        # Prepare sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop_token_ids=None
        )
        
        # Format the prompt
        messages = [{"role": "user", "content": "<|audio|>\n"}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Prepare input with audio
        mm_data = {"audio": audio_data}
        inputs = {"prompt": prompt, "multi_modal_data": mm_data}
        
        # Generate transcription
        results_generator = self.engine.generate(
            inputs, sampling_params, str(time.time())
        )
        
        # Collect the full transcription
        full_text = ""
        print("\nTranscription (streaming):")
        print("=" * 60)
        
        async for output in results_generator:
            prompt_output = output.outputs
            new_text = prompt_output[0].text[len(full_text):]
            full_text = prompt_output[0].text
            
            # Print new text as it arrives
            if new_text:
                print(new_text, end="", flush=True)
        
        print("\n" + "=" * 60)
        return full_text


async def main():
    """Main function to run the STT"""
    
    print("=" * 60)
    print("ULTRAVOX SPEECH-TO-TEXT PROCESSOR")
    print("=" * 60)
    
    # Configuration
    audio_file = "ul.mp3"
    model_name = "fixie-ai/ultravox-v0_6-gemma-3-27b"
    
    # Check if audio file exists
    if not os.path.exists(audio_file):
        print(f"\nERROR: Audio file '{audio_file}' not found!")
        print("Please ensure 'ul.mp3' exists in the current directory.")
        return
    
    # Initialize STT
    stt = UltravoxSTT(model_name=model_name)
    
    # Load audio
    audio_data = stt.load_audio(audio_file)
    
    # Transcribe
    transcription = await stt.transcribe(audio_data, temperature=0.0, max_tokens=1000)
    
    # Save transcription to file
    output_file = "transcription.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(transcription)
    
    print(f"\n✓ Transcription saved to: {output_file}")
    print(f"\n{'=' * 60}")
    print("FINAL TRANSCRIPTION:")
    print("=" * 60)
    print(transcription)
    print("=" * 60)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
