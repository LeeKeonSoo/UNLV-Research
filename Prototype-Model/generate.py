"""
Inference script - generate text from a trained model
"""

import torch
import tiktoken
from models import GPTModel
from config import ModelConfig


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get model config from checkpoint
    model_config = checkpoint['config']
    
    # Initialize model
    model = GPTModel(model_config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"Parameters: {model.get_num_params()/1e6:.2f}M")
    
    return model


def generate_text(
    model,
    prompt: str = "",
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 200,
    tokenizer_name: str = 'gpt2',
    device: str = 'cuda'
):
    """
    Generate text from a prompt
    
    Args:
        model: trained GPTModel
        prompt: input text to continue from
        max_new_tokens: number of tokens to generate
        temperature: sampling temperature (higher = more random)
        top_k: keep only top k tokens when sampling
        tokenizer_name: tiktoken tokenizer to use
        device: device to run on
    """
    # Load tokenizer
    enc = tiktoken.get_encoding(tokenizer_name)
    
    # Encode prompt
    if prompt:
        tokens = enc.encode_ordinary(prompt)
        print(f"Prompt: {prompt}")
    else:
        tokens = [enc.eot_token]  # Start with end-of-text token
        print("No prompt provided, starting from scratch")
    
    print(f"Generating {max_new_tokens} tokens...")
    print("-" * 80)
    
    # Convert to tensor
    idx = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    # Generate
    with torch.no_grad():
        generated = model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )
    
    # Decode
    generated_tokens = generated[0].tolist()
    generated_text = enc.decode(generated_tokens)
    
    print(generated_text)
    print("-" * 80)
    
    return generated_text


def interactive_generation(checkpoint_path: str, device: str = 'cuda'):
    """Interactive text generation loop"""
    model = load_model(checkpoint_path, device)
    enc = tiktoken.get_encoding('gpt2')
    
    print("\n" + "=" * 80)
    print("Interactive Text Generation")
    print("Enter a prompt (or 'quit' to exit)")
    print("=" * 80)
    
    while True:
        prompt = input("\nPrompt: ").strip()
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            break
        
        if not prompt:
            print("Please enter a prompt or 'quit' to exit")
            continue
        
        # Get generation parameters
        try:
            max_tokens = int(input("Max tokens (default 100): ") or "100")
            temperature = float(input("Temperature (default 0.8): ") or "0.8")
            top_k = int(input("Top-k (default 200): ") or "200")
        except ValueError:
            print("Invalid input, using defaults")
            max_tokens = 100
            temperature = 0.8
            top_k = 200
        
        # Generate
        generate_text(
            model,
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            device=device
        )


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate text from trained model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, default='', help='Input prompt')
    parser.add_argument('--max-tokens', type=int, default=100, help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=200, help='Top-k sampling')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            print("CUDA not available, using MPS")
            args.device = 'mps'
        else:
            print("CUDA not available, falling back to CPU")
            args.device = 'cpu'
    
    if args.interactive:
        interactive_generation(args.checkpoint, args.device)
    else:
        model = load_model(args.checkpoint, args.device)
        generate_text(
            model,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=args.device
        )
