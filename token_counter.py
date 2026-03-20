#!/usr/bin/env python3
"""
Anthropic Token Counting API Integration

Uses the official Anthropic Token Counting API to:
1. Count tokens before sending (cost estimation)
2. Count tokens after receiving (actual cost)
3. Track total usage and costs
"""

import os
import json
from anthropic import Anthropic

# Pricing (2026-03)
PRICING = {
    "claude-haiku-4-5": {"input": 0.80, "output": 2.40},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-opus-4-1": {"input": 15.00, "output": 75.00},
}


def count_tokens(messages: list, model: str = "claude-haiku-4-5") -> int:
    """
    Count tokens in a message using Anthropic Token Counting API.
    
    Args:
        messages: List of message dicts [{"role": "user", "content": "..."}]
        model: Claude model name
    
    Returns:
        Number of input tokens
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not set")
        return 0
    
    client = Anthropic(api_key=api_key)
    
    try:
        response = client.messages.count_tokens(
            model=model,
            messages=messages
        )
        return response.input_tokens
    except Exception as e:
        print(f"⚠️  Token counting failed: {e}")
        return 0


def estimate_cost(messages: list, model: str, output_tokens: int = 100) -> dict:
    """
    Estimate cost before sending a message.
    
    Args:
        messages: List of messages
        model: Claude model
        output_tokens: Estimated output tokens (default 100)
    
    Returns:
        dict with input_tokens, output_tokens, input_cost, output_cost, total_cost
    """
    input_tokens = count_tokens(messages, model)
    
    rates = PRICING.get(model, {"input": 0, "output": 0})
    input_cost = input_tokens * rates["input"] / 1_000_000
    output_cost = output_tokens * rates["output"] / 1_000_000
    total_cost = input_cost + output_cost
    
    return {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": f"${input_cost:.6f}",
        "output_cost": f"${output_cost:.6f}",
        "total_cost": f"${total_cost:.6f}"
    }


def main():
    """Test the token counter."""
    print("🧮 Anthropic Token Counting API Test")
    print("=" * 60)
    
    # Test message
    messages = [
        {
            "role": "user",
            "content": "Analyze this trading setup: BTC at $70,000 with RSI divergence and bullish FVG"
        }
    ]
    
    print("\n📝 Test Message:")
    print(f"  {messages[0]['content']}")
    
    # Count tokens
    print("\n🔍 Counting tokens...")
    estimate = estimate_cost(messages, "claude-haiku-4-5", output_tokens=150)
    
    print("\n💰 Cost Estimate (before sending):")
    print(f"  Model: {estimate['model']}")
    print(f"  Input tokens: {estimate['input_tokens']}")
    print(f"  Output tokens (est): {estimate['output_tokens']}")
    print(f"  Input cost: {estimate['input_cost']}")
    print(f"  Output cost (est): {estimate['output_cost']}")
    print(f"  Total cost (est): {estimate['total_cost']}")
    
    # Test with different models
    print("\n" + "=" * 60)
    print("Testing different models...")
    
    for model in ["claude-haiku-4-5", "claude-sonnet-4-6"]:
        estimate = estimate_cost(messages, model, output_tokens=150)
        print(f"\n{model}:")
        print(f"  Tokens: {estimate['input_tokens']} input")
        print(f"  Est. total cost: {estimate['total_cost']}")


if __name__ == "__main__":
    main()
