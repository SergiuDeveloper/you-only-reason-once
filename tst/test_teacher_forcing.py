"""
Tests to verify that teacher forcing and autoregressive modes produce equivalent outputs.
"""

import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from subnet_model import SubnetLLM

def test_batch_equivalence():
    """
    Test that batched sequences produce equivalent outputs when:
    - Autoregressive: processes sequence incrementally with cache
    - Teacher forcing: processes full sequence with position-based routing

    Uses pre-defined sequences to avoid compounding errors from token generation.
    """
    print("\n" + "="*80)
    print("Test: Batch Equivalence with Fixed Sequences")
    print("="*80)

    # Setup
    model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    cache_dir = './cache'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16

    model = SubnetLLM(
        model_name=model_name,
        cache_dir=cache_dir,
        embedding_layers=2,
        coherence_layers=2,
        adaptation_layers=2,
        compensation_layers=2,
        concatenation_layers=1,
        device=device,
        dtype=dtype
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    # Use fixed sequences
    test_texts = [
        "Hello world. This is a test sequence for validation.",
        "Machine learning is fascinating and continues to grow.",
        "The cat sat on the mat and looked around carefully."
    ]

    print(f"Test sequences:")
    for i, text in enumerate(test_texts):
        print(f"  {i+1}. {text}")

    # Tokenize
    inputs = tokenizer(test_texts, return_tensors='pt', padding=True)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    batch_size = input_ids.shape[0]
    seq_length = input_ids.shape[1]
    prompt_length = 5  # First 5 tokens are "prompt", rest are "generated"

    num_generated = seq_length - prompt_length

    print(f"\nBatch size: {batch_size}")
    print(f"Sequence length: {seq_length}")
    print(f"Prompt length: {prompt_length}")
    print(f"Generated length: {num_generated}")

    # Test 1: Autoregressive mode - generate tokens one by one
    print("\n--- Autoregressive Mode ---")

    prompt_ids = input_ids[:, :prompt_length]
    prompt_mask = attention_mask[:, :prompt_length]

    ar_logits_for_generated_positions = []
    ar_predicted_tokens = []

    with torch.no_grad():
        current_ids = prompt_ids
        current_mask = prompt_mask
        cached_reasoning_outputs = [None] * batch_size

        # Generate tokens one by one
        for i in range(num_generated):
            # Forward pass
            logits_ar, cached_reasoning_outputs = model(
                input_ids=current_ids,
                cached_reasoning_outputs=cached_reasoning_outputs,
                attention_mask=current_mask,
                use_teacher_forcing=False
            )

            # Get logits for last position
            last_pos_logits = logits_ar[:, -1, :]
            ar_logits_for_generated_positions.append(last_pos_logits.clone())

            # Get predicted token
            predicted_token = last_pos_logits.argmax(dim=-1, keepdim=True)
            ar_predicted_tokens.append(predicted_token)

            # Use ground truth token to continue (teacher forcing for sequence construction)
            next_token = input_ids[:, prompt_length + i:prompt_length + i + 1]
            current_ids = torch.cat([current_ids, next_token], dim=1)
            current_mask = torch.cat([
                current_mask,
                torch.ones((batch_size, 1), device=device, dtype=current_mask.dtype)
            ], dim=1)

    ar_predicted_tokens = torch.cat(ar_predicted_tokens, dim=1)  # [batch_size, num_generated]
    print(f"Generated {num_generated} tokens")

    # Test 2: Teacher forcing - process full sequence at once
    print("\n--- Teacher Forcing Mode ---")
    with torch.no_grad():
        logits_tf, _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_teacher_forcing=True,
            prompt_length=prompt_length
        )

    print(f"Output shape: {logits_tf.shape}")

    # Extract logits and predictions for generated positions
    # Logits at position i predict token at position i+1
    # So to predict tokens at positions [prompt_length, seq_length),
    # we need logits from positions [prompt_length-1, seq_length-1)
    tf_logits_for_generated_positions = []
    tf_predicted_tokens = []
    for i in range(num_generated):
        logit_pos = prompt_length - 1 + i  # Position whose logits predict the next token
        pos_logits = logits_tf[:, logit_pos, :]
        tf_logits_for_generated_positions.append(pos_logits)
        predicted_token = pos_logits.argmax(dim=-1, keepdim=True)
        tf_predicted_tokens.append(predicted_token)

    tf_predicted_tokens = torch.cat(tf_predicted_tokens, dim=1)  # [batch_size, num_generated]
    print(f"Predicted {num_generated} tokens")

    # Compare the logits
    print("\n--- Comparison ---")

    all_match = True
    num_positions = len(ar_logits_for_generated_positions)

    for idx in range(num_positions):
        pos = prompt_length + idx
        ar_logits = ar_logits_for_generated_positions[idx]
        tf_logits = tf_logits_for_generated_positions[idx]

        max_diff = (ar_logits - tf_logits).abs().max().item()
        mean_diff = (ar_logits - tf_logits).abs().mean().item()

        # Check if predicted tokens match
        ar_tokens = ar_logits.argmax(dim=-1)
        tf_tokens = tf_logits.argmax(dim=-1)
        tokens_match = (ar_tokens == tf_tokens).all().item()

        if idx < 5 or not tokens_match:  # Show first 5 and any mismatches
            print(f"\nPosition {pos}:")
            print(f"  Max diff: {max_diff:.6f}")
            print(f"  Mean diff: {mean_diff:.6f}")
            print(f"  Tokens match: {tokens_match}")
            print(f"  AR predicted tokens: {ar_tokens.tolist()}")
            print(f"  TF predicted tokens: {tf_tokens.tolist()}")

        if not tokens_match:
            all_match = False

    # Print predicted tokens for each sequence
    print(f"\n--- Predicted Token IDs ---")
    for i in range(batch_size):
        print(f"\nSequence {i+1}:")
        print(f"  Ground Truth: {input_ids[i, prompt_length:].tolist()}")
        print(f"  AR Predicted: {ar_predicted_tokens[i].tolist()}")
        print(f"  TF Predicted: {tf_predicted_tokens[i].tolist()}")
        match = (ar_predicted_tokens[i] == tf_predicted_tokens[i]).all().item()
        print(f"  AR == TF: {match}")

    if all_match:
        print(f"\nPASS: All {num_positions} generated positions match")
        return True
    else:
        print(f"\nFAIL: Some positions have mismatched predictions")
        return False

def test_speed_comparison():
    """
    Compare speed of teacher forcing vs autoregressive mode for different sequence lengths.
    """
    print("\n" + "="*80)
    print("Speed Comparison Test")
    print("="*80)

    # Setup
    model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    cache_dir = './cache'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16

    print(f"\nDevice: {device}")
    print("Loading model...")

    model = SubnetLLM(
        model_name=model_name,
        cache_dir=cache_dir,
        embedding_layers=2,
        coherence_layers=2,
        adaptation_layers=2,
        compensation_layers=2,
        concatenation_layers=1,
        device=device,
        dtype=dtype
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    # Test with longer sequences (at least 20 tokens)
    test_texts = [
        "Hello world, this is a test sequence for validation. We need to make sure it has enough tokens for proper testing.",
        "Machine learning is fascinating and continues to evolve. Deep learning models are becoming more sophisticated every day."
    ]
    batch_size = len(test_texts)

    print(f"Batch size: {batch_size}")

    # Tokenize full sequences
    full_inputs = tokenizer(test_texts, return_tensors='pt', padding=True)
    full_ids = full_inputs['input_ids'].to(device)
    full_mask = full_inputs['attention_mask'].to(device)
    full_length = full_ids.shape[1]

    # Use first half as prompt
    prompt_length = full_length // 2
    prompt_ids = full_ids[:, :prompt_length]
    prompt_mask = full_mask[:, :prompt_length]

    print(f"Full sequence length: {full_length} tokens")
    print(f"Prompt length: {prompt_length} tokens")
    print(f"Available for generation: {full_length - prompt_length} tokens")

    # Token counts to test
    token_counts = [1, 10]

    print("\n" + "="*80)
    print(f"{'Tokens':<10} {'AR Time (s)':<15} {'TF Time (s)':<15} {'Speedup':<20}")
    print("="*80)

    for num_tokens in token_counts:
        # Use ground truth tokens from the full sequence
        target_ids = full_ids[:, :prompt_length + num_tokens]
        target_mask = full_mask[:, :prompt_length + num_tokens]

        # Warmup
        with torch.no_grad():
            _ = model(target_ids, attention_mask=target_mask, use_teacher_forcing=True, prompt_length=prompt_length)
            if device == 'cuda':
                torch.cuda.synchronize()

        # Test 1: Autoregressive mode
        if device == 'cuda':
            torch.cuda.synchronize()

        start_time = time.time()

        with torch.no_grad():
            current_ids = prompt_ids
            current_mask = prompt_mask
            cached_reasoning_outputs = [None] * batch_size

            # Generate tokens one at a time
            for i in range(num_tokens):
                _, cached_reasoning_outputs = model(
                    current_ids,
                    cached_reasoning_outputs=cached_reasoning_outputs,
                    attention_mask=current_mask,
                    use_teacher_forcing=False
                )

                # Use ground truth next token
                next_token = full_ids[:, prompt_length + i:prompt_length + i + 1]
                current_ids = torch.cat([current_ids, next_token], dim=1)
                current_mask = torch.cat([
                    current_mask,
                    torch.ones((batch_size, 1), device=device, dtype=current_mask.dtype)
                ], dim=1)

        if device == 'cuda':
            torch.cuda.synchronize()

        ar_time = time.time() - start_time

        # Test 2: Teacher forcing mode
        if device == 'cuda':
            torch.cuda.synchronize()

        start_time = time.time()

        with torch.no_grad():
            _ = model(
                target_ids,
                attention_mask=target_mask,
                use_teacher_forcing=True,
                prompt_length=prompt_length
            )

        if device == 'cuda':
            torch.cuda.synchronize()

        tf_time = time.time() - start_time

        # Calculate speedup
        speedup = ar_time / tf_time
        speedup_percent = (speedup - 1) * 100

        # Format speedup nicely
        if speedup_percent >= 1000:
            speedup_str = f"{speedup:.1f}x  ({speedup_percent:>6,.0f}% faster)"
        else:
            speedup_str = f"{speedup:.2f}x  ({speedup_percent:>6.1f}% faster)"

        print(f"{num_tokens:<10} {ar_time:<15.4f} {tf_time:<15.4f} {speedup_str:<20}")

    print("="*80)
    print("\nNote: Teacher forcing processes all positions in parallel,")
    print("while autoregressive processes them sequentially.")

def test_speed_comparison_vs_original():
    """
    Compare speed of SubnetLLM (autoregressive) vs original TinyLlama model.
    """
    print("\n" + "="*80)
    print("Speed Comparison: SubnetLLM vs Original Model")
    print("="*80)

    # Setup
    model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    cache_dir = './cache'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16

    print(f"\nDevice: {device}")
    print("Loading SubnetLLM...")

    # Load SubnetLLM
    subnet_model = SubnetLLM(
        model_name=model_name,
        cache_dir=cache_dir,
        embedding_layers=2,
        coherence_layers=2,
        adaptation_layers=2,
        compensation_layers=2,
        concatenation_layers=1,
        device=device,
        dtype=dtype
    )
    subnet_model.eval()

    print("Loading original TinyLlama...")

    # Load original model
    original_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        dtype=dtype,
        device_map=device
    )
    original_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    # Test with small batch
    test_texts = [
        "Hello world, this is a test.",
        "Machine learning is fascinating."
    ]
    batch_size = len(test_texts)

    print(f"Batch size: {batch_size}")
    print(f"Prompt: {test_texts[0]}")

    # Token counts to test
    token_counts = [1, 10, 50, 100, 200]

    print("\n" + "="*80)
    print(f"{'Tokens':<10} {'Original (s)':<15} {'SubnetLLM (s)':<15} {'Speedup':<20}")
    print("="*80)

    for num_tokens in token_counts:
        # Prepare input
        inputs = tokenizer(test_texts, return_tensors='pt', padding=True)
        prompt_ids = inputs['input_ids'].to(device)
        prompt_mask = inputs['attention_mask'].to(device)

        # Warmup both models
        with torch.no_grad():
            _ = original_model(prompt_ids, attention_mask=prompt_mask)
            _ = subnet_model(prompt_ids, attention_mask=prompt_mask, use_teacher_forcing=False)
            if device == 'cuda':
                torch.cuda.synchronize()

        # Test 1: Original model - actually generate num_tokens tokens one by one
        if device == 'cuda':
            torch.cuda.synchronize()

        start_time = time.time()

        with torch.no_grad():
            current_ids = prompt_ids
            current_mask = prompt_mask

            # Generate num_tokens tokens, one at a time
            for _ in range(num_tokens):
                # Forward pass on current sequence
                outputs = original_model(
                    current_ids,
                    attention_mask=current_mask
                )

                # Get predicted next token from logits
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

                # Append token
                current_ids = torch.cat([current_ids, next_token], dim=1)
                current_mask = torch.cat([
                    current_mask,
                    torch.ones((batch_size, 1), device=device, dtype=current_mask.dtype)
                ], dim=1)

        if device == 'cuda':
            torch.cuda.synchronize()

        original_time = time.time() - start_time

        # Test 2: SubnetLLM autoregressive mode - actually generate num_tokens tokens one by one
        if device == 'cuda':
            torch.cuda.synchronize()

        start_time = time.time()

        with torch.no_grad():
            current_ids = prompt_ids
            current_mask = prompt_mask
            cached_reasoning_outputs = [None] * batch_size

            # Generate num_tokens tokens, one at a time
            for _ in range(num_tokens):
                # Forward pass on current sequence
                logits, cached_reasoning_outputs = subnet_model(
                    current_ids,
                    cached_reasoning_outputs=cached_reasoning_outputs,
                    attention_mask=current_mask,
                    use_teacher_forcing=False
                )

                # Get predicted next token from logits
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

                # Append token
                current_ids = torch.cat([current_ids, next_token], dim=1)
                current_mask = torch.cat([
                    current_mask,
                    torch.ones((batch_size, 1), device=device, dtype=current_mask.dtype)
                ], dim=1)

        if device == 'cuda':
            torch.cuda.synchronize()

        subnet_time = time.time() - start_time

        # Calculate speedup
        speedup = original_time / subnet_time
        speedup_percent = (speedup - 1) * 100

        # Format speedup nicely
        if speedup_percent >= 1000:
            speedup_str = f"{speedup:.1f}x  ({speedup_percent:>6,.0f}% faster)"
        else:
            speedup_str = f"{speedup:.2f}x  ({speedup_percent:>6.1f}% faster)"

        print(f"{num_tokens:<10} {original_time:<15.4f} {subnet_time:<15.4f} {speedup_str:<20}")

    print("="*80)

def main():
    assert torch.cuda.is_available(), "CUDA is required to run these tests."

    print("\n" + "="*80)
    print("TEACHER FORCING EQUIVALENCE TESTS")
    print("="*80)

    result = test_batch_equivalence()

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    status = "PASS" if result else "FAIL"
    print(f"{status}: Batch Equivalence")

    if result:
        print("\nTest passed!")
    else:
        print("\nTest failed")
        return 1

    # Run speed tests
    test_speed_comparison()
    test_speed_comparison_vs_original()

    return 0

if __name__ == "__main__":
    exit(main())
