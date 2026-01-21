"""
Experimental Decoding Methods for Diffusion Language Models

This module implements two novel decoding strategies:
1. Graph-Aware Historical Remasking: Cascading remask based on token dependencies
2. Margin-Budget: Confusion-based remask with budget control

Author: Research Implementation
Date: 2026-01-17
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List, Optional, Set
from collections import deque

# Import necessary functions from base decoding module
from decoding import (
    add_gumbel_noise,
    margin_function,
    get_num_transfer_tokens
)


# ============================================================================
# Helper Functions for Graph-Aware Method
# ============================================================================

def extract_attention_influence(
    model,
    x: torch.Tensor,
    layer_idx: int = -1
) -> Optional[torch.Tensor]:
    """
    Extract attention weights from model if available.
    
    Args:
        model: The LLaDA model
        x: Input tensor [batch, seq_len]
        layer_idx: Which layer to extract from (-1 for last layer)
    
    Returns:
        Attention weights [batch, seq_len, seq_len] or None if not available
    """
    try:
        # 1. Try standard calling convention
        try:
            outputs = model(x, output_attentions=True)
            if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                attention = outputs.attentions[layer_idx]  # [batch, heads, seq, seq]
                attention = attention.mean(dim=1)  # [batch, seq, seq]
                return attention
        except:
            pass
            
        # 2. Try accessing inner model (if wrapper)
        if hasattr(model, 'model'):
             outputs = model.model(x, output_attentions=True)
             if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                attention = outputs.attentions[layer_idx]
                attention = attention.mean(dim=1)
                return attention
                
    except Exception as e:
        return None
    
    return None


def approximate_attention_uniform(
    x: torch.Tensor,
    generation_history: Dict[int, int]
) -> torch.Tensor:
    """
    Fallback: approximate attention using uniform weights over generated tokens.
    
    Args:
        x: Input tensor [batch, seq_len]
        generation_history: Dict mapping position -> generation_step
    
    Returns:
        Approximate attention weights [batch, seq_len, seq_len]
    """
    batch_size, seq_len = x.shape
    attention = torch.zeros(batch_size, seq_len, seq_len, device=x.device)
    
    # For each position, give uniform attention to all previously generated positions
    for i in range(seq_len):
        if i in generation_history:
            # Find all positions generated before this one
            earlier_positions = [j for j in range(i) if j in generation_history 
                               and generation_history[j] < generation_history[i]]
            if earlier_positions:
                # Uniform attention over earlier positions
                attention[:, i, earlier_positions] = 1.0 / len(earlier_positions)
    
    return attention


def build_dependency_graph(
    attention_weights: torch.Tensor,
    generation_history: Dict[int, int],
    threshold: float = 0.1
) -> Dict[int, Set[int]]:
    """
    Build dependency graph from attention weights.
    
    Args:
        attention_weights: [batch, seq_len, seq_len]
        generation_history: Dict mapping position -> generation_step
        threshold: Minimum attention weight to consider as dependency
    
    Returns:
        Graph as adjacency list: position -> set of related positions
    """
    batch_size, seq_len, _ = attention_weights.shape
    graph = {i: set() for i in generation_history.keys()}
    
    # Build edges based on attention weights
    for i in generation_history.keys():
        for j in generation_history.keys():
            if i != j and attention_weights[0, i, j] > threshold:
                graph[i].add(j)
    
    return graph


def find_related_tokens_bfs(
    token_pos: int,
    graph: Dict[int, Set[int]],
    generation_history: Dict[int, int],
    max_depth: int = 2
) -> List[Tuple[int, int]]:
    """
    Find related tokens using BFS with depth limit.
    
    Args:
        token_pos: Starting token position
        graph: Dependency graph
        generation_history: Generation step for each position
        max_depth: Maximum search depth
    
    Returns:
        List of (position, depth) tuples
    """
    if token_pos not in graph:
        return []
    
    visited = {token_pos}
    queue = deque([(token_pos, 0)])
    related = []
    
    while queue:
        pos, depth = queue.popleft()
        
        if depth >= max_depth:
            continue
        
        # Only consider tokens generated before current token
        current_step = generation_history[token_pos]
        
        for neighbor in graph[pos]:
            if neighbor not in visited and neighbor in generation_history:
                neighbor_step = generation_history[neighbor]
                if neighbor_step < current_step:
                    visited.add(neighbor)
                    related.append((neighbor, depth + 1))
                    queue.append((neighbor, depth + 1))
    
    return related


def compute_temporal_proximity(
    token_pos: int,
    related_pos: int,
    generation_history: Dict[int, int],
    decay: float = 0.5
) -> float:
    """
    Compute temporal proximity weight.
    
    Args:
        token_pos: Current token position
        related_pos: Related token position
        generation_history: Generation step for each position
        decay: Decay rate
    
    Returns:
        Temporal proximity score [0, 1]
    """
    step_diff = generation_history[token_pos] - generation_history[related_pos]
    return np.exp(-decay * step_diff)


# ============================================================================
# Helper Functions for Graph-Aware V2
# ============================================================================

def is_numeric_or_operator(token_id: int, tokenizer=None) -> bool:
    """
    Check if token is numeric or mathematical operator (GSM8K specific).
    
    Args:
        token_id: Token ID to check
        tokenizer: Optional tokenizer for decoding
    
    Returns:
        True if token is numeric/operator
    """
    # Common numeric/operator token IDs (approximate, may need adjustment)
    # This is a heuristic - ideally use tokenizer.decode()
    if tokenizer is not None:
        try:
            token_str = tokenizer.decode([token_id]).strip()
            # Check if it's a number or operator
            if token_str.isdigit() or token_str in ['+', '-', '*', '/', '=', '(', ')', '.', ',']:
                return True
            # Check for multi-digit numbers
            if token_str.replace(',', '').replace('.', '').isdigit():
                return True
        except:
            pass
    
    return False


def compute_confidence_modulated_priority(
    relation_score: float,
    confidence: float,
    confidence_threshold: float = 0.6
) -> float:
    """
    Compute remask priority with confidence modulation.
    
    Args:
        relation_score: Graph-based relation score
        confidence: Token confidence
        confidence_threshold: Threshold for high confidence
    
    Returns:
        Priority score (higher = more likely to remask)
    """
    if confidence >= confidence_threshold:
        # High confidence -> no remask
        return 0.0
    else:
        # Lower confidence -> higher priority
        # g(c) = (1 - c) for c < threshold
        return relation_score * (1.0 - confidence)


def get_adaptive_cascade_depth(
    current_step: int,
    estimated_total_steps: int,
    max_depth: int = 2,
    early_ratio: float = 0.3,
    mid_ratio: float = 0.5
) -> int:
    """
    Get cascade depth based on generation progress (Time-Adaptive).
    
    Args:
        current_step: Current generation step
        estimated_total_steps: Estimated total steps
        max_depth: Maximum cascade depth
        early_ratio: Progress ratio for early phase
        mid_ratio: Progress ratio for mid phase
    
    Returns:
        Cascade depth for current step
    """
    progress = current_step / max(estimated_total_steps, 1)
    
    if progress < early_ratio:
        return 0  # No cascading in early phase
    elif progress < mid_ratio:
        return 1  # Shallow cascading in mid phase
    else:
        return max_depth  # Full cascading in late phase



# ============================================================================
# Helper Functions for Margin-Budget Method
# ============================================================================

def compute_top2_margin(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute margin between top-1 and top-2 probabilities.
    
    Args:
        logits: [batch, seq_len, vocab_size]
    
    Returns:
        margin: [batch, seq_len]
        top1_indices: [batch, seq_len]
        top1_probs: [batch, seq_len]
    """
    probs = F.softmax(logits, dim=-1)
    top2_probs, top2_indices = torch.topk(probs, k=2, dim=-1)
    
    top1_probs = top2_probs[:, :, 0]
    top2_probs_second = top2_probs[:, :, 1]
    top1_indices = top2_indices[:, :, 0]
    
    margin = top1_probs - top2_probs_second
    
    return margin, top1_indices, top1_probs


def annealed_budget_schedule(
    step: int,
    initial_budget: int = 5,
    decay_factor: float = 0.95
) -> int:
    """
    Compute budget using exponential decay (Annealing).
    
    B_t = B_0 * (decay_factor ^ step)
    
    This guarantees convergence as, eventually, B_t < 1, stopping all remasking.
    """
    decayed_budget = initial_budget * (decay_factor ** step)
    return int(decayed_budget)


# ============================================================================
# Main Decoding Functions
# ============================================================================

@torch.no_grad()
def decoding_graph_aware(
    model,
    prompt: torch.Tensor,
    gen_length: int = 256,
    block_length: int = 256,
    temperature: float = 0.0,
    mask_id: int = 126336,
    attention_threshold: float = 0.15,
    cascade_depth: int = 2,
    temporal_decay: float = 0.5,
    confidence_threshold: float = 0.6,
    adaptive: bool = True
) -> Tuple[torch.Tensor, int]:
    """
    Original Graph-Aware Cascading Remasking (Attention-based/Uniform).
    """
    return _decoding_graph_aware_impl(
        model, prompt, gen_length, block_length, temperature, mask_id,
        confidence_threshold, attention_threshold, cascade_depth, temporal_decay, adaptive,
        use_gradient=False
    )

def compute_token_importance_by_gradient(
    model,
    x: torch.Tensor,
    generation_history: Dict[int, int],
    mask_id: int
) -> torch.Tensor:
    """
    Gradient-based Token Importance Estimation.
    
    Computes the importance of each token by measuring the gradient of the loss
    with respect to the token embeddings. High gradient norm indicates that the
    token is critical for the current model prediction quality.
    
    Returns:
        importance: [batch, seq_len, seq_len] tensor
    """
    batch_size, seq_len = x.shape
    device = x.device
    
    # Enable gradient calculation for embeddings
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        embedding_layer = model.model.embed_tokens
    elif hasattr(model, "get_input_embeddings"):
        embedding_layer = model.get_input_embeddings()
    else:
        return approximate_attention_uniform(x, generation_history)
        
    try:
        # Create a fresh graph for gradient calculation
        with torch.enable_grad():
            # We need to manually perform embedding lookup to track gradients
            # Note: We must ensure x inputs are long tensors
            inputs_embeds = embedding_layer(x)
            inputs_embeds.retain_grad()
            
            # Forward pass
            outputs = model(inputs_embeds=inputs_embeds)
            logits = outputs.logits
            
            # Calculate pseudo-loss (Self-Correction Loss)
            # We define "Importance" as: How much does changing embedding[j] affect Total LogProb of valid tokens?
            
            # Log probs: [batch, seq_len, vocab_size]
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Gather log probs for the current tokens in x
            # Target is x itself
            selected_log_probs = torch.gather(log_probs, -1, x.unsqueeze(-1)).squeeze(-1)
            
            # Mask out special tokens (mask_id) from the loss calculation
            # We only care about the confidence of generated tokens
            valid_mask = (x != mask_id).float()
            
            # Simple sum of log probs for valid tokens
            total_log_prob = (selected_log_probs * valid_mask).sum()
            
            # Backward pass
            # We want to maximize log prob, so loss is negative sum?
            # Actually we just want gradients magnitude, so direction doesn't matter much
            # But normally we do backward on scalar.
            total_log_prob.backward()
            
            # Get gradients: [batch, seq_len, hidden_dim]
            grads = inputs_embeds.grad
            
            if grads is None:
                return approximate_attention_uniform(x, generation_history)
                
            # Gradient Norm: [batch, seq_len]
            token_importance = grads.norm(dim=-1)
            
            # Construct Importance Matrix [batch, seq_len, seq_len]
            # importance[b, i, j] = importance of token j to token i
            # In this global approximation (one backward pass), we get global importance of j
            # So importance[b, :, j] = token_importance[b, j]
            
            importance_matrix = torch.zeros(batch_size, seq_len, seq_len, device=device)
            importance_matrix = token_importance.unsqueeze(1).expand(-1, seq_len, -1)
            
            # Normalize to [0, 1] range per row
            max_val = importance_matrix.max(dim=-1, keepdim=True)[0]
            importance_matrix = importance_matrix / (max_val + 1e-6)
            
            return importance_matrix.detach()
            
    except Exception as e:
        print(f"Gradient computation failed: {e}")
        return approximate_attention_uniform(x, generation_history)

@torch.no_grad()
def decoding_graph_aware_gradient(
    model, 
    prompt: torch.Tensor, 
    gen_length: int = 256, 
    block_length: int = 256,
    temperature: float = 0.0,
    mask_id: int = 126336,
    confidence_threshold: float = 0.6,
    attention_threshold: float = 0.15,
    cascade_depth: int = 2,
    temporal_decay: float = 0.5,
    adaptive: bool = True
) -> Tuple[torch.Tensor, int]:
    """
    Graph-Aware Cascading Remasking with Gradient-based Importance (Loss-based).
    Use gradient of loss w.r.t input embeddings to determine token importance.
    """
    return _decoding_graph_aware_impl(
        model, prompt, gen_length, block_length, temperature, mask_id,
        confidence_threshold, attention_threshold, cascade_depth, temporal_decay, adaptive,
        use_gradient=True
    )

def _decoding_graph_aware_impl(
    model, 
    prompt: torch.Tensor, 
    gen_length: int, 
    block_length: int,
    temperature: float,
    mask_id: int,
    confidence_threshold: float,
    attention_threshold: float,
    cascade_depth: int,
    temporal_decay: float,
    adaptive: bool,
    use_gradient: bool
) -> Tuple[torch.Tensor, int]:
    
    device = model.device
    batch_size = prompt.shape[0]
    
    # Initialize masked sequence
    x = torch.full(
        (batch_size, prompt.shape[1] + gen_length),
        mask_id,
        dtype=torch.long,
        device=device
    )
    x[:, :prompt.shape[1]] = prompt.clone()
    
    generation_history = {} # pos -> step
    current_step = 0
    
    # Block-wise generation
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    
    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = block_start + block_length
        
        # Adaptive thresholds per block if enabled
        if adaptive:
            progress = num_block / num_blocks
            current_threshold = attention_threshold * (1 - progress * 0.5)
            current_depth = min(cascade_depth + int(progress * 2), cascade_depth + 2)
        else:
            current_threshold = attention_threshold
            current_depth = cascade_depth
        
        # Generate block
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        
        while block_mask_index.any():
            current_step += 1
            
            # Forward pass
            with torch.no_grad():
                logits = model(x).logits
            
            # Add Gumbel noise and sample
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            
            # Compute confidence
            p = F.softmax(logits.to(torch.float64), dim=-1)
            x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            
            # Mask index for current block
            mask_index = (x == mask_id)
            mask_index[:, block_end:] = False
            
            # Confidence for masked positions
            confidence = torch.where(mask_index, x0_p, -np.inf)
            
            # Select tokens to unmask (high confidence)
            high_conf_mask = (confidence > confidence_threshold) & mask_index
            
            if high_conf_mask.sum() == 0:
                # Force at least one token
                max_conf_pos = confidence.argmax(dim=1)
                for b in range(batch_size):
                    high_conf_mask[b, max_conf_pos[b]] = True
            
            # Unmask high confidence tokens
            x[high_conf_mask] = x0[high_conf_mask]
            
            # Update generation history
            for b in range(batch_size):
                positions = torch.where(high_conf_mask[b])[0]
                for pos in positions:
                    generation_history[pos.item()] = current_step
            
            # Graph-Aware Cascading Remask
            # Find tokens to potentially remask based on dependencies
            if len(generation_history) > 1:
                # Determine Token Importance / Dependency
                if use_gradient:
                    attention = compute_token_importance_by_gradient(model, x, generation_history, mask_id)
                else:
                    attention = extract_attention_influence(model, x)
                    if attention is None:
                        attention = approximate_attention_uniform(x, generation_history)
                
                # Build dependency graph
                graph = build_dependency_graph(
                    attention, generation_history, current_threshold
                )
                
                # For each newly unmasked token, check if we should remask related tokens
                remask_candidates = set()
                
                for b in range(batch_size):
                    positions = torch.where(high_conf_mask[b])[0]
                    for pos in positions:
                        pos_item = pos.item()
                        if pos_item in graph:
                            # Find related tokens
                            related = find_related_tokens_bfs(
                                pos_item, graph, generation_history, current_depth
                            )
                            
                            # Check if related tokens should be remasked
                            for rel_pos, depth in related:
                                # Compute relation score
                                temporal_prox = compute_temporal_proximity(
                                    pos_item, rel_pos, generation_history, temporal_decay
                                )
                                if attention.dim() == 3:
                                    attn_score = attention[b, pos_item, rel_pos].item()
                                else:
                                    # Fallback if dimension mismatch
                                    attn_score = 1.0 / len(generation_history)
                                
                                # Use temporal proximity and importance score combined
                                # Weighted combination: 70% importance, 30% temporal
                                relation_score = 0.7 * attn_score + 0.3 * temporal_prox
                                
                                # If relation is strong and confidence is low, remask
                                if rel_pos < x.shape[1] and \
                                   (relation_score > current_threshold and 
                                    x0_p[b, rel_pos] < confidence_threshold):
                                    remask_candidates.add((b, rel_pos))
                
                # Apply cascading remask
                for b, pos in remask_candidates:
                    if x[b, pos] != mask_id:  # Only remask if currently unmasked
                        x[b, pos] = mask_id
                        if pos in generation_history:
                            del generation_history[pos]
            
            # Update block mask index
            block_mask_index = (x[:, block_start:block_end] == mask_id)
    
    return x, current_step

@torch.no_grad()
def decoding_margin_budget(
    model,
    prompt: torch.Tensor,
    gen_length: int = 256,
    block_length: int = 256,
    temperature: float = 0.0,
    mask_id: int = 126336,
    margin_threshold: float = 0.1,
    budget_per_step: int = 5,
    adaptive_budget: bool = True
) -> Tuple[torch.Tensor, int]:
    """
    Margin-Budget decoding with Annealed Budget Strategy.
    """
    device = model.device
    batch_size = prompt.shape[0]
    
    # Initialize masked sequence
    x = torch.full(
        (batch_size, prompt.shape[1] + gen_length),
        mask_id,
        dtype=torch.long,
        device=device
    )
    x[:, :prompt.shape[1]] = prompt.clone()
    
    # Track remask counts (Safety guarantee, though annealing ensures convergence)
    remask_counts = torch.zeros_like(x, dtype=torch.long)
    MAX_REMASK_PER_TOKEN = 2  # Slightly relaxed as budget controls total remasks
    
    prompt_index = (x != mask_id)
    current_step = 0
    decay_factor = 0.99  # Slow decay
    
    # Block-wise generation
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    
    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = block_start + block_length
        
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        
        while block_mask_index.any():
            current_step += 1
            
            # Determine budget using Annealed Schedule
            if adaptive_budget:
                current_budget = annealed_budget_schedule(
                    current_step, budget_per_step, decay_factor
                )
            else:
                current_budget = budget_per_step
            
            # Forward pass
            logits = model(x).logits
            
            # Compute margin
            margin, top1_indices, top1_probs = compute_top2_margin(logits)
            
            # Dynamic Thresholding (Mean - 0.5 * Std)
            valid_mask = (x != mask_id) & ~prompt_index
            valid_mask[:, block_end:] = False
            
            if valid_mask.any():
                valid_margins = margin[valid_mask]
                if valid_margins.numel() > 1:
                    current_threshold = valid_margins.mean() - 0.5 * valid_margins.std()
                else:
                    current_threshold = valid_margins.mean()
                current_threshold = torch.clamp(current_threshold, 0.01, 0.5)
            else:
                current_threshold = margin_threshold

            # Add Gumbel noise for sampling
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            
            # Mask index for current block
            mask_index = (x == mask_id)
            mask_index[:, block_end:] = False
            
            # Unmask tokens with high margin
            high_margin_mask = (margin > current_threshold) & mask_index
            x[high_margin_mask] = top1_indices[high_margin_mask]
            
            # Find confused tokens
            unmasked_index = (x != mask_id) & ~prompt_index
            unmasked_index[:, block_end:] = False
            
            can_remask = (remask_counts < MAX_REMASK_PER_TOKEN)
            valid_remask_mask = unmasked_index & can_remask
            
            confusion_score = torch.where(valid_remask_mask, margin, float('inf'))
            
            # Budget-controlled remasking
            # Only remask if budget allows (>0) and we made progress
            if high_margin_mask.any() and current_budget > 0 and valid_remask_mask.sum() > 0:
                for b in range(batch_size):
                    valid_candidates = (confusion_score[b] < float('inf')).nonzero(as_tuple=True)[0]
                    num_valid = len(valid_candidates)
                    
                    if num_valid > 0:
                        k = min(current_budget, num_valid)
                        _, top_indices = torch.topk(confusion_score[b], k=k, largest=False)
                        x[b, top_indices] = mask_id
                        remask_counts[b, top_indices] += 1
            
            # Force progress if no tokens were unmasked
            if not high_margin_mask.any():
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                confidence = torch.where(mask_index, x0_p, -np.inf)
                
                for b in range(batch_size):
                    if mask_index[b].any():
                        max_conf_pos = confidence[b].argmax()
                        x[b, max_conf_pos] = x0[b, max_conf_pos]
            
            # Update block mask index
            block_mask_index = (x[:, block_start:block_end] == mask_id)
    
    return x, current_step


@torch.no_grad()
def decoding_graph_aware_v2(
    model,
    prompt: torch.Tensor,
    gen_length: int = 256,
    block_length: int = 256,
    temperature: float = 0.0,
    mask_id: int = 126336,
    # V2-specific parameters
    early_commit_ratio: float = 0.25,  # First 25% steps are anchors
    cascade_start_ratio: float = 0.3,  # Start cascading at 30% progress
    cascade_full_ratio: float = 0.5,   # Full cascading at 50% progress
    confidence_high: float = 0.7,      # High confidence threshold (protection)
    confidence_low: float = 0.4,       # Low confidence threshold
    attention_threshold: float = 0.15,
    temporal_decay_strong: float = 1.0,  # Early-step temporal decay
    temporal_decay_base: float = 0.5,    # Base temporal decay
    remask_budget: int = 8,            # Max remasks per step
    cooldown_period: int = 3,          # Steps before token can be remasked
    protect_symbols: bool = True,      # GSM8K-specific symbol protection
    tokenizer = None                   # For symbol detection
) -> Tuple[torch.Tensor, int]:
    """
    Graph-Aware V2: Structurally redesigned remasking strategy.
    
    Implements three protective layers:
    - Layer A: Anchor Protection (prompt/early-commit/symbol locks)
    - Layer B: Time-Adaptive Cascading (step-gated depth)
    - Layer C: Soft Priority with Bounded Budget
    
    Args:
        model: LLaDA model
        prompt: Input prompt tensor
        gen_length: Generation length
        block_length: Block size
        temperature: Sampling temperature
        mask_id: Mask token ID
        early_commit_ratio: Ratio of total steps for early-commit anchors
        cascade_start_ratio: When to start cascading
        cascade_full_ratio: When to enable full cascading
        confidence_high: High confidence protection threshold
        confidence_low: Low confidence threshold for remask eligibility
        attention_threshold: Attention weight threshold
        temporal_decay_strong: Strong temporal decay for early steps
        temporal_decay_base: Base temporal decay
        remask_budget: Maximum remasks per step
        cooldown_period: Steps before remasking same token
        protect_symbols: Enable symbol/numeric protection (GSM8K)
        tokenizer: Tokenizer for symbol detection
    
    Returns:
        (generated_sequence, num_steps)
    """
    device = model.device
    batch_size = prompt.shape[0]
    
    # Initialize masked sequence
    x = torch.full(
        (batch_size, prompt.shape[1] + gen_length),
        mask_id,
        dtype=torch.long,
        device=device
    )
    x[:, :prompt.shape[1]] = prompt.clone()
    
    # V2 State Tracking
    generation_history = {}  # pos -> step when unmasked
    committed_step = {}      # pos -> step when first unmasked
    cooldown_tracker = {}    # pos -> remaining cooldown steps
    remask_count = {}        # pos -> total times remasked
    current_step = 0
    
    # Estimate total steps for time-adaptive logic
    estimated_total_steps = gen_length  # Rough estimate
    
    # Block-wise generation
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    
    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = block_start + block_length
        
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        
        while block_mask_index.any():
            current_step += 1
            
            # Update cooldown
            for pos in list(cooldown_tracker.keys()):
                cooldown_tracker[pos] -= 1
                if cooldown_tracker[pos] <= 0:
                    del cooldown_tracker[pos]
            
            # Forward pass
            with torch.no_grad():
                logits = model(x).logits
            
            # Add Gumbel noise
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            
            # Compute confidence
            p = F.softmax(logits.to(torch.float64), dim=-1)
            x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            
            # Mask index for current block
            mask_index = (x == mask_id)
            mask_index[:, block_end:] = False
            
            # Confidence for masked positions
            confidence = torch.where(mask_index, x0_p, -np.inf)
            
            # ======================================================================
            # UNMASK: Select tokens to unmask (high confidence)
            # ======================================================================
            high_conf_mask = (confidence > confidence_high) & mask_index
            
            if high_conf_mask.sum() == 0:
                # Force at least one token
                max_conf_pos = confidence.argmax(dim=1)
                for b in range(batch_size):
                    if mask_index[b].any():
                        high_conf_mask[b, max_conf_pos[b]] = True
            
            # Unmask high confidence tokens
            x[high_conf_mask] = x0[high_conf_mask]
            
            # Update generation/commit history
            newly_unmasked_positions = []
            for b in range(batch_size):
                positions = torch.where(high_conf_mask[b])[0]
                for pos in positions:
                    pos_item = pos.item()
                    generation_history[pos_item] = current_step
                    
                    # Track first commit
                    if pos_item not in committed_step:
                        committed_step[pos_item] = current_step
                    
                    newly_unmasked_positions.append(pos_item)
            
            # ======================================================================
            # V2 REMASKING: Three-Layer Architecture
            # ======================================================================
            
            if len(generation_history) > 1 and len(newly_unmasked_positions) > 0:
                
                # ------------------------------------------------------------------
                # Layer A: Define Anchor Protection (Inviolable Zones)
                # ------------------------------------------------------------------
                anchors = set()
                
                # A1: Prompt Lock
                for pos in range(prompt.shape[1]):
                    anchors.add(pos)
                
                # A2: Early-Commit Lock
                early_commit_threshold = int(early_commit_ratio * estimated_total_steps)
                for pos, commit_step in committed_step.items():
                    if commit_step <= early_commit_threshold:
                        anchors.add(pos)
                
                # A3: Symbol/Numeric Lock (if enabled)
                if protect_symbols and tokenizer is not None:
                    for pos in generation_history.keys():
                        if is_numeric_or_operator(x[0, pos].item(), tokenizer):
                            anchors.add(pos)
                
                # Remask eligibility
                remask_eligible = set()
                for pos in generation_history.keys():
                    if pos not in anchors and pos not in cooldown_tracker:
                        remask_eligible.add(pos)
                
                if len(remask_eligible) == 0:
                    # No candidates, skip remasking
                    block_mask_index = (x[:, block_start:block_end] == mask_id)
                    continue
                
                # ------------------------------------------------------------------
                # Layer B: Time-Adaptive Cascading Depth & Temporal Decay
                # ------------------------------------------------------------------
                cascade_depth = get_adaptive_cascade_depth(
                    current_step,
                    estimated_total_steps,
                    max_depth=2,
                    early_ratio=cascade_start_ratio,
                    mid_ratio=cascade_full_ratio
                )
                
                # Time-adaptive temporal decay
                progress = current_step / max(estimated_total_steps, 1)
                if progress < cascade_start_ratio:
                    current_temporal_decay = temporal_decay_strong
                else:
                    current_temporal_decay = temporal_decay_base
                
                # ------------------------------------------------------------------
                # Get Attention/Importance
                # ------------------------------------------------------------------
                attention = extract_attention_influence(model, x)
                if attention is None:
                    attention = approximate_attention_uniform(x, generation_history)
                
                # Build dependency graph
                graph = build_dependency_graph(
                    attention, generation_history, attention_threshold
                )
                
                # ------------------------------------------------------------------
                # Layer C: Soft Priority Scoring with Bounded Budget
                # ------------------------------------------------------------------
                priority_scores = {}  # pos -> priority score
                
                for b in range(batch_size):
                    for new_pos in newly_unmasked_positions:
                        if new_pos in graph and new_pos < x.shape[1]:
                            # Find related tokens via BFS
                            related = find_related_tokens_bfs(
                                new_pos, graph, generation_history, cascade_depth
                            )
                            
                            for rel_pos, depth in related:
                                if rel_pos not in remask_eligible:
                                    continue
                                
                                # Compute relation score
                                temporal_prox = compute_temporal_proximity(
                                    new_pos, rel_pos, generation_history, current_temporal_decay
                                )
                                
                                if attention.dim() == 3 and b < attention.shape[0]:
                                    attn_score = attention[b, new_pos, rel_pos].item()
                                else:
                                    attn_score = 1.0 / max(len(generation_history), 1)
                                
                                # Relation score: attention + temporal
                                R = 0.7 * attn_score + 0.3 * temporal_prox
                                
                                # Confidence modulation: g(c)
                                token_confidence = x0_p[b, rel_pos].item()
                                priority = compute_confidence_modulated_priority(
                                    R, token_confidence, confidence_high
                                )
                                
                                # Track highest priority for each position
                                if rel_pos not in priority_scores or priority > priority_scores[rel_pos]:
                                    priority_scores[rel_pos] = priority
                
                # ------------------------------------------------------------------
                # Bounded Budget Selection (Top-K)
                # ------------------------------------------------------------------
                if len(priority_scores) > 0:
                    # Sort by priority (descending)
                    sorted_candidates = sorted(
                        priority_scores.items(),
                        key=lambda item: item[1],
                        reverse=True
                    )
                    
                    # Select top-k up to budget
                    remask_set = []
                    for pos, score in sorted_candidates[:remask_budget]:
                        if score > 0:  # Only remask if priority > 0
                            remask_set.append(pos)
                    
                    # Apply remasks
                    for pos in remask_set:
                        if x[0, pos] != mask_id:  # Only remask if currently unmasked
                            x[:, pos] = mask_id
                            
                            # Update tracking
                            if pos in generation_history:
                                del generation_history[pos]
                            cooldown_tracker[pos] = cooldown_period
                            remask_count[pos] = remask_count.get(pos, 0) + 1
            
            # Update block mask index
            block_mask_index = (x[:, block_start:block_end] == mask_id)
    
    return x, current_step

# ==========================================
# Score-Grounded Graph-Aware (SG-GA) Helpers
# ==========================================

def compute_hidden_representation(model, x):
    """
    Compute hidden states as proxy for score vectors.
    Stable and fast alternative to autograd.grad.
    """
    # Forward pass with hidden states
    outputs = model(x, output_hidden_states=True)
    
    # Use the last layer's hidden state (normalized)
    hidden = outputs.hidden_states[-1] # [B, L, H]
    
    # Normalize for cosine similarity usage later
    hidden = F.normalize(hidden, p=2, dim=-1)
    
    return hidden, outputs.logits

def compute_entropy(logits):
    """Compute entropy H(x) for each position."""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy

def compute_curvature(current_hidden, prev_hidden):
    """
    Compute path curvature based on change in hidden states.
    """
    if prev_hidden is None:
        return 0.0
    
    # Cosine distance = 1 - Cosine Similarity
    # Since hidden are normalized, ||a-b||^2 = 2(1-cos)
    # Simple Euclidean distance on sphere
    delta = current_hidden - prev_hidden
    delta_norm = torch.norm(delta, dim=-1).mean()
    
    return delta_norm.item()

def compute_entropy_budget(logits, alpha=0.5, min_b=1, max_b=15, curvature=0.0, step_decay=1.0):
    """
    Compute dynamic budget based on Entropy (Uncertainty) and Curvature.
    Replaces Fisher Info with Entropy proxy.
    """
    # Mean entropy of the sequence
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean().item()
    
    # Base budget from Entropy
    base_budget = alpha * entropy * 10 # Scale factor
    
    # Modulate
    modulated_budget = base_budget * (1.0 + curvature * 5.0)
    
    # Decay
    final_budget = int(modulated_budget * step_decay)
    
    return max(min_b, min(final_budget, max_b))

@torch.no_grad()
def decoding_graph_aware_sg_ga(
    model,
    prompt: torch.Tensor,
    gen_length: int = 256,
    block_length: int = 256,
    temperature: float = 0.0,
    mask_id: int = 126336,
    
    # SG-GA Theoretical Parameters
    alpha: float = 0.5,        # Fisher scaling
    tau_low: float = 0.05,     # Curvature threshold low
    tau_high: float = 0.15,    # Curvature threshold high
    
    # Fallbacks/Bounds
    min_budget: int = 3,
    max_budget: int = 15,
    graph_threshold: float = 0.2,  # Cosine similarity threshold (Optimized 0.1 -> 0.2)
    confidence_threshold: float = 0.5, # MATCH WINO BASELINE (0.5) for fair comparison
    trace_log_path: str = None,
    sample_id: int = None,
    shard_id: int = None,
    **kwargs
) -> Tuple[torch.Tensor, int]:
    """
    Score-Grounded Graph-Aware (SG-GA) Decoding.
    """
    # DEBUG parameter injection
    if sample_id is not None and sample_id == 0:
        print(f"[SG-GA-DEBUG] Started. TracePath={trace_log_path}, ConfThresh={confidence_threshold} (Matched WINO)")
    # Delegate immediately to implementation to avoid duplicate code name error
    return _decoding_sg_ga_impl(
        model, prompt, gen_length, block_length, temperature, mask_id,
        alpha, tau_low, tau_high, min_budget, max_budget, graph_threshold, confidence_threshold
    )
    
    print(f"[SG-GA] Function called. Gen Length: {gen_length}", flush=True)
    device = model.device
    batch_size = prompt.shape[0]
    
    print(f"[SG-GA] Allocation tensor on {device}...", flush=True)
    x = torch.full((batch_size, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt.shape[1]] = prompt.clone()
    print(f"[SG-GA] Tensor allocated.", flush=True)
    
    generation_history = {}
    current_step = 0
    prev_score_vectors = None
    
    # Block-wise generation setup
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    print(f"[SG-GA] Starting block loop. Num blocks: {num_blocks}", flush=True)

    for num_block in range(num_blocks):
        print(f"[SG-GA] Processing Block {num_block}", flush=True)
        block_start = prompt.shape[1] + num_block * block_length
        block_end = block_start + block_length
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        
        while block_mask_index.any():
            current_step += 1
            
            # 1. Forward Pass & Unmasking (Standard)
            with torch.enable_grad(): # Enable grad only for score computation logic if integrated, but here we separate
                # Standard inference (no grad needed)
                logits = model(x).logits
            
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            
            # Unmasking logic (similar to standard)
            p = F.softmax(logits.to(torch.float64), dim=-1)
            x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            
            mask_index = (x == mask_id)
            mask_index[:, block_end:] = False # Only unmask current block
            
            # Simple confidence-based unmasking (can be replaced by score-based unmasking later)
            # Keeping WINO-like unmasking for fairness/baseline
            confidence = torch.where(mask_index, x0_p, -np.inf)
            # Dynamic threshold could be applied here too, but using heuristic 0.5 for unmasking to match others
            high_conf_mask = (confidence > 0.5) & mask_index
            
            if high_conf_mask.sum() == 0:
                max_conf_pos = confidence.argmax(dim=1)
                for b in range(batch_size):
                    high_conf_mask[b, max_conf_pos[b]] = True
            
            # Record newly unmasked
            newly_unmasked = []
            for b in range(batch_size):
                newly_unmasked.append(torch.where(high_conf_mask[b])[0].tolist())
            
            x[high_conf_mask] = x0[high_conf_mask]
            
            # Update history
            for b in range(batch_size):
                for pos in newly_unmasked[b]:
                    generation_history[pos] = current_step

            # ==========================================
            # SG-GA Step: Theoretical Refinement
            # ==========================================
            
            # 2. Compute Theoretical Metrics (Hidden State based)
            # Replaced Autograd with Hidden States to prevent Deadlock/Hang
            with torch.no_grad():
                # Get hidden states (Feature Representation)
                hidden_states, logits_curr = compute_hidden_representation(model, x)
            
            # 3. Dynamic Cascade Depth from Curvature
            curvature = compute_curvature(hidden_states, prev_score_vectors)
            prev_score_vectors = hidden_states.detach() # naming kept for compat, but holds hidden
            
            # Curvature-based cascade depth
            current_cascade_depth = 0
            if curvature >= tau_high:
                current_cascade_depth = 2
            elif curvature >= tau_low:
                current_cascade_depth = 1
            
            # 4. Dynamic Budget from Entropy/Curvature
            step_decay = max(0.0, 1.0 - (loop_step * 0.2)) 
            current_budget = compute_entropy_budget(logits_curr, alpha, min_budget, max_budget,
                                                  curvature=curvature, step_decay=step_decay)
            
            # 5. Build Graph & Priority (if cascading needed)
            # 5. Build Graph & Priority (if cascading needed)
            if current_cascade_depth > 0:
                batch_entropy = compute_entropy(logits_curr)
                
                # Normalize scores for cosine similarity (already normalized in func, but safe check)
                normalized_scores = hidden_states
                
                # We can't build full NxN graph easily (expensive). 
                # We only check relations from `newly_unmasked` to others.
                
                remask_candidates = {} # pos -> priority
                
                for b in range(batch_size):
                    # Candidates to check: existing generated tokens
                    existing_tokens = [p for p in generation_history.keys() if x[b, p] != mask_id]
                    
                    unmasked_in_this_step = newly_unmasked[b]
                    
                    for i in unmasked_in_this_step:
                        # Vectorized cosine similarity: i vs all existing
                        if not existing_tokens: continue
                        
                        # i's score: [1, Dim]
                        score_i = normalized_scores[b, i].unsqueeze(0)
                        
                        # Others scores: [N, Dim]
                        others_indices = torch.tensor(existing_tokens, device=device)
                        score_others = normalized_scores[b, others_indices]
                        
                        # Cosine sim: [1, N]
                        cos_sims = torch.mm(score_i, score_others.t()).abs().squeeze(0)
                        
                        # Filter by threshold
                        connected_indices = torch.where(cos_sims > graph_threshold)[0]
                        
                        for idx_in_indices in connected_indices:
                            j = existing_tokens[idx_in_indices.item()]
                            edge_weight = cos_sims[idx_in_indices].item()
                            
                            # Priority = Edge * Entropy
                            entropy_j = batch_entropy[b, j].item()
                            priority = edge_weight * entropy_j
                            
                            if j not in remask_candidates:
                                remask_candidates[j] = priority
                            else:
                                remask_candidates[j] = max(remask_candidates[j], priority)
                
                # 6. Apply Remask with Budget
                if remask_candidates:
                    sorted_cands = sorted(remask_candidates.items(), key=lambda item: item[1], reverse=True)
                    
                    # Apply top-k
                    count = 0
                    for pos, prio in sorted_cands:
                        if count >= current_budget:
                            break
                        # Do not remask highly confident ones? 
                        # SG-GA implicitly handles this via entropy (low entropy -> low priority)
                        # But we can add safety check if priority is too low
                        
                        if x[0, pos] != mask_id: # Simplify to batch 0 for now or handle batch properly
                             # Note: remask_candidates logic above was mixed batch. 
                             # Assuming batch=1 for typical generation or need to separate remask_candidates by batch
                             # Let's handle batch=1 strictly or fix dict to be (b, pos) -> priority
                             pass 
                             
                    # Start over for batch handling
                    # Fix: remask_candidates should be per batch
                    pass 

    # Clean implementation for batch support
    return _decoding_sg_ga_impl(
        model, prompt, gen_length, block_length, temperature, mask_id,
        alpha, tau_low, tau_high, min_budget, max_budget, graph_threshold, confidence_threshold,
        trace_log_path=trace_log_path, sample_id=sample_id, shard_id=shard_id
    )

def _decoding_sg_ga_impl(
    model, prompt, gen_length, block_length, temperature, mask_id,
    alpha, tau_low, tau_high, min_budget, max_budget, graph_threshold, confidence_threshold=0.75,
    trace_log_path=None, sample_id=None, shard_id=None
):
    device = model.device
    batch_size = prompt.shape[0]
    
    x = torch.full((batch_size, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt.shape[1]] = prompt.clone()
    
    generation_history = {}
    current_step = 0
    prev_score_vectors = None
    
    # Block-wise generation
    num_blocks = gen_length // block_length
    
    # Safety: Cooldown to prevent oscillation
    remask_cooldown = {} # (b, pos) -> last_remask_step
    
    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = block_start + block_length
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        
        loop_step = 0
        max_loop_steps = 100 # Increased to 100 to match WINO's NFE distribution (Avg ~48)
        
        while block_mask_index.any():
            current_step += 1
            loop_step += 1
            
            # Periodic logging to check progress
            if current_step % 10 == 0:  # Log occasionally
                 print(f"[SG-GA Status] Block {num_block}, Step {current_step} (Loop {loop_step}), Masked: {block_mask_index.sum().item()}", flush=True)
            
            # Safety Break with Detailed Logging
            if loop_step > max_loop_steps:
                print(f"\n[SG-GA WARNING] Max Loop Steps ({max_loop_steps}) Reached!", flush=True)
                print(f" - Block: {num_block}, Global Step: {current_step}, Loop Step: {loop_step}", flush=True)
                print(f" - Remaining Masks: {block_mask_index.sum().item()}", flush=True)
                print(f" - Action: Forcing mask fill for remaining tokens and breaking block.", flush=True)
                
                # Force fill masks with current prediction and break
                with torch.no_grad():
                    logits = model(x).logits
                    x0 = torch.argmax(logits, dim=-1)
                    mask_indices = torch.nonzero(x == mask_id, as_tuple=True)
                    x[mask_indices] = x0[mask_indices]
                break
            
            # FORCE PROGRESS Strategy:
            # If loop_step is high (e.g. > 50), disable remasking to ensure convergence
            force_convergence = (loop_step > 50)
            
            # OPTIMIZATION 4: Unmask Threshold Annealing
            # Decay from confidence_threshold (0.75) to 0.5 as we approach max_loop_steps
            # This helps clear stubborn tokens in late stages without forcing 1-by-1
            progress_ratio = min(1.0, loop_step / max_loop_steps)
            current_conf_thresh = confidence_threshold - (confidence_threshold - 0.5) * progress_ratio
            
            # --- 1. Unified Forward (OPTIMIZATION 1: NFE Halving) ---
            # Get logits AND hidden states in one go
            outputs = model(x, output_hidden_states=True)
            logits = outputs.logits
            # Last layer hidden state (usually the last item in tuple)
            # LLaMA/Mistral style: hidden_states is a tuple of (layer_0, ..., layer_N)
            # We want the last one.
            current_hidden = outputs.hidden_states[-1] 
            
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            p = F.softmax(logits.to(torch.float64), dim=-1)
            x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            
            mask_index = (x == mask_id)
            mask_index[:, block_end:] = False
            
            confidence = torch.where(mask_index, x0_p, -np.inf)
            
            # Use annealed threshold
            high_conf_mask = (confidence > current_conf_thresh) & mask_index
            
            if high_conf_mask.sum() == 0:
                max_conf_pos = confidence.argmax(dim=1)
                for b in range(batch_size):
                    high_conf_mask[b, max_conf_pos[b]] = True
                    
            newly_unmasked = []
            for b in range(batch_size):
                newly_unmasked.append(torch.where(high_conf_mask[b])[0].tolist())
                
            x[high_conf_mask] = x0[high_conf_mask]
            
            for b in range(batch_size):
                # Only track positions generated by model, not prompt
                for pos in newly_unmasked[b]:
                    generation_history[(b, pos)] = current_step
            
            # --- 2. Calculate Theoretical Metrics (Attention/Hidden based) ---
            # OPTIMIZATION: Reuse hidden states from first forward (No extra forward!)
            logits_curr = logits
            hidden_states = current_hidden 
            # Note: We skip normalization here as compute_curvature might expect it or handle it.
            # Usually compute_hidden_representation did normalization. Let's check or just normalize here.
            # Assuming compute_hidden_representation logic:
            # "hidden_states = outputs.hidden_states[-1] ; return F.normalize(hidden_states, dim=-1), logits"
            # So we should normalize.
            hidden_states = F.normalize(hidden_states, p=2, dim=-1)
            
            curvature = compute_curvature(hidden_states, prev_score_vectors)
            prev_score_vectors = hidden_states.detach() 
            
            # Curvature-based cascade depth
            if curvature >= tau_high:
                cascade_depth = 2
            elif curvature >= tau_low:
                cascade_depth = 1
            else:
                cascade_depth = 0
                
            # Entropy-based budget (Proxy for Fisher)
            # Dynamic Step Decay: Linear decay to 0 at max_loop_steps
            step_decay = max(0.0, 1.0 - (loop_step / max_loop_steps)) 
            if force_convergence:
                step_decay = 0.0 # Strict cut-off
                
            theoretical_budget = compute_entropy_budget(logits_curr, alpha, min_budget, max_budget, 
                                          curvature=curvature, step_decay=step_decay)
            
            # === LOGGING START ===
            log_entry = {
                "step": current_step,
                "loop_step": loop_step,
                # Identity fields
                "sample_id": sample_id if sample_id is not None else -1, 
                "shard_id": shard_id if shard_id is not None else "main",
                "masked_count": block_mask_index.sum().item(),
                "metrics": {
                    "curvature": float(curvature),
                    "theoretical_budget": float(theoretical_budget),
                    "step_decay": float(step_decay),
                    "cascade_depth": cascade_depth
                },
                "unmasked": [],
                "remasked": []
            }
            
            for b in range(batch_size):
                 unmasked_ids = newly_unmasked[b]
                 log_entry["unmasked"].append({"batch": b, "ids": unmasked_ids, "count": len(unmasked_ids)})
            # === LOGGING END ===

            budget = theoretical_budget

            if cascade_depth > 0 and not force_convergence:
                batch_entropy = compute_entropy(logits_curr) # [B, L]
                # Hidden states are already normalized in compute_hidden_representation
                normalized_scores = hidden_states 
                
                # Debug: Check graph connectivity stats
                # if loop_step % 10 == 0:
                #     print(f"  > [Metrics] Curv: {curvature:.4f}, Budget: {budget}, Decay: {step_decay:.2f}", flush=True)
                
                remask_requests = [] # List of (b, pos, priority)
                
                for b in range(batch_size):
                    # STRICT CONVERGENCE CHECK:
                    # Ensure we remask strictly FEWER than we just unmasked (or a fraction)
                    # WINO approach: remask <= unmask - 1
                    num_unmasked_this_step = len(newly_unmasked[b])
                    
                    if num_unmasked_this_step <= 1:
                        # If we barely made progress, don't remask anything to force forward movement
                        current_budget_b = 0
                    else:
                        # Cap budget at (unmasked - 1)
                        # This guarantees net +1 token progress per step
                        current_budget_b = min(budget, num_unmasked_this_step - 1)
                        
                    if current_budget_b <= 0: continue

                    # OPTIMIZATION: Protect newly unmasked tokens from immediate remasking
                    # Exclude tokens that were just added in this step
                    unmasked_this = set(newly_unmasked[b])
                    
                    existing_gen_tokens = [p for (bb, p) in generation_history.keys() 
                                         if bb == b and x[b, p] != mask_id and p not in unmasked_this]
                    if not existing_gen_tokens: continue
                    
                    if not unmasked_this: continue
                    
                    # Convert to tensor for batch ops
                    target_indices = torch.tensor(existing_gen_tokens, device=device)
                    source_indices = torch.tensor(list(unmasked_this), device=device)
                    
                    # Scores [N_source, Dim], [N_target, Dim]
                    src_scores = normalized_scores[b, source_indices]
                    tgt_scores = normalized_scores[b, target_indices]
                    
                    # Sim [N_source, N_target]
                    # OPTIMIZATION 3: Sparse Graph (Remove .abs())
                    # Only allow positive correlations to trigger remasking
                    sim_matrix = torch.mm(src_scores, tgt_scores.t())
                    
                    # Find connections > threshold
                    # We want max priority for each target from any source
                    max_sim_per_target, _ = sim_matrix.max(dim=0) # [N_target]
                    
                    # Filter & Compute Priority
                    # Priority = Edge(Sim) * Entropy
                    
                    current_entropies = batch_entropy[b, target_indices]
                    priorities = max_sim_per_target * current_entropies
                    
                    # Identify candidates
                    mask = (max_sim_per_target > graph_threshold)
                    
                    valid_cands_indices = torch.where(mask)[0]
                    for idx in valid_cands_indices:
                        pos = existing_gen_tokens[idx.item()]
                        prio = priorities[idx].item()
                        remask_requests.append((b, pos, prio))
                        
                # Sort globally or per batch? Per batch usually.
                # Let's process per batch
                requests_by_batch = {}
                for b, pos, prio in remask_requests:
                    if b not in requests_by_batch: requests_by_batch[b] = []
                    requests_by_batch[b].append((pos, prio))
                
                for b in requests_by_batch:
                    cands = sorted(requests_by_batch[b], key=lambda i: i[1], reverse=True)
                    
                    # Recalculate budget for this batch (same logic as above)
                    num_unmasked_this_step = len(newly_unmasked[b])
                    if num_unmasked_this_step <= 1:
                        effective_budget = 0
                    else:
                        effective_budget = min(budget, num_unmasked_this_step - 1)
                    
                    if "effective_budgets" not in log_entry["metrics"]: log_entry["metrics"]["effective_budgets"] = {}
                    log_entry["metrics"]["effective_budgets"][b] = float(effective_budget)

                    applied_count = 0
                    for pos, prio in cands:
                        if applied_count >= effective_budget: break
                        
                        # Cooldown Check (e.g., 3 steps)
                        last_remask = remask_cooldown.get((b, pos), -999)
                        if current_step - last_remask < 3:
                            continue
                            
                        if x[b, pos] != mask_id:
                            x[b, pos] = mask_id
                            if (b, pos) in generation_history:
                                del generation_history[(b, pos)]
                            
                            # Update Cooldown
                            remask_cooldown[(b, pos)] = current_step
                            applied_count += 1
                            log_entry["remasked"].append({"batch": b, "pos": pos, "priority": float(prio)})
            
            # Dump log entry
            try:
                import json
                import os
                
                # Determine log path
                if trace_log_path:
                    # Use provided absolute path
                    final_path = trace_log_path
                    # Ensure directory exists (though run_experiments should handle this, safety first)
                    os.makedirs(os.path.dirname(final_path), exist_ok=True)
                else:
                    # Fallback to local file
                    if shard_id is not None:
                        final_path = f"trace_sag_ga_shard{shard_id}.jsonl"
                    else:
                        final_path = "trace_sag_ga.jsonl"
                    
                with open(final_path, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
            except Exception as e:
                # print(f"Logging failed: {e}")
                pass
                            
            block_mask_index = (x[:, block_start:block_end] == mask_id)
            
    return x, current_step


# ============================================================================
# SGGA-WINO Hybrid Decoding
# ============================================================================

@torch.no_grad()
def decoding_sgga_wino_hybrid(
    model,
    prompt: torch.Tensor,
    gen_length: int = 256,
    block_length: int = 128,
    temperature: float = 0.0,
    mask_id: int = 126336,
    # WINO parameters
    threshold: float = 0.5,
    threshold_back: float = 0.9,
    # SGGA parameters
    graph_threshold: float = 0.2,
    alpha: float = 0.5,
    min_budget: int = 3,
    max_budget: int = 15,
    tau_low: float = 0.05,
    tau_high: float = 0.15,
    **kwargs
) -> Tuple[torch.Tensor, int]:
    """
    SGGA-WINO Hybrid: Combines WINO's speculative decoding with SGGA's graph-aware refinement.
    
    Strategy:
    1. Use WINO's block-wise lookahead structure for efficient parallel generation
    2. Apply SGGA's graph-based analysis for intelligent remasking
    3. Combine confidence thresholds with graph coherence for accept/reject
    
    Args:
        model: LLaDA model
        prompt: Input prompt tensor
        gen_length: Generation length
        block_length: Block size for WINO-style lookahead
        temperature: Sampling temperature
        mask_id: Mask token ID
        threshold: WINO forward confidence threshold
        threshold_back: WINO backward confidence threshold
        graph_threshold: SGGA cosine similarity threshold
        alpha: SGGA entropy scaling factor
        min_budget: Minimum remask budget
        max_budget: Maximum remask budget
        tau_low: Low curvature threshold
        tau_high: High curvature threshold
    
    Returns:
        (generated_sequence, num_steps)
    """
    device = model.device
    batch_size = prompt.shape[0]
    
    # WINO-style setup: extended block for lookahead
    x_block = torch.full(
        (batch_size, prompt.shape[1] + gen_length + block_length),
        mask_id,
        dtype=torch.long,
        device=device
    )
    x_block[:, :prompt.shape[1]] = prompt.clone()
    
    prompt_index = (x_block != mask_id)
    
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    total_steps = 0
    
    # SGGA tracking
    generation_history = {}
    prev_hidden = None
    
    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = block_start + block_length
        
        # WINO-style mask and unmask tracking
        mask_index_block = (x_block == mask_id)
        mask_index_block[:, block_end:] = False
        
        unmask_index_block = torch.full_like(mask_index_block, False)
        unmask_index_block[:, -block_length:] = ~mask_index_block[:, block_start:block_end]
        
        # WINO-style position IDs and attention mask
        position_ids = torch.cat([
            torch.arange(prompt.shape[1] + gen_length, device=device),
            torch.arange(block_start, block_end, device=device)
        ])
        
        attention_mask = torch.ones(
            batch_size, 1, x_block.shape[1], x_block.shape[1],
            dtype=torch.bool,
            device=device
        )
        attention_mask[:, :, :, -block_length:] = False
        attention_mask[:, :, -block_length:, -block_length:] = torch.ones(
            block_length, block_length, dtype=torch.bool, device=device
        )
        attention_mask[:, :, -block_length:, block_start:block_end] = ~torch.eye(
            block_length, dtype=torch.bool, device=device
        )
        
        last_accept = 30
        block_step = 0
        
        while mask_index_block.any():
            block_step += 1
            total_steps += 1
            
            # === Phase 1: WINO-style Speculative Generation ===
            max_accept = min(max(int(mask_index_block.sum() * 0.7), 5), 20)
            
            # Forward pass (with hidden states for SGGA)
            outputs = model(
                x_block,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True
            )
            logits = outputs.logits
            
            # Sample with Gumbel noise
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            
            # Copy already unmasked tokens
            unmask_index_block_shift_left = torch.zeros_like(unmask_index_block)
            unmask_index_block_shift_left[:, block_start:block_end] = unmask_index_block[:, -block_length:]
            x0[unmask_index_block] = x_block[unmask_index_block_shift_left]
            
            # Compute confidence
            p = F.softmax(logits.to(torch.float64), dim=-1)
            x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            
            x0 = torch.where(mask_index_block, x0, x_block)
            confidence = torch.where(mask_index_block, x0_p, -np.inf)
            confidence_back = torch.where(unmask_index_block, x0_p, np.inf)
            
            # === Phase 2: SGGA Graph-Aware Analysis ===
            # Extract hidden states
            hidden_states = outputs.hidden_states[-1]
            hidden_states = F.normalize(hidden_states, p=2, dim=-1)
            
            # Compute curvature
            curvature = 0.0
            if prev_hidden is not None:
                delta = hidden_states - prev_hidden
                curvature = torch.norm(delta, dim=-1).mean().item()
            prev_hidden = hidden_states.detach()
            
            # Compute entropy
            batch_entropy = compute_entropy(logits)
            
            # Dynamic cascade depth based on curvature
            cascade_depth = 0
            if curvature >= tau_high:
                cascade_depth = 2
            elif curvature >= tau_low:
                cascade_depth = 1
            
            # Dynamic budget
            step_decay = max(0.0, 1.0 - (block_step * 0.1))
            budget = compute_entropy_budget(
                logits, alpha, min_budget, max_budget,
                curvature=curvature, step_decay=step_decay
            )
            
            # === Phase 3: Hybrid Accept/Reject ===
            # Accept tokens with high confidence
            transfer_index = confidence > threshold
            
            # Apply SGGA graph coherence check
            if cascade_depth > 0 and transfer_index.any():
                # Build graph for newly accepted tokens
                for b in range(batch_size):
                    accepted_positions = torch.where(transfer_index[b])[0].tolist()
                    
                    if not accepted_positions:
                        continue
                    
                    # Check graph coherence with existing tokens
                    existing_tokens = [p for p in range(block_start, block_end)
                                     if x_block[b, p] != mask_id and p not in accepted_positions]
                    
                    if existing_tokens:
                        accepted_tensor = torch.tensor(accepted_positions, device=device)
                        existing_tensor = torch.tensor(existing_tokens, device=device)
                        
                        # Compute cosine similarity
                        accepted_hidden = hidden_states[b, accepted_tensor]
                        existing_hidden = hidden_states[b, existing_tensor]
                        
                        sim_matrix = torch.mm(accepted_hidden, existing_hidden.t())
                        max_sim, _ = sim_matrix.max(dim=1)
                        
                        # Reject tokens with low graph coherence
                        low_coherence = max_sim < graph_threshold
                        for idx, pos in enumerate(accepted_positions):
                            if low_coherence[idx]:
                                transfer_index[b, pos] = False
            
            # Limit accept count
            if transfer_index.sum() > max_accept:
                for b in range(batch_size):
                    if transfer_index[b].sum() > max_accept:
                        conf_b = confidence[b]
                        _, indices = torch.topk(conf_b, k=max_accept, largest=True)
                        new_transfer = torch.zeros_like(transfer_index[b])
                        new_transfer[indices] = True
                        transfer_index[b] = new_transfer
            
            # Always transfer at least one token
            if not transfer_index.any():
                for b in range(batch_size):
                    max_confidence_index = torch.argmax(confidence[b])
                    transfer_index[b, max_confidence_index] = True
            
            # Apply transfer
            x_block[transfer_index] = x0[transfer_index]
            num_accept = transfer_index.sum()
            
            # Update generation history
            for b in range(batch_size):
                positions = torch.where(transfer_index[b])[0]
                for pos in positions:
                    generation_history[(b, pos.item())] = total_steps
            
            # === Phase 4: SGGA-enhanced Backward Remasking ===
            remask_index = torch.zeros_like(transfer_index)
            
            if num_accept > 1 and cascade_depth > 0:
                # WINO-style backward remasking
                wino_remask = confidence_back < threshold_back
                
                # SGGA-style graph-based remasking
                sgga_remask_candidates = []
                
                for b in range(batch_size):
                    unmask_positions = torch.where(unmask_index_block[b, -block_length:])[0].tolist()
                    
                    if not unmask_positions:
                        continue
                    
                    # Compute priorities
                    for pos_idx in unmask_positions:
                        actual_pos = block_start + pos_idx
                        
                        # Priority = entropy (uncertainty)
                        entropy_val = batch_entropy[b, actual_pos].item()
                        
                        # Check graph connectivity
                        pos_hidden = hidden_states[b, actual_pos].unsqueeze(0)
                        other_positions = [p for p in range(block_start, block_end)
                                         if x_block[b, p] != mask_id and p != actual_pos]
                        
                        if other_positions:
                            other_tensor = torch.tensor(other_positions, device=device)
                            other_hidden = hidden_states[b, other_tensor]
                            
                            sim = torch.mm(pos_hidden, other_hidden.t()).squeeze(0)
                            max_sim_val = sim.max().item()
                            
                            # Priority = similarity * entropy
                            priority = max_sim_val * entropy_val
                            
                            if max_sim_val > graph_threshold:
                                sgga_remask_candidates.append((b, actual_pos, priority))
                
                # Combine WINO and SGGA remasking
                if wino_remask.sum() >= last_accept:
                    num_remask = last_accept - 1
                    confidence_flat = confidence_back.view(-1)
                    temp_mask = torch.zeros_like(confidence_flat, dtype=torch.bool)
                    _, indices = torch.topk(confidence_flat, k=num_remask, largest=False)
                    temp_mask[indices] = True
                    remask_index = temp_mask.view(confidence_back.shape)
                else:
                    # Use SGGA candidates with budget
                    sgga_remask_candidates.sort(key=lambda x: x[2], reverse=True)
                    
                    effective_budget = min(budget, num_accept.item() - 1)
                    for i, (b, pos, _) in enumerate(sgga_remask_candidates):
                        if i >= effective_budget:
                            break
                        
                        # Map to block coordinates
                        if block_start <= pos < block_end:
                            block_pos = pos - block_start + (x_block.shape[1] - block_length)
                            remask_index[b, block_pos] = True
            
            # Apply remasking
            remask_index_shift = torch.zeros_like(remask_index)
            remask_index_shift[:, block_start:block_end] = remask_index[:, -block_length:]
            x_block[remask_index_shift] = mask_id
            
            # Update masks
            mask_index_block[transfer_index] = False
            mask_index_block[remask_index_shift] = True
            
            transfer_index_shift = torch.zeros_like(transfer_index)
            transfer_index_shift[:, -block_length:] = transfer_index[:, block_start:block_end]
            unmask_index_block[transfer_index_shift] = True
            unmask_index_block[remask_index] = False
            
            last_accept = num_accept.item()
    
    return x_block[:, :prompt.shape[1] + gen_length], total_steps
