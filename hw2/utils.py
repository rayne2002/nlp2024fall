import torch
import copy
import math
import torch.nn as nn
from torch.nn.functional import pad
import sacrebleu


## Dummy functions defined to use the same function run_epoch() during eval
class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0


def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):
        pe = self.pe.unsqueeze(0)
        x = x + pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)



def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys


import torch
import torch.nn.functional as F
import math

# Assuming you have a subsequent mask function defined

def subsequent_mask(size):
    """
    Mask out subsequent positions (used for decoder).
    """
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0

def beam_search_decode(model, src, src_mask, max_len, start_symbol, beam_size, end_idx):
    """
    Implement beam search decoding with 'beam_size' width
    """
    device = src.device

    # Step 1: Encode source input using the model
    memory = model.encode(src, src_mask)

    # Initialize decoder input and scores
    ys = torch.ones(beam_size, 1).fill_(start_symbol).type_as(src.data)
    scores = torch.zeros(beam_size, device=device)  # Keep log-probabilities

    # Keep track of whether sequences have ended
    ended = torch.zeros(beam_size, dtype=torch.bool, device=device)

    # Expand memory and src_mask to beam_size
    memory = memory.expand(beam_size, -1, -1)
    src_mask = src_mask.expand(beam_size, -1, -1)

    for i in range(max_len - 1):
        # Decode using the model, memory, and source mask
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )

        # Calculate probabilities for the next token
        prob = model.generator(out[:, -1, :])  # Shape: (beam_size, vocab_size)
        vocab_size = prob.size(1)

        # For beams that have ended, set log_prob to -inf except at end_idx
        log_prob = prob.clone()
        for beam_idx in range(beam_size):
            if ended[beam_idx]:
                log_prob[beam_idx, :] = float('-inf')
                log_prob[beam_idx, end_idx] = 0.0  # Log probability of 1 at end_idx

        # Update scores
        expanded_scores = scores.unsqueeze(1) + log_prob  # Shape: (beam_size, vocab_size)

        # Flatten scores to consider all possible next steps
        flat_scores = expanded_scores.view(-1)  # Shape: (beam_size * vocab_size)

        # Get top-k scores and indices
        next_scores, next_positions = torch.topk(flat_scores, beam_size, dim=0)

        # Extract beam indices and token indices from top-k scores
        beam_indices = next_positions // vocab_size
        token_indices = next_positions % vocab_size

        # Prepare next decoder input
        ys = ys[beam_indices]  # Select the corresponding beams
        ys = torch.cat([ys, token_indices.unsqueeze(1)], dim=1)  # Append new tokens

        # Update scores
        scores = next_scores

        # Handle end token condition
        ended = ended[beam_indices] | (token_indices == end_idx)

        # Check if all beams are finished, exit
        if ended.all():
            break

    # Return the top-scored sequence
    # Convert the top scored sequence to a list of tokens
    best_score_index = scores.argmax()
    best_sequence = ys[best_score_index]
    return [best_sequence.cpu().numpy()]




# Example usage
# model, src, src_mask, max_len, start_symbol, beam_size, end_idx are assumed to be defined before usage
# result = beam_search_decode(model, src, src_mask, max_len, start_symbol, beam_size, end_idx)




def collate_batch(
    batch,
    src_pipeline,
    tgt_pipeline,
    src_vocab,
    tgt_vocab,
    device,
    max_padding=128,
    pad_id=2,
):
    bs_id = torch.tensor([0], device=device)  # <s> token id
    eos_id = torch.tensor([1], device=device)  # </s> token id
    src_list, tgt_list = [], []
    for s in batch:
        _src = s['de']
        _tgt = s['en']
        processed_src = torch.cat(
            [
                bs_id,
                torch.tensor(
                    src_vocab(src_pipeline(_src)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        processed_tgt = torch.cat(
            [
                bs_id,
                torch.tensor(
                    tgt_vocab(tgt_pipeline(_tgt)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        src_list.append(
            # warning - overwrites values for negative values of padding - len
            pad(
                processed_src,
                (
                    0,
                    max_padding - len(processed_src),
                ),
                value=pad_id,
            )
        )
        tgt_list.append(
            pad(
                processed_tgt,
                (0, max_padding - len(processed_tgt)),
                value=pad_id,
            )
        )

    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)
    return (src, tgt)


def remove_start_end_tokens(sent):

    if sent.startswith('<s>'):
        sent = sent[3:]

    if sent.endswith('</s>'):
        sent = sent[:-4]

    return sent


def compute_corpus_level_bleu(refs, hyps):

    refs = [remove_start_end_tokens(sent) for sent in refs]
    hyps = [remove_start_end_tokens(sent) for sent in hyps]

    bleu = sacrebleu.corpus_bleu(hyps, [refs])

    return bleu.score

