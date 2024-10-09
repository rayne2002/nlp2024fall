import torch
import copy
import math
import torch.nn as nn
from torch.nn.functional import pad
import sacrebleu

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


def beam_search_decode(model, src, src_mask, max_len, start_symbol, beam_size, end_idx):
    # # Your code here
    # memory = model.encode(src, src_mask)

    # # Initiate beam search with a single sequence containing only the start symbol
    # beam = [(torch.tensor([start_symbol], dtype=torch.long), 0)]  # Each element is a tuple (sequence, score)
    # completed_sq = []

    # # Execute the loop until the sequence reaches the maximum permitted length
    # for _ in range(max_len):
    #     # Gather all potential sequences at this step
    #     all_candid = []
    #     for seq, score in beam:
    #         # Stop expanding the sequence if it ends with the end symbol
    #         if seq[-1].item() == end_idx:
    #             completed_sq.append((seq, score))
    #             continue

    #         # Compute next probabilities using the current sequence
    #         tgt_mask = subsequent_mask(len(seq)).type_as(src_mask.data)
    #         out = model.decode(memory, src_mask, seq.unsqueeze(0), tgt_mask)

    #         # Extract log probabilities for the latest element in the sequence
    #         log_probs = model.generator(out[:, -1])

    #         # Identify the top k possible next steps
    #         topk_log_probs, topk_indices = torch.topk(log_probs, beam_size)

    #         # Construct new sequences by appending top candidates to current sequence
    #         for i in range(beam_size):
    #             candidate = torch.cat([seq, topk_indices[0, i].unsqueeze(0)], dim=0)
    #             candidate_score = score + topk_log_probs[0, i].item()  # Update sequence score
    #             all_candid.append((candidate, candidate_score))

    #     # Select the top scoring sequences to continue beam search
    #     ordered = sorted(all_candid, key=lambda x: x[1], reverse=True)
    #     beam = ordered[:beam_size]

    #     # Terminate if all sequences are complete
    #     if len(completed_sq) == beam_size:
    #         break

    # # Return the sequence with the highest score if any have completed
    # if completed_sq:
    #     return sorted(completed_sq, key=lambda x: x[1], reverse=True)[0][0]

    # # Otherwise, return the highest scoring ongoing sequence
    # return beam[0][0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    src = src.to(device)           # Move src tensor to device (GPU/CPU)
    src_mask = src_mask.to(device)
    memory = model.encode(src, src_mask)
    beam = [(torch.tensor([start_symbol]), 0.0)]  # List of tuples (sequence, score)
    
    # Iterating through sequence length
    for _ in range(max_len - 1):
        new_beam = []
        for seq, score in beam:
            if seq[-1] == end_idx:
                # If sequence has ended, keep it in the beam
                new_beam.append((seq, score))
                continue
            # Decode the next token
            out = model.decode(memory, src_mask, seq.unsqueeze(0), subsequent_mask(seq.size(0)).type_as(src.data))
            prob = model.generator(out[:, -1])  # Predict the next token
            
            # Get top beam_size tokens
            top_probs, top_idx = torch.topk(prob, beam_size)
            
            # Add new sequences to the beam
            for i in range(beam_size):
                next_seq = torch.cat([seq, torch.tensor([top_idx[i]])])
                next_score = score + top_probs[i].item()
                new_beam.append((next_seq, next_score))
        
        # Sort the beam by score and keep the best beam_size sequences
        new_beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:beam_size]
        beam = new_beam
    
    # Return the sequence with the highest score
    return beam[0][0]  # Return the best sequence



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

