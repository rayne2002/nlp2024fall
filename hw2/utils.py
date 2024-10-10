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


def beam_search_decode(model, src, src_mask, max_len, start_symbol, beam_size, end_idx):
    # # Your code here
    memory = model.encode(src, src_mask)
    memory = memory.expand(beam_size, *memory.shape[1:])
    src_mask = src_mask.expand(beam_size, *src_mask.shape[1:])

    # Initialize the beam with the start symbol and scores set to zeros
    ys = torch.zeros(beam_size, 1).fill_(start_symbol).type_as(src.data).cuda()
    scores = torch.zeros(beam_size).cuda()
    completed_sequences = []

    for _ in range(max_len):
        all_candidates = []

        # Generate beam candidates for each sequence in the current beam
        for i in range(beam_size):
            if ys[i, -1].item() == end_idx and _ > 5:
                completed_sequences.append((ys[i], scores[i]))
                continue

            # Generate the next probabilities from the model
            tgt_mask = subsequent_mask(ys.size(1)).type_as(src_mask.data)
            out = model.decode(memory[i:i+1], src_mask[i:i+1], ys[i:i+1], tgt_mask)
            log_probs = model.generator(out[:, -1])

            # Take the top k candidates (beam size)
            topk_log_probs, topk_indices = torch.topk(log_probs, beam_size)

            # Append each top token to the current sequence and update the score
            for k in range(len(topk_indices)):
                # Ensure ys[i] is treated as a 2D tensor
                ys_i = ys[i].unsqueeze(0) if ys[i].dim() == 1 else ys[i]

                # Reshape topk_indices[k] properly for concatenation
                next_token = topk_indices[k].unsqueeze(0).unsqueeze(0)  # Shape it as (1, 1)

                # Concatenate the token to the sequence
                new_seq = torch.cat([ys_i, next_token], dim=1)  # Concatenating along sequence length

                new_score = scores[i] + topk_log_probs[k].item()
                all_candidates.append((new_seq, new_score))

        if len(all_candidates) == 0:
            raise RuntimeError("No candidates generated at this step.")

        ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        beam = ordered[:beam_size]

        # Update ys and scores with the new beam sequences and scores
        ys = torch.stack([seq for seq, _ in beam])
        scores = torch.tensor([score for _, score in beam]).cuda()

        if len(completed_sequences) == beam_size:
            break

    if completed_sequences:
        return sorted(completed_sequences, key=lambda x: x[1], reverse=True)[0][0]

    return beam[0][0]



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

