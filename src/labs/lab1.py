import math
import torch
import torch.nn as nn

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

MODEL_ID = "openai-community/gpt2"

VOCAB_SIZE = 50257
CONTEXT_LENGTH = 1024

N_EMBD = 768
N_LAYER = 12
N_HEAD = 12
HEAD_DIM = 64
N_INNER = 3072

class Conv1D(nn.Module):
    def __init__(self, out_features: int, in_features: int):
        super().__init__()
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_shape = x.shape[:-1] + (self.out_features,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        return x.view(output_shape)

class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_embd = N_EMBD
        self.n_head = N_HEAD
        self.head_dim = HEAD_DIM
        self.c_attn = Conv1D(3 * self.n_embd, self.n_embd)
        self.c_proj = Conv1D(self.n_embd, self.n_embd)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Attention Head
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1,2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1,2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1,2)
        
        # QK Trans.
        attn_scores = q @ k.transpose(-1, -2)
        scale = torch.full([], self.head_dim**0.5, dtype=attn_scores.dtype, device=attn_scores.device)
        attn_scores = attn_scores / scale
        
        # Casual Mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        mask_value = torch.full([], torch.finfo(attn_scores.dtype).min, dtype=attn_scores.dtype, device=attn_scores.device)
        attn_scores = torch.where(causal_mask[None, None, :, :], attn_scores, mask_value)
        
        # Padding Mask
        key_mask = attention_mask[:, None, None, :].bool()
        attn_scores = torch.where(key_mask, attn_scores, mask_value)
        
        # Softmax & Attention Probability
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Merge Output
        output = attn_probs @ v
        output = (
            output
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.n_embd)
        )
        
        output = self.c_proj(output)
        
        return output

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = Conv1D(N_INNER, N_EMBD)
        self.c_proj = Conv1D(N_EMBD, N_INNER)

    def forward(self, x):
        x = self.c_fc(x)
        x = gelu_lab1(x)
        x = self.c_proj(x)

        return x

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.LayerNorm(N_EMBD, eps=1e-5)
        self.attn = CausalSelfAttention()
        self.ln_2 = nn.LayerNorm(N_EMBD, eps=1e-5)
        self.mlp = MLP()

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT2Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.wpe = nn.Embedding(CONTEXT_LENGTH, N_EMBD)
        self.ln_f = nn.LayerNorm(N_EMBD, eps=1e-5)
        self.h = nn.ModuleList(
            [Block() for _ in range(N_LAYER)]
        )

    def forward(self, input_ids, attention_mask, position_ids):
        token_embeddings = self.wte(input_ids)
        position_embeddings = self.wpe(position_ids)
        x = token_embeddings + position_embeddings
        
        for block in self.h:
            x = block(x, attention_mask)
        
        x = self.ln_f(x)
        
        return x

class GPT2LMHeadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = GPT2Transformer()
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.transformer.wte.weight # Weight Tying

    def forward(self, input_ids, attention_mask, position_ids):
        hidden_states = self.transformer(input_ids, attention_mask, position_ids)
        logits = self.lm_head(hidden_states)
        return logits
    
def gelu_lab1(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (
        1.0 
        + torch.tanh(
            math.sqrt(2.0/math.pi) # sqrt(2/pi)
            *(x + 0.044715*torch.pow(x, 3.0)) # x + (0.44715 * x^3)
        )
    )

def load_gpt2():
    checkpoint_path = hf_hub_download(repo_id=MODEL_ID, filename="model.safetensors")
    state_dict = load_file(checkpoint_path)
    model = GPT2LMHeadModel()
    missing, unexpected = model.transformer.load_state_dict(state_dict, strict=False)
    
    print("missing:", missing)
    print("unexpected:", unexpected)
    
    model.lm_head.weight = model.transformer.wte.weight
    model = model.to(dtype=torch.float16)
    model.eval()
    
    return model

def gpt2_complete(
    input: list[str],
    max_seq_length: int = 1024,
) -> tuple[list[str], torch.Tensor]:
    """Generate greedy completions with a from-scratch GPT-2 Small implementation.

    Load pretrained GPT-2 Small weights into manually implemented transformer
    blocks. Generate for the entire batch at once, choosing the highest-logit
    token for every unfinished sequence at each step. Stop each sequence at EOS
    or max_seq_length total tokens, including the prompt.

    Return newly generated text for each prompt and a tensor of pre-selection
    logits shaped (batch_size, decoding_steps, 50257). Fill logits with zero
    after a row has finished while other rows continue.
    """
    # Load Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = load_gpt2()
    
    # Tokenize
    prompt_ids = [
        tokenizer.encode(prompt) for prompt in input
    ]
    lengths = [
        len(ids) for ids in prompt_ids
    ]
    
    # Left Padding
    width = max(lengths)
    eos_id = tokenizer.eos_token_id
    input_ids = torch.full(
        (len(input), width), eos_id, dtype=torch.long
    )
    attention_mask = torch.zeros_like(input_ids)
    
    for i, ids in enumerate(prompt_ids):
        input_ids[i, -len(ids):] = torch.tensor(ids)
        attention_mask[i, -len(ids):] = 1
    
    # Finished Mask
    generated = [[] for _ in input]
    finished = torch.tensor(
        [length >= max_seq_length for length in lengths]
    )
    steps = []
    
    # Generation Loop
    with torch.inference_mode():
        while not bool(finished.all()):
            position_ids = (attention_mask.cumsum(dim=1) - 1).clamp_min(0)
            
            # Forward
            all_logits = model(input_ids, attention_mask, position_ids)
            logits = (all_logits[:, -1, :].clone())
            
            # Clean Finished Logits
            active = ~finished
            logits[~active] = 0
            
            # Greedy
            next_ids = logits.argmax(dim=-1)
            
            # Append EOS
            next_ids[~active] = eos_id
            
            # Save Logits
            steps.append(logits)
            
            # Update Every Sequence
            for i in range(len(input)):
                if active[i]:
                    token = int(next_ids[i])
                    generated[i].append(token)
                    lengths[i] += 1
                    if(token == eos_id or lengths[i] >= max_seq_length):
                        finished[i] = True
            
            input_ids = torch.cat((input_ids, next_ids[:, None]), dim=1)
            attention_mask = torch.cat((attention_mask, active[:, None].to(dtype=attention_mask)), dim=1)
    
    # Decode
    completions = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated]
    
    # Logits Shape
    if steps:
        logits = torch.stack(steps, dim=1)
    else:
        logits = torch.empty((len(input), 0, VOCAB_SIZE), dtype=torch.float16)
    
    return completions, logits