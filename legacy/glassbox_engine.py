import math
import hashlib
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyGlassboxTransformer(nn.Module):
    """
    Tiny inspectable Causal Transformer architecture matching Glassbox blueprint v0.1:
    - 2 Transformer Layers
    - 8 Channel Width (D=8)
    - 2 Attention Heads (H=2, head_dim=4)
    - MLP Expansion Width 16 (8 -> 16 -> 8)
    - GELU Activation
    - Learned Absolute Position Embedding
    - Pre-LayerNorm Structure
    - Tied Input/Output Embedding Weights
    - Causal Masking
    """
    def __init__(self, vocab_size=50, width=8, heads=2, mlp_width=16, layers=2, context=96):
        super().__init__()
        self.vocab_size = vocab_size
        self.width = width
        self.heads = heads
        self.head_dim = width // heads
        self.mlp_width = mlp_width
        self.layers = layers
        self.context = context

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.position_embedding = nn.Embedding(context, width)

        # Layers
        self.block0_norm_att = nn.LayerNorm(width)
        self.block0_q = nn.Linear(width, width, bias=False)
        self.block0_k = nn.Linear(width, width, bias=False)
        self.block0_v = nn.Linear(width, width, bias=False)
        self.block0_att_out = nn.Linear(width, width, bias=False)

        self.block0_norm_mlp = nn.LayerNorm(width)
        self.block0_mlp_up = nn.Linear(width, mlp_width)
        self.block0_mlp_down = nn.Linear(mlp_width, width)

        self.block1_norm_att = nn.LayerNorm(width)
        self.block1_q = nn.Linear(width, width, bias=False)
        self.block1_k = nn.Linear(width, width, bias=False)
        self.block1_v = nn.Linear(width, width, bias=False)
        self.block1_att_out = nn.Linear(width, width, bias=False)

        self.block1_norm_mlp = nn.LayerNorm(width)
        self.block1_mlp_up = nn.Linear(width, mlp_width)
        self.block1_mlp_down = nn.Linear(mlp_width, width)

        self.final_norm = nn.LayerNorm(width)
        # Output embedding is tied to token_embedding.weight in forward pass

    def forward(self, input_ids, capture_tensors=False):
        B, T = input_ids.shape
        pos = torch.arange(0, T, device=input_ids.device).unsqueeze(0)

        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(pos)
        x = tok_emb + pos_emb

        def wrap_tensor(t):
            if isinstance(t, torch.Tensor):
                s = list(t.shape)
                v = t.detach().tolist()
            else:
                s = []
                curr = t
                while isinstance(curr, list):
                    s.append(len(curr))
                    curr = curr[0] if len(curr) > 0 else None
                v = t

            def replace_inf(obj):
                if isinstance(obj, list):
                    return [replace_inf(i) for i in obj]
                elif isinstance(obj, float) and (obj == float('-inf') or math.isinf(obj)):
                    return None
                return obj

            return {"shape": s, "values": replace_inf(v)}

        tensors = {}
        if capture_tensors:
            tensors['token_embedding'] = wrap_tensor(tok_emb)
            tensors['position_embedding'] = wrap_tensor(pos_emb)
            tensors['residual_start'] = wrap_tensor(x)

        def run_block(blk_idx, in_x, norm_att, q_proj, k_proj, v_proj, att_out_proj, norm_mlp, mlp_up, mlp_down):
            # Attention
            normed_att = norm_att(in_x)
            q = q_proj(normed_att)
            k = k_proj(normed_att)
            v = v_proj(normed_att)

            # Reshape into heads: [B, T, width] -> [B, heads, T, head_dim]
            q_heads = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
            k_heads = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
            v_heads = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)

            scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # Causal mask
            causal_mask = torch.tril(torch.ones((T, T), device=input_ids.device)).view(1, 1, T, T)
            masked_scores = scores.masked_fill(causal_mask == 0, float('-inf'))
            att_weights = F.softmax(masked_scores, dim=-1)

            # Weighted values
            w_val = torch.matmul(att_weights, v_heads)
            merged_heads = w_val.transpose(1, 2).contiguous().view(B, T, self.width)
            att_update = att_out_proj(merged_heads)
            res_att = in_x + att_update

            # MLP
            normed_mlp = norm_mlp(res_att)
            mlp_u = mlp_up(normed_mlp)
            mlp_act = F.gelu(mlp_u)
            mlp_d = mlp_down(mlp_act)
            res_out = res_att + mlp_d

            if capture_tensors:
                prefix = f'block{blk_idx}.'
                tensors[prefix + 'norm_attention'] = wrap_tensor(normed_att)
                tensors[prefix + 'q'] = wrap_tensor(q)
                tensors[prefix + 'k'] = wrap_tensor(k)
                tensors[prefix + 'v'] = wrap_tensor(v)
                tensors[prefix + 'q_heads'] = wrap_tensor(q_heads)
                tensors[prefix + 'k_heads'] = wrap_tensor(k_heads)
                tensors[prefix + 'v_heads'] = wrap_tensor(v_heads)
                tensors[prefix + 'scores'] = wrap_tensor(scores)
                tensors[prefix + 'masked_scores'] = wrap_tensor(masked_scores)
                tensors[prefix + 'attention'] = wrap_tensor(att_weights)
                tensors[prefix + 'weighted_values'] = wrap_tensor(w_val)
                tensors[prefix + 'merged_heads'] = wrap_tensor(merged_heads)
                tensors[prefix + 'attention_update'] = wrap_tensor(att_update)
                tensors[prefix + 'residual_attention'] = wrap_tensor(res_att)

                tensors[prefix + 'norm_mlp'] = wrap_tensor(normed_mlp)
                tensors[prefix + 'mlp_up'] = wrap_tensor(mlp_u)
                tensors[prefix + 'mlp_activation'] = wrap_tensor(mlp_act)
                tensors[prefix + 'mlp_update'] = wrap_tensor(mlp_d)
                tensors[prefix + 'residual_out'] = wrap_tensor(res_out)

            return res_out

        x = run_block(0, x, self.block0_norm_att, self.block0_q, self.block0_k, self.block0_v, self.block0_att_out,
                      self.block0_norm_mlp, self.block0_mlp_up, self.block0_mlp_down)

        x = run_block(1, x, self.block1_norm_att, self.block1_q, self.block1_k, self.block1_v, self.block1_att_out,
                      self.block1_norm_mlp, self.block1_mlp_up, self.block1_mlp_down)

        final_x = self.final_norm(x)
        logits = F.linear(final_x, self.token_embedding.weight)
        probs = F.softmax(logits, dim=-1)

        if capture_tensors:
            tensors['final_norm'] = wrap_tensor(final_x)
            tensors['logits'] = wrap_tensor(logits)
            tensors['probabilities'] = wrap_tensor(probs)
            return logits, tensors

        return logits

def default_paragraphs():
    return [
        "pip is a small robot in a quiet garden. pip has two red stones and three blue stones. there are five stones in all.",
        "the red stones sit in a round box. the blue stones sit in a square box. pip opens the round box and sees two stones.",
        "mira visits the garden each morning. mira gives pip one green leaf. pip puts the green leaf beside the round box.",
        "a little lamp lights the garden at night. when the sun rises, pip turns the lamp off. when the sun sets, pip turns the lamp on.",
        "pip counts the red stones first. one stone and one stone make two stones. two red stones and three blue stones make five stones.",
        "mira likes short and clear answers. when mira asks how many red stones there are, pip says two. when she asks how many blue stones there are, pip says three.",
        "one day pip moves one red stone into the square box. the round box now has one stone. the square box now has four stones.",
        "pip puts the red stone back. the round box has two stones again. the square box has three stones again, and the garden is quiet."
    ]

def default_sft_examples():
    return [
        {"prompt": "how many red stones does pip have?", "answer": "two."},
        {"prompt": "how many blue stones does pip have?", "answer": "three."},
        {"prompt": "how many stones are there in all?", "answer": "five."},
        {"prompt": "who visits the garden?", "answer": "mira."},
        {"prompt": "what is pip?", "answer": "a small robot."},
        {"prompt": "where are the red stones?", "answer": "in the round box."},
        {"prompt": "where are the blue stones?", "answer": "in the square box."},
        {"prompt": "what color is the leaf?", "answer": "green."},
        {"prompt": "what does pip turn on at night?", "answer": "the lamp."},
        {"prompt": "what does mira like?", "answer": "short and clear answers."},
        {"prompt": "how many stones are one and one?", "answer": "two."},
        {"prompt": "how many stones are two and three?", "answer": "five."}
    ]

class GlassboxEngine:
    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

        self.paragraphs = default_paragraphs()
        self.validation_paragraphs = [
            "mira sees a small robot near the garden lamp. there are two stones in the round box and three in the square box.",
            "at night the lamp is on. at sunrise the robot turns it off. mira asks a clear question about the stones."
        ]
        self.sft_examples = default_sft_examples()

        # Build vocabulary
        all_text = "".join(self.paragraphs + self.validation_paragraphs)
        for ex in self.sft_examples:
            all_text += ex["prompt"] + ex["answer"]
        all_text += "<pad><eos>\n !"

        chars = sorted(list(set(all_text)))
        special_tokens = ["<pad>", "<eos>", "\n"]
        for st in special_tokens:
            if st in chars:
                chars.remove(st)
        self.vocab = special_tokens + chars
        self.char2id = {c: i for i, c in enumerate(self.vocab)}
        self.id2char = {i: c for i, c in enumerate(self.vocab)}

        self.model = TinyGlassboxTransformer(vocab_size=len(self.vocab), width=8, heads=2, mlp_width=16, layers=2)
        
        # Checkpoints state
        self.snapshots = {}
        self.histories = {"base": [], "sft": []}
        
    def encode(self, text):
        return [self.char2id.get(c, 0) for c in text]

    def decode(self, ids):
        return "".join([self.id2char.get(i, "") for i in ids])

    def generate(self, prompt, max_new_tokens=20, temperature=1.0):
        self.model.eval()
        ids = self.encode(prompt)
        input_tensor = torch.tensor([ids], dtype=torch.long)
        
        generated = list(ids)
        with torch.no_grad():
            for _ in range(max_new_tokens):
                curr = torch.tensor([generated[-32:]], dtype=torch.long)
                logits = self.model(curr)
                next_logits = logits[0, -1, :] / max(temperature, 1e-4)
                probs = F.softmax(next_logits, dim=-1)
                next_id = torch.argmax(probs).item()
                generated.append(next_id)
                if self.id2char.get(next_id) == "<eos>":
                    break
                    
        return self.decode(generated[len(ids):])

    def train_full_trace(self, base_steps=200, sft_steps=100):
        torch.manual_seed(self.seed)
        self.model = TinyGlassboxTransformer(vocab_size=len(self.vocab), width=8, heads=2, mlp_width=16, layers=2)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)

        # Fixed inspection batch prompt: "pip has " -> target "ip has t"
        input_text = "pip has "
        input_ids = self.encode(input_text)
        target_text = "ip has t"
        target_ids = self.encode(target_text)

        # Record Initial Snapshot
        self.model.eval()
        logits, init_tensors = self.model(torch.tensor([input_ids]), capture_tensors=True)
        loss_fn = nn.CrossEntropyLoss()
        init_loss = loss_fn(logits.view(-1, len(self.vocab)), torch.tensor(target_ids)).item()
        
        # SGD update demonstration on Layer 1 mlp_up
        param = self.model.block0_mlp_up.weight
        before_w = param.detach().clone().tolist()
        logits_demo = self.model(torch.tensor([input_ids]))
        loss_demo = loss_fn(logits_demo.view(-1, len(self.vocab)), torch.tensor(target_ids))
        loss_demo.backward()
        
        lr_demo = 0.05
        grad_w = param.grad.detach().clone().tolist()
        delta_w = (-lr_demo * param.grad).detach().clone().tolist()
        
        # Finite difference check at [5, 5]
        row_i, col_j = 5, 5
        eps = 1e-4
        with torch.no_grad():
            param[row_i, col_j] += eps
            l_plus = loss_fn(self.model(torch.tensor([input_ids])).view(-1, len(self.vocab)), torch.tensor(target_ids)).item()
            param[row_i, col_j] -= 2 * eps
            l_minus = loss_fn(self.model(torch.tensor([input_ids])).view(-1, len(self.vocab)), torch.tensor(target_ids)).item()
            param[row_i, col_j] += eps
            
            # Apply manual SGD step to get after_w
            param.data += -lr_demo * param.grad.data
            loss_after_demo = loss_fn(self.model(torch.tensor([input_ids])).view(-1, len(self.vocab)), torch.tensor(target_ids)).item()

        after_w = param.detach().clone().tolist()
        fd_grad = (l_plus - l_minus) / (2 * eps)
        analytic_grad = grad_w[row_i][col_j]
        abs_err = abs(fd_grad - analytic_grad)

        optimizer.zero_grad()

        self.snapshots["initial"] = {
            "loss": init_loss,
            "tensors": init_tensors,
            "weights": {
                "up": before_w,
                "up_bias": self.model.block0_mlp_up.bias.detach().tolist(),
                "down": self.model.block0_mlp_down.weight.detach().tolist(),
                "down_bias": self.model.block0_mlp_down.bias.detach().tolist()
            },
            "sgd": {
                "lr": lr_demo,
                "row": row_i,
                "column": col_j,
                "before": before_w,
                "gradient": grad_w,
                "delta": delta_w,
                "loss_before": loss_demo.item(),
                "after": after_w,
                "loss_after": loss_after_demo,
                "finite_difference": fd_grad,
                "analytic_gradient": analytic_grad,
                "finite_difference_abs_error": abs_err
            },
            "generation": {
                "pip has ": self.generate("pip has "),
                "user: how many red stones does pip have?\nassistant: ": self.generate("user: how many red stones does pip have?\nassistant: ")
            },
            "trace_source": "live_pytorch_engine"
        }

        # Train Base Model
        self.histories["base"] = []
        self.model.train()
        train_data = "".join(self.paragraphs)
        train_encoded = self.encode(train_data)
        
        for step in range(0, base_steps + 1):
            if step > 0:
                # Random window training
                idx = (step * 7) % max(1, len(train_encoded) - 32)
                batch_in = torch.tensor([train_encoded[idx:idx+32]], dtype=torch.long)
                batch_tgt = torch.tensor([train_encoded[idx+1:idx+33]], dtype=torch.long)
                
                optimizer.zero_grad()
                out_logits = self.model(batch_in)
                loss = loss_fn(out_logits.view(-1, len(self.vocab)), batch_tgt.view(-1))
                loss.backward()
                optimizer.step()

            if step % max(1, (base_steps // 10)) == 0 or step == base_steps:
                self.model.eval()
                with torch.no_grad():
                    b_in = torch.tensor([train_encoded[:32]], dtype=torch.long)
                    b_tgt = torch.tensor([train_encoded[1:33]], dtype=torch.long)
                    l_tr = loss_fn(self.model(b_in).view(-1, len(self.vocab)), b_tgt.view(-1)).item()
                    
                    val_enc = self.encode("".join(self.validation_paragraphs))[:32]
                    v_in = torch.tensor([val_enc[:-1]], dtype=torch.long)
                    v_tgt = torch.tensor([val_enc[1:]], dtype=torch.long)
                    l_val = loss_fn(self.model(v_in).view(-1, len(self.vocab)), v_tgt.view(-1)).item()
                    
                    self.histories["base"].append({"step": step, "train": l_tr, "validation": l_val})
                self.model.train()

        # Record Base Snapshot
        self.model.eval()
        base_logits, base_tensors = self.model(torch.tensor([input_ids]), capture_tensors=True)
        base_loss = loss_fn(base_logits.view(-1, len(self.vocab)), torch.tensor(target_ids)).item()
        
        self.snapshots["base"] = {
            "loss": base_loss,
            "tensors": base_tensors,
            "weights": {
                "up": self.model.block0_mlp_up.weight.detach().tolist(),
                "up_bias": self.model.block0_mlp_up.bias.detach().tolist(),
                "down": self.model.block0_mlp_down.weight.detach().tolist(),
                "down_bias": self.model.block0_mlp_down.bias.detach().tolist()
            },
            "sgd": self.snapshots["initial"]["sgd"],
            "generation": {
                "pip has ": self.generate("pip has "),
                "user: how many red stones does pip have?\nassistant: ": self.generate("user: how many red stones does pip have?\nassistant: ")
            },
            "trace_source": "live_pytorch_engine"
        }

        # SFT Training (Answer-only loss masking with -100)
        self.histories["sft"] = []
        self.model.train()
        sft_opt = torch.optim.AdamW(self.model.parameters(), lr=0.005)

        for step in range(0, sft_steps + 1):
            if step > 0:
                ex = self.sft_examples[step % len(self.sft_examples)]
                p_ids = self.encode("user: " + ex["prompt"] + "\nassistant: ")
                a_ids = self.encode(ex["answer"])
                full_ids = p_ids + a_ids
                
                # -100 for prompt tokens
                tgt_ids = [-100] * len(p_ids) + a_ids
                
                s_in = torch.tensor([full_ids[:-1]], dtype=torch.long)
                s_tgt = torch.tensor([tgt_ids[1:]], dtype=torch.long)

                sft_opt.zero_grad()
                out_logits = self.model(s_in)
                loss = loss_fn(out_logits.view(-1, len(self.vocab)), s_tgt.view(-1))
                loss.backward()
                sft_opt.step()

            if step % max(1, (sft_steps // 10)) == 0 or step == sft_steps:
                self.model.eval()
                with torch.no_grad():
                    ex = self.sft_examples[0]
                    p_ids = self.encode("user: " + ex["prompt"] + "\nassistant: ")
                    a_ids = self.encode(ex["answer"])
                    full_ids = p_ids + a_ids
                    tgt_ids = [-100] * len(p_ids) + a_ids
                    
                    s_in = torch.tensor([full_ids[:-1]], dtype=torch.long)
                    s_tgt = torch.tensor([tgt_ids[1:]], dtype=torch.long)
                    l_sft = loss_fn(self.model(s_in).view(-1, len(self.vocab)), s_tgt.view(-1)).item()
                    
                    val_enc = self.encode("".join(self.validation_paragraphs))[:32]
                    v_in = torch.tensor([val_enc[:-1]], dtype=torch.long)
                    v_tgt = torch.tensor([val_enc[1:]], dtype=torch.long)
                    l_base_val = loss_fn(self.model(v_in).view(-1, len(self.vocab)), v_tgt.view(-1)).item()

                    self.histories["sft"].append({"step": step, "train": l_sft, "base_validation": l_base_val})
                self.model.train()

        # Record SFT Snapshot
        self.model.eval()
        sft_logits, sft_tensors = self.model(torch.tensor([input_ids]), capture_tensors=True)
        sft_loss = loss_fn(sft_logits.view(-1, len(self.vocab)), torch.tensor(target_ids)).item()

        self.snapshots["sft"] = {
            "loss": sft_loss,
            "tensors": sft_tensors,
            "weights": {
                "up": self.model.block0_mlp_up.weight.detach().tolist(),
                "up_bias": self.model.block0_mlp_up.bias.detach().tolist(),
                "down": self.model.block0_mlp_down.weight.detach().tolist(),
                "down_bias": self.model.block0_mlp_down.bias.detach().tolist()
            },
            "sgd": self.snapshots["initial"]["sgd"],
            "generation": {
                "pip has ": self.generate("pip has "),
                "user: how many red stones does pip have?\nassistant: ": self.generate("user: how many red stones does pip have?\nassistant: ")
            },
            "trace_source": "live_pytorch_engine"
        }

        # Build Mask Example for SFT page
        ex_mask = self.sft_examples[0]
        p_ids = self.encode("user: " + ex_mask["prompt"] + "\nassistant: ")
        a_ids = self.encode(ex_mask["answer"])
        full_ids = p_ids + a_ids
        mask_target_ids = [-100] * (len(p_ids) - 1) + [full_ids[len(p_ids)]] + a_ids[1:] + [1] # 1 is eos
        
        corpus_hash = hashlib.sha256("".join(self.paragraphs).encode('utf-8')).hexdigest()

        return {
            "metadata": {
                "schema_version": "0.1",
                "seed": self.seed,
                "device": "cpu",
                "torch_version": torch.__version__,
                "python_version": sys.version.split()[0],
                "model": {
                    "layers": 2,
                    "width": 8,
                    "heads": 2,
                    "head_dim": 4,
                    "mlp_width": 16,
                    "context": 96,
                    "vocab_size": len(self.vocab),
                    "parameters": sum(p.numel() for p in self.model.parameters()),
                    "normalization": "pre-LayerNorm",
                    "activation": "GELU",
                    "position": "learned absolute",
                    "output_embedding_tied": True,
                    "dropout": 0
                },
                "base_steps": base_steps,
                "sft_steps": sft_steps,
                "corpus_sha256": corpus_hash,
                "training_note": "Generated live by PyTorch GlassboxEngine."
            },
            "paragraphs": self.paragraphs,
            "validation_paragraphs": self.validation_paragraphs,
            "vocab": self.vocab,
            "input": input_text,
            "tokens": list(input_text),
            "ids": input_ids,
            "targets": list(target_text),
            "target_ids": target_ids,
            "sft_examples": self.sft_examples,
            "mask_example": {
                "input_ids": full_ids,
                "target_ids": mask_target_ids
            },
            "histories": self.histories,
            "snapshots": self.snapshots
        }
