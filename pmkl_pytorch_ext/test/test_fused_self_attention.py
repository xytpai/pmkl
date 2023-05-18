import torch
import torch.nn.functional as F
from pmkl_function import fused_self_attention


def self_attention_cpu(q, k, v, is_causal):
    return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=is_causal)


for batch_size in [2, 3]:
    for num_heads in [16]:
        for seq_len in [128]:
            for hidden_size in [64, 128]:
                for is_causal in [False, True]:
                    q_ref = torch.randn(batch_size, num_heads, seq_len, hidden_size)
                    q_ref.requires_grad = True
                    k_ref = torch.randn(batch_size, num_heads, seq_len, hidden_size)
                    k_ref.requires_grad = True
                    v_ref = torch.randn(batch_size, num_heads, seq_len, hidden_size)
                    v_ref.requires_grad = True
                    out_ref = self_attention_cpu(q_ref, k_ref, v_ref, is_causal)
                    loss = out_ref.sum()
                    loss.backward()

                    q = q_ref.detach().clone()
                    q.requires_grad = True
                    k = k_ref.detach().clone()
                    k.requires_grad = True
                    v = v_ref.detach().clone()
                    v.requires_grad = True
                    out, _, _ = fused_self_attention(q, k, v, is_causal, True)
                    loss = out.sum()
                    loss.backward()

                    diff_out = (out_ref - out).abs().max().item()                
                    diff_dq = (q_ref.grad - q.grad).abs().max().item()
                    diff_dk = (k_ref.grad - k.grad).abs().max().item()    
                    diff_dv = (v_ref.grad - v.grad).abs().max().item()

                    diff = {'diff_out':diff_out, 'diff_dq':diff_dq, 'diff_dk':diff_dk, 'diff_dv':diff_dv}
                    print(diff)
