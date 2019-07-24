import torch
word_emb=torch.randn(5,10)

dec_attn_mask = torch.triu(
    word_emb.new_ones(5,10), diagonal=5+1)
print(dec_attn_mask)