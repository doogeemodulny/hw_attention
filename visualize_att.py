import os
import torch
import matplotlib.pyplot as plt
from model import convert_batch


class AttentionVisualizer:
    def __init__(self, model, word_field, save_dir="attention_mech_visualization", blocks_count=4):
        self.model = model
        self.word_field = word_field
        self.save_dir = save_dir
        self.layer_count = blocks_count
        
    def extract_attention(self, ex_v, layer, head, mode):
        if mode == "encoder":
            attn = self.model.encoder._blocks[layer]._self_attn._attn_probs
        elif mode == "decoder":
            attn = self.model.decoder._blocks[layer]._self_attn._attn_probs
        return attn[0, head].data.cpu().numpy()
    
    def visualize_layer(self, layer, ex_v, ex_i, example_save_dir, mode):
        words = [self.word_field.vocab.itos[i] for i in ex_v.source]    
        attn = self.model.encoder._blocks[layer]._self_attn._attn_probs
        n_heads = attn.shape[1]
        for h in range(n_heads):
            plt.figure(figsize=(20, 20))
            plt.matshow(self.extract_attention(ex_v, layer, h, mode), cmap='Oranges', fignum=1)
            plt.xticks(ticks=range(len(words)), labels=words, rotation=90)
            plt.yticks(ticks=range(len(words)), labels=words)
            plt.colorbar()
            plt.savefig(os.path.join(example_save_dir, f"layer_{layer+1}_head_{h+1}.jpg"))
            plt.close()

    def attention_mech_visualization(self, ex_v, ex_i, mode="encoder"):
        example_save_dir = os.path.join(self.save_dir, f"example_{ex_i+1}")
        os.makedirs(example_save_dir, exist_ok=True)
        with torch.no_grad():
            _ = self.model.encoder(*convert_batch(ex_v)[::2])
        for layer in range(self.layer_count):
            self.visualize_layer(layer, ex_v, ex_i, example_save_dir, mode)


