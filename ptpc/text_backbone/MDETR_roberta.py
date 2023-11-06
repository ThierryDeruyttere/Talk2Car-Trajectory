import os

os.environ['TRANSFORMERS_CACHE'] = "../pretrained/huggingface"
from transformers import RobertaModel, RobertaTokenizerFast
import torch.nn as nn
import torch

class MDETR_Roberta(nn.Module):
    """

    Available models:
        Pre-trained MDETR model (ENB5): https://zenodo.org/record/4721981/files/pretrained_EB5_checkpoint.pth?download=1

    """
    def __init__(self, pretrained_path, output_dim, projection_layer="linear", text_encoder_type="roberta-base",
                 freeze=False, dropout=0):
        super(MDETR_Roberta, self).__init__()

        self.tokenizer = RobertaTokenizerFast.from_pretrained(text_encoder_type)
        self.text_encoder = RobertaModel.from_pretrained(text_encoder_type)

        state = torch.load(pretrained_path, map_location="cpu")
        filtered_key_values = {k.replace("transformer.text_encoder.", ""): v for k, v in state["model"].items() if
                               "text_encoder" in k}
        self.text_encoder.load_state_dict(filtered_key_values)

        if freeze:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)

        if projection_layer == 'linear':
            self.linear_layer = nn.Linear(self.text_encoder.embeddings.token_type_embeddings.embedding_dim, output_dim)

        elif projection_layer == 'mlp':
            self.linear_layer = nn.Sequential(
                                    nn.Linear(self.text_encoder.embeddings.token_type_embeddings.embedding_dim, output_dim),
                                    nn.ReLU(), nn.Linear(output_dim, output_dim))

        self.sentence_token = nn.Embedding(num_embeddings=1, embedding_dim=output_dim)
        self.sentence_combiner = nn.MultiheadAttention(num_heads=8, dropout=dropout, embed_dim=output_dim, batch_first=True)

    def forward(self, input_seq_raw, device, embedding_type="sentence_embedding"):

        tokenized = self.tokenizer.batch_encode_plus(input_seq_raw, padding="longest",
                                                     return_tensors="pt").to(device)
        encoded_text = self.text_encoder(**tokenized)
        tokens = self.linear_layer(encoded_text.last_hidden_state)
        sentence_emb = self.sentence_token(torch.zeros(len(input_seq_raw), device=device).long()).unsqueeze(1)

        sentence_emb, attns = self.sentence_combiner(query=sentence_emb, key=tokens, value=tokens)

        return sentence_emb.squeeze(1)

def main():
    model = MDETR_Roberta(pretrained_path="../pretrained/MDETR_pretrained_EB5_checkpoint.pth", output_dim=512)
    model.forward(["Take the next turn left.", "Drop me off near my dad"], "cpu")


if __name__ == "__main__":
    main()