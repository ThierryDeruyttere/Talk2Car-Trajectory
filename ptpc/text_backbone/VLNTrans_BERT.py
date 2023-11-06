import torch
from torch import nn
import texar.torch as tx

# https://github.com/VegB/VLN-Transformer

class VLNTrans_BERT(nn.Module):

    def __init__(self,
                 pretrained_path="/cw/liir_code/NoCsBack/thierry/PTPC/vlntrans_encoders/vlntrans/finetuned_mask/ckpt_model_SPD_best.pth.tar"):
        super(VLNTrans_BERT, self).__init__()
        self.bert = tx.modules.BERTEncoder(pretrained_model_name='bert-base-uncased').to("cpu")

        # load weights
        checkpoint = torch.load(
            pretrained_path,
            map_location="cpu")
        self.bert.load_state_dict({k.replace("module.", ""): v for k, v in checkpoint['instr_encoder_state_dict'].items()})

        self.tokenizer = tx.data.BERTTokenizer(pretrained_model_name='bert-base-uncased',
                                      hparams={'vocab_file': "/cw/liir_code/NoCsBack/thierry/PTPC/vlntrans_encoders/vlntrans_voc.txt"})

    def forward(self, sent, device):
        input_ids = []
        segment_ids = []
        input_mask = []

        for s in sent:
            tmp = self.tokenizer.encode_text(text_a=s,text_b=None, max_seq_length=60)
            input_ids.append(tmp[0])
            segment_ids.append(tmp[1])
            input_mask.append(tmp[2])

        word_tokens, cls_token = self.bert(torch.tensor(input_ids, device=device))

        return cls_token

def main():
    bert = VLNTrans_BERT()
    bert.encode_sentence(["Take a next turn left", "Drop me off near the guy in the white"], "cpu")

if __name__ == '__main__':
    main()

