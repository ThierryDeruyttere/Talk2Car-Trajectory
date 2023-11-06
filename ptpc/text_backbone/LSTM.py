# Code based/copied from https://github.com/VegB/VLN-Transformer
import numpy as np
import json, re, string
import sys
import torch
from torch import nn

padding_idx = 0

def read_vocab(path):
    with open(path, encoding="utf-8") as f:
        vocab = [word.strip() for word in f.readlines()]
    return vocab

class Tokenizer(object):
    """ Class to tokenize and encode a sentence. """
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')  # Split on any non-alphanumeric character

    def __init__(self, remove_punctuation=False, reversed=True, vocab=None, encoding_length=20):
        self.remove_punctuation = remove_punctuation
        self.reversed = reversed
        self.encoding_length = encoding_length
        self.vocab = vocab
        self.table = str.maketrans({key: None for key in string.punctuation})
        self.word_to_index = {}
        if vocab:
            for i, word in enumerate(vocab):
                self.word_to_index[word] = i

    def split_sentence(self, sentence):
        """ Break sentence into a list of words and punctuation """
        toks = []
        for word in [s.strip().lower() for s in self.SENTENCE_SPLIT_REGEX.split(sentence.strip()) if
                     len(s.strip()) > 0]:
            # Break up any words containing punctuation only, e.g. '!?', unless it is multiple full stops e.g. '..'
            if all(c in string.punctuation for c in word) and not all(c in '.' for c in word):
                toks += list(word)
            else:
                toks.append(word)
        return toks

    def encode_sentence(self, sentence):
        if len(self.word_to_index) == 0:
            sys.exit('Tokenizer has no vocab')
        encoding = []

        splited = self.split_sentence(sentence)
        if self.reversed:
            splited = splited[::-1]

        if self.remove_punctuation:
            splited = [word for word in splited if word not in string.punctuation]

        for word in splited:  # reverse input sentences
            if word in self.word_to_index:
                encoding.append(self.word_to_index[word])
            else:
                encoding.append(self.word_to_index['<UNK>'])

        encoding.append(self.word_to_index['<EOS>'])
        encoding.insert(0, self.word_to_index['<START>'])

        if len(encoding) < self.encoding_length:
            encoding += [self.word_to_index['<PAD>']] * (self.encoding_length - len(encoding))
        return np.array(encoding[:self.encoding_length]), min(len(splited)+2, self.encoding_length)

    def encode_instructions(self, instructions):
        rst = []
        for sent in instructions.strip().split('. '):
            rst.append(self.encode_sentence(sent))
        return rst

class CustomRNN(nn.Module):
    """
    A module that runs multiple steps of RNN cell
    With this module, you can use mask for variable-length input
    """
    def __init__(self, cell_class, input_size, hidden_size, num_layers=1,
                 use_bias=True, batch_first=False, dropout=0, **kwargs):
        super(CustomRNN, self).__init__()
        self.cell_class = cell_class
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_bias = use_bias
        self.batch_first = batch_first

        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            cell = cell_class(input_size=layer_input_size,
                              hidden_size=hidden_size,
                              **kwargs)
            setattr(self, 'cell_{}'.format(layer), cell)
        self.dropout_layer = nn.Dropout(dropout)
        self.reset_parameters()

    def get_cell(self, layer):
        return getattr(self, 'cell_{}'.format(layer))

    def reset_parameters(self):
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            cell.reset_parameters()

    @staticmethod
    def _forward_rnn(cell, input_, mask, hx):
        max_time = input_.size(0)
        output = []
        for time in range(max_time):
            h_next, c_next = cell(input_[time], hx=hx)
            mask_ = mask[time].unsqueeze(1).expand_as(h_next)
            h_next = h_next*mask_ + hx[0]*(1 - mask_)
            c_next = c_next*mask_ + hx[1]*(1 - mask_)
            hx_next = (h_next, c_next)
            output.append(h_next)
            hx = hx_next
        output = torch.stack(output, 0)
        return output, hx

    def forward(self, input_, mask, hx=None):
        if self.batch_first:
            input_ = input_.transpose(0, 1)
            mask = mask.transpose(0, 1)
        max_time, batch_size, _ = input_.size()

        if hx is None:
            hx = input_.new(batch_size, self.hidden_size).zero_()
            hx = (hx, hx)
        h_n = []
        c_n = []
        layer_output = None
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            layer_output, (layer_h_n, layer_c_n) = CustomRNN._forward_rnn(
                cell=cell, input_=input_, mask=mask, hx=hx)
            input_ = self.dropout_layer(layer_output)
            h_n.append(layer_h_n)
            c_n.append(layer_c_n)
        output = layer_output
        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)
        if self.batch_first:
            output = output.transpose(0, 1)
        return output, (h_n, c_n)

class Embed_RNN(nn.Module):
    def __init__(self, vocab_size):
        super(Embed_RNN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding = nn.Embedding(vocab_size, 32, padding_idx)
        self.rnn_kwargs = {'cell_class': nn.LSTMCell,
                           'input_size': 32,
                           'hidden_size': 256,
                           'num_layers': 1,
                           'batch_first': True,
                           'dropout': 0,
                           }
        self.rnn = CustomRNN(**self.rnn_kwargs)

    def create_mask(self, batchsize, max_length, length, device):
        """Given the length create a mask given a padded tensor"""
        tensor_mask = torch.zeros(batchsize, max_length)
        for idx, row in enumerate(tensor_mask):
            row[:length[idx]] = 1
        return tensor_mask.to(device)

    def forward(self, x, lengths):
        x = self.embedding(x)  # [batchsize, max_length, 32]
        embeds_mask = self.create_mask(x.size(0), x.size(1), lengths, x.device)
        x, _ = self.rnn(x, mask=embeds_mask)  # [batch_size, max_length, 256]
        x = x[:, -1, :]  # [batch_size, 256]
        return x.unsqueeze(1)  # [batch_size, 1, 256]


class LSTM(nn.Module):

    def __init__(self, hidden_size=768, pretrained=True):
        super(LSTM, self).__init__()
        vocab_file = "/cw/liir_code/NoCsBack/thierry/PTPC/vlntrans_encoders/nobert_vocab.txt"
        vocab = read_vocab(vocab_file)

        self.tokenizer = Tokenizer(vocab=vocab, encoding_length=60)
        self.rnn = Embed_RNN(len(vocab))

        if pretrained:
            weights = torch.load(
                "/cw/liir_code/NoCsBack/thierry/PTPC/vlntrans_encoders/rconcat/finetuned_mask/ckpt_model_SPD_best.pth.tar",
                map_location="cpu")
            self.rnn.load_state_dict(weights["instr_encoder_state_dict"])

        self.projection = nn.Linear(256, hidden_size)

    def forward(self, sentences, device):
        tokenized = [self.tokenizer.encode_sentence(x) for x in sentences]
        token_ixes = []
        sent_lengths = []
        for (t, l) in tokenized:
            token_ixes.append(t)
            sent_lengths.append(l)

        sent_emb = self.rnn(torch.tensor(token_ixes, device=device).long(), torch.tensor(sent_lengths,  device=device).long())
        return self.projection(sent_emb).squeeze(1)

def main2():
    lstm = LSTM()
    emb = lstm(["Take a next turn left", "Drop me off near the guy in the white"], "cpu")

def main():
    vocab_file = "/cw/liir_code/NoCsBack/thierry/PTPC/vlntrans_encoders/nobert_vocab.txt"
    vocab = read_vocab(vocab_file)

    tokenizer = Tokenizer(vocab=vocab, encoding_length=60)
    rnn = Embed_RNN(len(vocab))

    weights = torch.load("/cw/liir_code/NoCsBack/thierry/PTPC/vlntrans_encoders/rconcat/finetuned_mask/ckpt_model_SPD_best.pth.tar",
               map_location="cpu")
    rnn.load_state_dict(weights["instr_encoder_state_dict"])

    sents = ["Take a next turn left", "Drop me off near the guy in the white"]
    tokenized = [tokenizer.encode_sentence(x) for x in sents]
    token_ixes = []
    sent_lengths = []
    for (t, l) in tokenized:
        token_ixes.append(t)
        sent_lengths.append(l)

    rnn(torch.tensor(token_ixes).long(), torch.tensor(sent_lengths).long())


if __name__ == "__main__":
    main2()