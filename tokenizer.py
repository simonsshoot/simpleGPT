import json

class Tokenizer():
    def __init__(self, vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as r:
            vocab = r.read()
            if not vocab:
                raise Exception("词表读取为空！")
        vocab = json.loads(vocab)
        self.word2id = vocab["word2id"]
        self.id2word = vocab["id2word"]
        self.pad_token = self.word2id["<pad>"]
        self.unk_token = self.word2id["<unk>"]
        self.sep_token = self.word2id["<sep>"]

    def encode(self, text, text1=None, max_length=128, pad_to_max_length=False):
        tokens = [self.word2id[word] if word in self.word2id else self.unk_token for word in text]
        tokens.append(self.sep_token)
        if text1:
            tokens.extend([self.word2id[word] if word in self.word2id else self.unk_token for word in text1])
            tokens.append(self.sep_token)
        att_mask = [1] * len(tokens)
        if pad_to_max_length:
            if len(tokens) > max_length:
                tokens = tokens[0:max_length]
                att_mask = att_mask[0:max_length]
            elif len(tokens) < max_length:
                tokens.extend([self.pad_token] * (max_length - len(tokens)))
                att_mask.extend([0] * (max_length - len(att_mask)))
        return tokens, att_mask

    def decode(self, token):
        if type(token) is tuple or type(token) is list:
            return [self.id2word[n] for n in token]
        else:
            return self.id2word[token]

    def get_vocab_size(self):
        return len(self.id2word)