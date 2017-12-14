import os
import torch
import numpy as np

from collections import Counter, OrderedDict


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class ByteCorpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        assert os.path.exists(path)
        # 256 bytes, that's all there is...
        with open(path, 'r', encoding='utf-8') as f:
            tokens = 0
            for line in f:
                line += '\n'
                chars = list(bytes(line, 'utf-8'))
                tokens += len(chars)
                for c in chars:
                    self.dictionary.add_word(c)

        # tokenize file content
        with open(path, 'r', encoding='utf-8') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                line += '\n'
                chars = list(bytes(line, 'utf-8'))
                for c in chars:
                    ids[token] = self.dictionary.word2idx[c]
                    token += 1

        return ids


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding='utf-8') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding='utf-8') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids


class SentCorpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding='utf-8') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        sents = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line:
                    continue
                words = line.split() + ['<eos>']
                sent = torch.LongTensor(len(words))
                for i, word in enumerate(words):
                    sent[i] = self.dictionary.word2idx[word]
                sents.append(sent)

        return sents


class BatchSentLoader(object):
    def __init__(self, sents, batch_size, pad_id=0, cuda=False, volatile=False):
        self.sents = sents
        self.batch_size = batch_size
        self.sort_sents = sorted(sents, key=lambda x: x.size(0))
        self.cuda = cuda
        self.volatile = volatile
        self.pad_id = pad_id

    def __next__(self):
        if self.idx >= len(self.sort_sents):
            raise StopIteration

        batch_size = min(self.batch_size, len(self.sort_sents)-self.idx)
        batch = self.sort_sents[self.idx:self.idx+batch_size]
        max_len = max([s.size(0) for s in batch])
        tensor = torch.LongTensor(max_len, batch_size).fill_(self.pad_id)
        for i in range(len(batch)):
            s = batch[i]
            tensor[:s.size(0),i].copy_(s)
        if self.cuda:
            tensor = tensor.cuda()

        self.idx += batch_size

        return tensor
    
    next = __next__

    def __iter__(self):
        self.idx = 0
        return self


def load_embeddings_txt(path, max=-1):
    train = OrderedDict()
    with open(path, 'rt', encoding='utf-8') as ef:
        for i, line in enumerate(ef):
            tokens = line.split()
            word = tokens[0]
            # 1 x EmbSize
            vector = np.array(tokens[1:], dtype=np.float32)[None, :]
            #vector = torch.from_numpy(vector)
            #vector = torch.autograd.Variable(vector, requires_grad=False)
            train[word] = vector
    words = list(train.keys())
    #embeddings = torch.cat(list(train.values()), 0)
    embeddings = np.concatenate(list(train.values()), 0)
    return words, embeddings


def check_compatibility(corpus, loaded_words):
    corpus_words = corpus.dictionary.idx2word
    print(loaded_words[:10])
    print(corpus_words[:10])
    assert loaded_words == corpus_words


def dump_embeddings(emb_file, emb_table, idx2word):
    with open(emb_file, 'wt') as f:
        for i, word in enumerate(idx2word):
            # print(word)
            emb_comp = emb_table[i].cpu().numpy().tolist()
            for x in emb_comp:
                word += ' ' + str(x)
            f.write(word + '\n')

if __name__ == '__main__':
    corpus = SentCorpus('../penn')
    loader = BatchSentLoader(corpus.test, 10)
    for i, d in enumerate(loader):
        print(i, d.size())
