import os
from collections import Counter, OrderedDict

import torch



# 这是词典类
class Vocab(object):
    def __init__(self, alinlen=3000,special=[], min_freq=10, max_size=100000, lower_case=True,
                 delimiter=None, vocab_file=None):
        '''
        code
        :param special:
        :param min_freq:
        :param max_size:
        :param lower_case:
        :param delimiter:
        :param vocab_file:
        '''
        self.counter = Counter()
        self.special = special
        self.min_freq = min_freq
        self.max_size = max_size
        self.lower_case = lower_case
        self.delimiter = delimiter
        self.vocab_file = vocab_file
        self.alinlen = alinlen


    # 许海明
    def tokenize(self, line):
        line = line.strip()
        # convert to lower case
        if self.lower_case:
            line = line.lower()

        # empty delimiter '' will evaluate False
        if self.delimiter == '':
            symbols = line
        else:
            symbols = line.split(self.delimiter)


        if len(symbols) > self.alinlen-2:
            symbols = symbols[:self.alinlen-2]

        symbols = ['<s>'] + symbols + ['</s>']

        len_pre = len(symbols)
        if len_pre == self.alinlen:
            return symbols
        else:
            assert len_pre<self.alinlen
            new_symbols=['<pad>']*(self.alinlen-len_pre)
            new_symbols.extend(symbols)
            assert  len(new_symbols)==self.alinlen
            return new_symbols



    '''
    统计单词
    并返回分好词的数据 每一行元
    '''
    def count_file(self, path, verbose=False):
        if verbose:
            print('counting file {} ...'.format(path))
        assert os.path.exists(path)

        sents = []
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if line is None or line=="":
                    continue
                if verbose and idx > 0 and idx % 500000 == 0:
                    print('    line {}'.format(idx))
                symbols = self.tokenize(line)
                self.counter.update(symbols)
                sents.append(symbols)

        return sents

    def count_sents(self, sents, verbose=False):
        """
            sents : a list of sentences, each a list of tokenized symbols
        """
        if verbose: print('counting {} sents ...'.format(len(sents)))
        for idx, symbols in enumerate(sents):
            if verbose and idx > 0 and idx % 500000 == 0:
                print('    line {}'.format(idx))
            self.counter.update(symbols)

    # 许海明
    def _build_from_file(self, vocab_file):
        self.idx2sym = []
        self.sym2idx = OrderedDict()

        with open(vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                symb = line.strip().split()[0]
                self.add_symbol(symb)
        self.unk_idx = self.sym2idx['<unk>']


    # 许海明
    def build_vocab(self):
        if self.vocab_file:
            print('building vocab from {}'.format(self.vocab_file))
            self._build_from_file(self.vocab_file)
            print('final vocab size {}'.format(len(self)))
        else:
            print('building vocab with min_freq={}, max_size={}'.format(
                self.min_freq, self.max_size))
            self.idx2sym = []
            self.sym2idx = OrderedDict()

            for sym in self.special:
                self.add_special(sym)

            for sym, cnt in self.counter.most_common(self.max_size):
                if cnt < self.min_freq:
                    break
                self.add_symbol(sym)

            print('final vocab size {} from {} unique tokens'.format(
                len(self), len(self.counter)))


    #  许海明  对一个文件进行编码
    '''
    tensor([1, 2, 3, 1, 2, 3, 1, 2, 3])
    
    a=torch.LongTensor([1,2,3])
    a1=torch.LongTensor([1,2,3])
    a2=torch.LongTensor([1,2,3])

    print(torch.cat([a,a1,a2]))
    '''
    def encode_file(self, path, ordered=False, verbose=False):
        '''
                           ordered=True
        :param path:
        :param ordered:
        :param verbose:
        :param add_eos:
        :param add_double_eos:
        :return:
        '''
        if verbose:
            print('encoding file {} ...'.format(path))
        assert os.path.exists(path)
        encoded = []
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if line is None or line == "":
                    continue
                if verbose and idx > 0 and idx % 500000 == 0:
                    print('    line {}'.format(idx))
                symbols = self.tokenize(line)  # code 每一行加上结束 <eos>

                encoded.append(self.convert_to_tensor(symbols))

        if ordered:
            encoded = torch.cat(encoded)

        return encoded

    # 许海明 这是对多个句子进行编码
    def encode_sents(self, sents, ordered=False, verbose=False):
        if verbose:
            print('encoding {} sents ...'.format(len(sents)))
        encoded = []
        for idx, symbols in enumerate(sents):
            if verbose and idx > 0 and idx % 500000 == 0:
                print('    line {}'.format(idx))
            encoded.append(self.convert_to_tensor(symbols))

        if ordered:
            encoded = torch.cat(encoded)

        return encoded

    # 许海明
    def add_special(self, sym):
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1
            setattr(self, '{}_idx'.format(sym.strip('<>')), self.sym2idx[sym])

    # 许海明
    def add_symbol(self, sym):
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1


    def save_symbol(self, vocabfile):
        with open(vocabfile, mode="w", encoding="utf-8") as fw:
            for word in self.sym2idx:
                fw.write(word+"\n")





    def get_sym(self, idx):
        assert 0 <= idx < len(self), 'Index {} out of range'.format(idx)
        return self.idx2sym[idx]

    def get_idx(self, sym):
        if sym in self.sym2idx:
            return self.sym2idx[sym]
        else:
            # print('encounter unk {}'.format(sym))
            assert '<eos>' not in sym
            assert hasattr(self, 'unk_idx')
            return self.sym2idx.get(sym, self.unk_idx)

    def get_symbols(self, indices):
        return [self.get_sym(idx) for idx in indices]

    def get_indices(self, symbols):
        return [self.get_idx(sym) for sym in symbols]

    # 许海明
    def convert_to_tensor(self, symbols):
        return torch.LongTensor(self.get_indices(symbols))

    def convert_to_sent(self, indices, exclude=None):
        if exclude is None:
            return ' '.join([self.get_sym(idx) for idx in indices])
        else:
            return ' '.join([self.get_sym(idx) for idx in indices if idx not in exclude])

    def __len__(self):
        return len(self.idx2sym)
