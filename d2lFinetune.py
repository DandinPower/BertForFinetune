import json
import multiprocessing
import os
import torch
from torch import nn
import pandas as pd # 引用套件並縮寫為 pd  
from models.BertModel import *

def load_pretrained_model(pretrained_dir, num_hiddens, ffn_num_hiddens,num_heads, num_layers, dropout, max_len, devices):
    data_dir = pretrained_dir
    data_dir = 'data/bert.small.torch/'
    #讀取vocab
    vocab = Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(data_dir,
        'vocab.json')))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(
        vocab.idx_to_token)}
    #加載預訓練模組
    bert = BERTModel(vocab_size = len(vocab), num_hiddens = num_hiddens,ffn_num_hiddens = ffn_num_hiddens, num_heads = num_heads,num_layers = num_layers, dropout = dropout, max_len = max_len,norm_shape = [256],ffn_num_input = 256)
    bert.load_state_dict(torch.load(os.path.join(data_dir,
                                                 'pretrained.params')))
    return bert, vocab

class BERTClassifier(nn.Module):
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        self.encoder = bert.encoder
        self.hidden = bert.hidden
        self.output = nn.Linear(256, 3)

    def forward(self, inputs):
        tokens_X, segments_X, valid_lens_x = inputs
        encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)
        return self.output(self.hidden(encoded_X[:, 0, :]))

class YelpDataset(torch.utils.data.Dataset):
    def __init__(self, datasetPath, max_len, vocab,train,splitRate):
        self.max_len = max_len
        self.labels = []
        self.vocab = vocab
        self.all_tokens_ids = []
        self.all_segments = []
        self.valid_lens = []
        self.path = datasetPath
        self.train = train
        self.splitRate = splitRate
        self.Preprocess()

    #將資料做預處理
    def Preprocess(self):
        texts,self.labels = self.ReadDataset()
        texts = [self.TruncatePairOfTokens(text)for text in texts]
        newTexts,newSegments = [],[]
        for text in texts:
            tokens,segments = self.GetTokensAndSegments(text)
            newTexts.append(tokens)
            newSegments.append(segments)
        self.PadBertInput(newTexts, newSegments)

    #讀取dataset
    def ReadDataset(self):
        df = pd.read_csv(self.path)  
        #print(df.Stars.values)
        labels = []
        for item in df.Stars.values:
            if item >= 4:
                labels.append(1)
            else:
                labels.append(0)
        #print(labels)
        texts = [line.strip().lower().split(' ')for line in df.Text.values]
        trainLen = int(len(df.Text.values) * self.splitRate) 
        if (self.train):
            texts = texts[0:trainLen]
            labels = labels[0:trainLen]
        else:
            texts = texts[trainLen:]
            labels = labels[trainLen:]
        return texts,labels

    def GetTokensAndSegments(self,tokensA, tokensB=None):
        tokens = ['<cls>'] + tokensA + ['<sep>']
        # 0 and 1 are marking segment A and B, respectively
        segments = [0] * (len(tokensA) + 2)
        if tokensB is not None:
            tokens += tokensB + ['<sep>']
            segments += [1] * (len(tokensB) + 1)
        return tokens, segments

    #給<CLS>,<SEP>,<SEP>保留位置
    def TruncatePairOfTokens(self, tokens):   
        while len(tokens) > self.max_len - 3:
            tokens.pop()
        return tokens

    #進行padding
    def PadBertInput(self,texts,segments):
        texts = self.vocab[texts]
        for (text,segment) in zip(texts,segments):
            self.all_tokens_ids.append(torch.tensor(text + [self.vocab['<pad>']] * (self.max_len - len(text)), dtype=torch.long))
            self.all_segments.append(torch.tensor(segment + [0] * (self.max_len - len(segment)), dtype=torch.long))
            #valid_lens不包括<pad>
            self.valid_lens.append(torch.tensor(len(text), dtype=torch.float32))

    def __getitem__(self, idx):
        return (self.all_tokens_ids[idx], self.all_segments[idx],
                self.valid_lens[idx]), self.labels[idx]

    def __len__(self):
        return len(self.all_tokens_ids)

def main():
    devices = d2l.try_all_gpus()
    batch_size, max_len= 5, 512
    train_test_rate = 0.9
    lr, num_epochs = 1e-4, 5
    print("Loading Pretraining Model...")
    bert, vocab = load_pretrained_model('bert.small', num_hiddens=256, ffn_num_hiddens=512, num_heads=4,num_layers=2, dropout=0.1, max_len=512, devices=devices)
    print("Loading Train Dataset...")
    trainDataset = YelpDataset('data/reviews_small.csv',max_len,vocab,True,train_test_rate)
    train_iter = torch.utils.data.DataLoader(trainDataset, batch_size, shuffle=True)
    print("Loading Test Dataset...")
    testDataset = YelpDataset('dataset/reviews_small.csv',max_len,vocab,False,train_test_rate)
    test_iter = torch.utils.data.DataLoader(testDataset, batch_size)
    net = BERTClassifier(bert)
    print("training...")
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(reduction='none')
    train_ch13(net, train_iter, test_iter, loss, trainer, 10,
        devices) 
    
if __name__ == "__main__":
    main()
