import numpy as np
import time, sys, random

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import functional as F
import math
import time

from torch import FloatTensor as FT
from torch import ByteTensor as BT

TINY = 1e-11
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Word2MatEncoder(nn.Module):

    def __init__(self, n_words, word_emb_dim = 784, padding_idx = 0, w2m_type = "cbow", initialization_strategy = "identity", _lambda=0, cnmow_version=1):
        """
        TODO: Method description for w2m encoder.
        """
        super(Word2MatEncoder, self).__init__()
        self.word_emb_dim = word_emb_dim
        self.n_words = n_words
        self.w2m_type = w2m_type
        self.initialization_strategy = initialization_strategy

        # add shared recurrent weights
        if w2m_type == "cnmow":
            assert (cnmow_version >= 1) and (cnmow_version <= 9)
            self.cnmow_version = cnmow_version


        # check that the word embedding size is a square
        assert word_emb_dim == int(math.sqrt(word_emb_dim)) ** 2
        
        
        if w2m_type == "cnmow" and (self.cnmow_version==5 or self.cnmow_version==6 or self.cnmow_version==8 or self.cnmow_version==9):
            if self.cnmow_version==5 or self.cnmow_version==8:
                self.lookup_table = nn.Embedding(self.n_words + 4,
                                  self.word_emb_dim,
                                  padding_idx=padding_idx,
                                  sparse = False)

                #self.shared_weights = nn.Embedding(3, self.word_emb_dim,
                #                                   sparse = False)
                #s_weights = torch.randn((3, self.word_emb_dim))
            else:
                self.lookup_table = nn.Embedding(self.n_words + 3,
                                  self.word_emb_dim,
                                  padding_idx=padding_idx,
                                  sparse = False)
                #self.shared_weights = nn.Embedding(2, self.word_emb_dim,
                #                                   sparse = False)
                #s_weights = torch.randn((2, self.word_emb_dim))


            #self.shared_weights.weight = nn.Parameter(s_weights)

        else:
            # set up word embedding table
            self.lookup_table = nn.Embedding(self.n_words + 1,
                                          self.word_emb_dim,
                                          padding_idx=padding_idx,
                                          sparse = False)

        # type of aggregation to use to combine two word matrices
        self.w2m_type = w2m_type
        if self.w2m_type not in ["cbow", "cmow", "cnmow"]:
            raise NotImplementedError("Operator " + self.operator + " is not yet implemented.")

        # set initial weights of word embeddings depending on the initialization strategy
        ## set weights of padding symbol such that it is the neutral element with respect to the operation
        if self.w2m_type == "cmow" or self.w2m_type == "cnmow":
            neutral_element = np.reshape(np.eye(int(np.sqrt(self.word_emb_dim)), dtype=np.float32), (1, -1))
            neutral_element = torch.from_numpy(neutral_element)
        elif self.w2m_type == "cbow":
            neutral_element = np.reshape(torch.from_numpy(np.zeros((self.word_emb_dim), dtype=np.float32)), (1, -1))
        
        if w2m_type == "cnmow" and (self.cnmow_version==5 or self.cnmow_version==6 or self.cnmow_version==8 or self.cnmow_version==9):
            if self.initialization_strategy == "identity":
                if self.cnmow_version==5 or self.cnmow_version==8:
                    init_weights = np.random.normal(size = (self.n_words+3, self.word_emb_dim),
                                                             loc = 0.,
                                                             scale = 0.1
                                                             ).astype(np.float32)
                else:
                    init_weights = np.random.normal(size = (self.n_words+2, self.word_emb_dim),
                                                             loc = 0.,
                                                             scale = 0.1
                                                             ).astype(np.float32)
                for i in range(self.n_words):
                    init_weights[i, :] += np.reshape(np.eye(int(np.sqrt(self.word_emb_dim)), dtype=np.float32), (-1,))
                init_weights = torch.from_numpy(init_weights)
            
        
            else:
                raise NotImplementedError("Unknown initialization strategy " + self.initialization_strategy)

        elif self.w2m_type == "cbow":
            init_weights = self._init_normal()
            
        ## set weights of rest others depending on the initialization strategy
        else:
            if self.initialization_strategy == "identity":
                init_weights = self._init_random_identity()

            elif self.initialization_strategy == "normalized":
                ### normalized initialization by (Glorot and Bengio, 2010)
                init_weights = torch.from_numpy(np.random.uniform(size = (self.n_words,
                                                                         self.word_emb_dim),
                                                                 low = -np.sqrt(6 / (2*self.word_emb_dim)),
                                                                 high = +np.sqrt(6 / (2*self.word_emb_dim))
                                                                 ).astype(np.float32)
                                       )
            elif self.initialization_strategy == "normal":
                ### normalized with N(0,0.1), which failed in study by Yessenalina
                init_weights = torch.from_numpy(np.random.normal(size = (self.n_words,
                                                                         self.word_emb_dim),
                                                                 loc = 0.0,
                                                                 scale = 0.1
                                                                 ).astype(np.float32)
                                       )

        ## concatenate and set weights in the lookup table
        weights = torch.cat([neutral_element,
                             init_weights],
                             dim=0)
        self.lookup_table.weight = nn.Parameter(weights)
        
        
        
        # hyper parameter used for the weighted skip connection in th cnmow model
        self._lambda = _lambda

    def forward(self, sent):

        sent = self.lookup_table(sent)
        seq_length = sent.size()[1]
        matrix_dim = self._matrix_dim()

        # reshape vectors to matrices
        word_matrices = sent.view(-1, seq_length, matrix_dim, matrix_dim)

        # aggregate matrices
        if self.w2m_type == "cmow":
            cur_emb = self._continual_multiplication(word_matrices)
        elif self.w2m_type == "cbow":
            cur_emb = torch.sum(word_matrices, 1)
        elif self.w2m_type == "cnmow":
            cur_emb = self._continual_multiplication_nn(word_matrices)

        # flatten final matrix
        emb = self._flatten_matrix(cur_emb)

        return emb

    def _continual_multiplication(self, word_matrices):
        cur_emb = word_matrices[:, 0, :]
        for i in range(1, word_matrices.size()[1]):
            cur_emb = torch.bmm(cur_emb, word_matrices[:, i, :])
        return cur_emb

    def _continual_multiplication_nn(self, word_matrices):
        cur_emb = word_matrices[:, 0, :]
        for i in range(1, word_matrices.size()[1]):
            # 1- add non-linearity: including the very first word of the sentence
            if self.cnmow_version == 1:
                cur_emb = torch.bmm(F.relu(cur_emb), word_matrices[:, i, :])

            # 2- add non-linearity: excluding the very first word of the sentence
            if self.cnmow_version == 2:
                cur_emb = F.relu(torch.bmm(cur_emb, word_matrices[:, i, :]))
            # Note: adding relu might messe up being close to the identity matrix

            # 3- add weighted skip connections
            if self.cnmow_version == 3:
                cur_emb = self._lambda*cur_emb + (1-self._lambda)*torch.bmm(cur_emb , word_matrices[:, i, :])

            # 4- skip connections and non linearity
            if self.cnmow_version == 4:
                cur_emb = self._lambda*cur_emb + (1-self._lambda)*F.relu(torch.bmm(cur_emb , word_matrices[:, i, :]))

            ## TODO: compute lambda as a function of the word embeddings using sigmoid to keep it between 0 and 1
            # 5- skip connections with learned parameter without non-linearity
            if self.cnmow_version == 5:
                
                temp = torch.cat((cur_emb, word_matrices[:, i, :]),dim=1)
                #import pdb
                #pdb.set_trace()
                bias = self.lookup_table(torch.LongTensor([self.n_words+1]))
                bias = bias.view(self._matrix_dim(),self._matrix_dim()) 
                weights = self.lookup_table(torch.LongTensor([self.n_words+2,self.n_words+3]))
                weights = weights.view(self._matrix_dim(),2*self._matrix_dim()) 
                
                _lambda = torch.sigmoid(torch.matmul(weights ,temp)+bias)
                #print('grads:', self.lookup_table.weight.grad[self.n_words+2][0])
                #print('grads1:', self.shared_weights.weight.grad_fn)
                #print('grads2:', self.lookup_table.weight.grad)
                #print('grads3:', weights.grad)
                #print('grads:', _lambda.grad)
                print('weights:', weights[0][0])
                #print('look up table:', self.lookup_table.weight[0][0])
                #print('grads:', word_matrices[:,i,:].grad)
                cur_emd = _lambda*cur_emb + (1-_lambda)*torch.bmm(cur_emb , word_matrices[:, i, :])
                  
            # 6- add shared weights
            if self.cnmow_version == 6:
                bias = self.lookup_table(torch.LongTensor([self.n_words+1]))
                bias = bias.view(self._matrix_dim(),self._matrix_dim()) 
                weights = self.lookup_table(torch.LongTensor([self.n_words+2]))
                weights = weights.view(self._matrix_dim(),self._matrix_dim()) 
                
                cur_emb = torch.bmm(cur_emb, word_matrices[:, i, :])
                cur_emb = torch.matmul(weights,cur_emb) + bias
                cur_emb = F.relu(cur_emb)
                #print('grads:', self.lookup_table.weight.grad[self.n_words+2][0])
            # 7: 4 + non-linearity including first word
            if self.cnmow_version == 7:
                cur_emb = F.relu(cur_emb)
                cur_emb = self._lambda*cur_emb + (1-self._lambda)*torch.bmm(cur_emb , word_matrices[:, i, :])
                
            # 8- 5 + non-linearity including the first word
            if self.cnmow_version == 8:
                temp = torch.cat((cur_emb, word_matrices[:, i, :]),dim=1)
                _lambda = torch.sigmoid(torch.matmul(self.weights,temp) +self.bias)
                cur_emd = _lambda*F.relu(cur_emb) + (1-_lambda)*torch.bmm(F.relu(cur_emb) , word_matrices[:, i, :])
                
            # 9- 5 bmm instead of cat
            if self.cnmow_version == 9:
                #import pdb
                #pdb.set_trace()
                temp = torch.bmm(cur_emb, word_matrices[:, i, :])
                _lambda = torch.sigmoid(torch.matmul(self.weights,temp) +self.bias)
                cur_emd = _lambda*cur_emb + (1-_lambda)*torch.bmm(cur_emb , word_matrices[:, i, :])
        return cur_emb

    def _flatten_matrix(self, m):
        return m.view(-1, self.word_emb_dim)

    def _unflatten_matrix(self, m):
        return m.view(-1, self._matrix_dim(), self._matrix_dim())

    def _matrix_dim(self):
        return int(np.sqrt(self.word_emb_dim))

    def _init_random_identity(self):
        """Random normal initialization around 0., but add 1. at the diagonal"""
        init_weights = np.random.normal(size = (self.n_words, self.word_emb_dim),
                                                         loc = 0.,
                                                         scale = 0.1
                                                         ).astype(np.float32)
        for i in range(self.n_words):
            init_weights[i, :] += np.reshape(np.eye(int(np.sqrt(self.word_emb_dim)), dtype=np.float32), (-1,))
        init_weights = torch.from_numpy(init_weights)
        return init_weights

    def _init_normal(self):
        ### normal initialization around 0.
        init_weights = torch.from_numpy(np.random.normal(size = (self.n_words,
                                                                 self.word_emb_dim),
                                                         loc = 0.0,
                                                         scale = 0.1
                                                         ).astype(np.float32)
                               )
        return init_weights


class HybridEncoder(nn.Module):
    def __init__(self, cbow_encoder, cmow_encoder):
        super(HybridEncoder, self).__init__()
        self.cbow_encoder = cbow_encoder
        self.cmow_encoder = cmow_encoder

    def forward(self, sent_tuple):
        return torch.cat([self.cbow_encoder(sent_tuple), self.cmow_encoder(sent_tuple)], dim = 1)


def get_cmow_encoder(n_words, padding_idx = 0, word_emb_dim = 784, initialization_strategy = "identity"):
    encoder = Word2MatEncoder(n_words, word_emb_dim = word_emb_dim,
                              padding_idx = padding_idx, w2m_type = "cmow",
                              initialization_strategy = initialization_strategy)
    return encoder

def get_cnmow_encoder(n_words, padding_idx = 0, word_emb_dim = 784, initialization_strategy = "identity", _lambda = 0, cnmow_version = 1):
    encoder = Word2MatEncoder(n_words, word_emb_dim = word_emb_dim,
                              padding_idx = padding_idx, w2m_type = "cnmow",
                              initialization_strategy = initialization_strategy, _lambda = _lambda, cnmow_version=cnmow_version)
    return encoder

def get_cbow_encoder(n_words, padding_idx = 0, word_emb_dim = 784):
    encoder = Word2MatEncoder(n_words, word_emb_dim = word_emb_dim,
                              padding_idx = padding_idx, w2m_type = "cbow")
    return encoder

def get_cbow_cmow_hybrid_encoder(n_words, padding_idx = 0, word_emb_dim = 400, initialization_strategy = "identity", w2m_type = "cmow", _lambda = 0, cnmow_version=1):
    """
    The very last input is added so that the hybrid model can choose between cmow and cnmow
    """
    cbow_encoder = get_cbow_encoder(n_words, padding_idx = padding_idx, word_emb_dim = word_emb_dim)
    if w2m_type == "cmow":
        cmow_encoder = get_cmow_encoder(n_words, padding_idx = word_emb_dim,
                                   word_emb_dim = word_emb_dim,
                                   initialization_strategy = initialization_strategy)
    elif w2m_type == "cnmow":
        cmow_encoder = get_cnmow_encoder(n_words, padding_idx = word_emb_dim,
                                   word_emb_dim = word_emb_dim,
                                   initialization_strategy = initialization_strategy, _lambda = _lambda, cnmow_version=cnmow_version)

    encoder = HybridEncoder(cbow_encoder, cmow_encoder)
    return encoder
