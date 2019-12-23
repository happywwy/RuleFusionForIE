# -*- coding: utf-8 -*-


import numpy as np
import cPickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math, copy, time
import evaluate as eva

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    """
    Encoder is a multi-layer transformer with BiGRU
    """
    def __init__(self, encoder, classifier, logic, src_embed, pos_emb, dropout):
        super(Model, self).__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.logic = logic
        self.src_embed = src_embed
        self.pos_emb = pos_emb
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, pos_ind, src_mask, entity=False, rel=False):
        memory, attn = self.encode(src, src_mask, pos_ind)     
        out = self.decode(memory, src_mask, entity, rel)
        return out, attn
    
    def encode(self, src, src_mask, pos_ind):
        X = torch.cat((self.dropout(self.src_embed(src)), self.pos_emb[pos_ind]), dim=2)
        return self.encoder(X, src_mask)
    
    def decode(self, memory, src_mask, entity, rel):
        return self.classifier(memory, src_mask, entity, rel)


        
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
    
    
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N, d_in, d_h):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.gru = nn.GRU(d_in, d_h, batch_first=True, dropout=0.1, bidirectional=True)
        
    def forward(self, X, mask):
        x, _ = self.gru(X)
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
            attn = layer.self_attn.attn
        return x, attn
        
        
class LayerNorm(nn.Module):
    "Construct a layernorm module."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        #return x + self.dropout(sublayer(self.norm(x)))
        return x + sublayer(x)
        
        
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
        

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    return torch.matmul(p_attn, value), p_attn
    
    
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        # d_k is the output dimension for each head
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
        
        
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
        
        
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab, emb):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model, _weight=emb)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x)
                
        
class PositionalEncodingEmb(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=100):
        super(PositionalEncodingEmb, self).__init__()
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)
        self.position = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(max_len + 1, d_model)), requires_grad=True)
        
    def forward(self, x):
        if x.size(1) < self.max_len:
            p = self.position[torch.arange(x.size(1))]
            p = p.unsqueeze(0)
            p = p.expand(x.size(0), p.size(1), p.size(2))
        else:
            i = torch.cat((torch.arange(self.max_len), -1 * torch.ones((x.size(1) - self.max_len), dtype=torch.long)))
            p = self.position[i]
            p = p.unsqueeze(0)
            p = p.expand(x.size(0), p.size(1), p.size(2))
        return self.dropout(x + p)


class Logic_module(nn.Module):
    def __init__(self, nhead):
        super(Logic_module, self).__init__()
        self.linear_con = nn.Linear(1, 1)
        self.linear_dis = nn.Linear(1, 1)
        self.nhead = nhead
        self.weights = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(1, 15)), requires_grad=True)
        self.Wrel = nn.Linear(10, 1)
    
    def forward(self, output, attn, mask):
        mask = mask.squeeze()
        output = output.view(mask.size(0), mask.size(1), -1)
        BA_idx, IA_idx, null_idx = self.fact_node(output)
        rules_as, loss_rel = self.rule_node(output, attn, BA_idx, IA_idx, null_idx, mask)
        return rules_as, loss_rel, F.sigmoid(self.weights)
        
    # output has size batch_size x seq_length x nclass
    def fact_node(self, output):
        # batch_size x seq_size
        idxs = torch.argmax(output, dim=2)
        BA_idx = [torch.nonzero((i == 1)).squeeze() for i in idxs]
        IA_idx = [torch.nonzero((i == 2)).squeeze() for i in idxs]
        null_idx = [torch.nonzero((i == 0)).squeeze() for i in idxs]
        # list with len=batch_size
        return BA_idx, IA_idx, null_idx
        
    def rule_node(self, output, attn, BA_idxs, IA_idxs, null_idxs, mask):
        # p_ao has size batch_size x seq_length       
        p_a = torch.max(output, dim=2)[0]
        rules_as = []
        for i in torch.arange(output.size(0)):
            seq_len = torch.nonzero(mask[i].squeeze()).size(0)

                            
            # rule 1: Null(i-1) -> !IA(i)
            null_idx = null_idxs[i]
            null_idx = null_idx[null_idx < (seq_len - 1)]
            p_null = p_a[i][null_idx]
            # IA
            ra_1 = torch.zeros(seq_len).to(device)
            if null_idx.nelement() != 0:
                ra_1[null_idx + 1] = torch.min(1.0 - p_null, 0.2 * torch.ones(p_null.nelement()).to(device))
            
                                       
            # rule 2: IA(i) -> BA(i-1) or IA(i-1)
            IA_idx = IA_idxs[i]
            IA_idx = IA_idx[IA_idx < seq_len]
            IA_idx = IA_idx[IA_idx > 0]
            p_IA = p_a[i][IA_idx]
            # BA
            ra_2 = torch.zeros(seq_len).to(device)
            # IA
            ra_3 = torch.zeros(seq_len).to(device)
            if IA_idx.nelement() != 0:
                ra_2[IA_idx - 1] = p_IA
                ra_3[IA_idx - 1] = p_IA
            
            
            loss_rel = 0.0
            
            attn_i = attn[i] # nhead x seq x seq
            BA_idx = BA_idxs[i] 
            count = 0
            if BA_idx.nelement() > 1:
                for ind, id1 in enumerate(BA_idx[:-1]):
                    for id2 in BA_idx[ind + 1:]:
                        count += 1
                        att1 = attn_i[:, id1, id2]
                        att2 = attn_i[:, id2, id1]
                        rel = F.sigmoid(self.Wrel(att1 + att2))
                        rule_rel = F.sigmoid(self.linear_con((p_a[i, id1] + p_a[i, id2] - 2).view(-1, 1)))
                        loss_rel += (rule_rel - rel) ** 2
            
                        
            rules_as.append([ra_1, ra_2, ra_3]) 
        
        if count > 0:
            loss_rel = (F.sigmoid(self.weights[0, 0]) * loss_rel / count).item()
                       
        return rules_as, loss_rel
            

# The decoder is also composed of a stack of N=6 identical layers.
class Classifier(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, sch_k, d_in, d_h, d_e, h, dchunk_out, drel_out, vocab, dropout):
        super(Classifier, self).__init__()
        self.sch_k = sch_k
        self.vocab = vocab
        self.d_in = d_in
        self.d_h = d_h
        self.h = h
        self.dchunk_out = dchunk_out
        self.lstm = nn.LSTM(3 * d_in, d_h, batch_first=True, dropout=0.5, bidirectional=True)
        self.gru = nn.GRU(3 * d_in, d_h, batch_first=True, dropout=0.1, bidirectional=True)
        self.chunk_out = nn.Linear(2 * d_h + d_e, dchunk_out)        
        self.softmax = nn.Softmax(dim=1)
        self.pad = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(1, d_in)), requires_grad=True)
        self.label_emb = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(dchunk_out + 1, d_e)), requires_grad=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, src_mask, entity=False, rel=False):
        
        # transformer with label GRU            
        X = self.dropout(X)    
        X_pad = torch.cat((self.pad.repeat(X.size(0),1,1), X, self.pad.repeat(X.size(0),1,1)), dim=1)
        l = X_pad[:, :-2, :]
        m = X_pad[:, 1:-1, :]
        r = X_pad[:, 2:, :]
        
        # transformer with Elman GRU
        output_h, _ = self.gru(torch.cat((l,m,r), dim=2))
        #output_h, _ = self.gru(X)
        # concatenate label embedding
        output_batch = torch.Tensor().to(device)
        output_h_ = output_h.permute(1,0,2)
        hi = torch.cat((output_h_[0,:,:], self.label_emb[-1].repeat(output_h_.size(1), 1)), dim=1)
        output_chunki = self.softmax(self.chunk_out(hi)).view(hi.size(0), 1, -1)
        output_batch = torch.cat((output_batch, output_chunki), dim=1)
        chunki = torch.argmax(self.softmax(self.chunk_out(hi)), dim=1)
        chunk_batch = torch.argmax(self.softmax(self.chunk_out(hi)), dim=1).view(-1, 1)
        for h in output_h_[1:,:,:]:
            hi = torch.cat((h, self.label_emb[chunki]), dim=1)
            output_chunki = self.softmax(self.chunk_out(hi)).view(hi.size(0), 1, -1)
            output_batch = torch.cat((output_batch, output_chunki), dim=1)
            chunki = torch.argmax(self.softmax(self.chunk_out(hi)), dim=1)
            chunk_batch = torch.cat((chunk_batch, chunki.view(-1, 1)), dim=1)
        output_chunk = output_batch.view(-1, self.dchunk_out)
        
        if entity and not rel:
            return output_chunk

        
    

# full model
def make_model(vocab, pos, emb, sch_k=1.0, N=2, d_in=350, d_h=100, d_e=25, d_p=50, dchunk_out=3,
               drel_out=2, d_model=200, d_emb=300, d_ff=200, h=10, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    pos_emb = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(pos, d_p)), requires_grad=True)
    position = PositionalEncodingEmb(d_emb, dropout)
    logic_module = Logic_module(nhead=10)
    model = Model(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N, 350, 100),
        Classifier(sch_k, d_model, d_h, d_e, h, dchunk_out, drel_out, vocab, 0.1),
        logic_module,
        nn.Sequential(Embeddings(d_emb, vocab, emb), c(position)),
        pos_emb, dropout)
    
    return model
    


class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, pos, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        self.trg = trg
        self.pos = pos
        self.ntokens = torch.tensor(self.src != pad, dtype=torch.float).data.sum()

    

def run_logic_epoch(data_iter, model, loss_compute, entity=False, rel=False):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    
    for i, batch in enumerate(data_iter):
        out_chunk, attn = model.forward(batch.src.to(device), batch.pos.to(device), \
                batch.src_mask.to(device), entity, rel)
        loss_logic = model.logic(out_chunk, attn, batch.src_mask.to(device))
        loss = loss_compute(out_chunk, loss_logic, batch.trg.to(device), \
            batch.ntokens.to(device), batch.src_mask.to(device))    
        total_loss += loss.item()
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        elapsed = time.time() - start
        print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                (i, loss, tokens / elapsed))
        start = time.time()
        tokens = 0
            
    return total_loss

def predict(data_iter, model, entity=False, rel=False):
    labels_all = []
    
    with torch.no_grad():        
        if entity and not rel:
            for batch in data_iter:
                out, _ = model.forward(batch.src.to(device), batch.pos.to(device), \
                        batch.src_mask.to(device), entity, rel)
                label = torch.argmax(out, dim=1)
                labels_all.append(label)
            return labels_all
                            
                
def data_batch(idxs_src, labels_src, pos_src, batchlen):
    assert len(idxs_src) == len(labels_src)
    batches = [idxs_src[x : x + batchlen] for x in xrange(0, len(idxs_src), batchlen)]
    label_batches = [labels_src[x : x + batchlen] for x in xrange(0, len(labels_src), batchlen)]    
    pos_batches = [pos_src[x : x + batchlen] for x in xrange(0, len(idxs_src), batchlen)]
    for batch, label, pos in zip(batches, label_batches, pos_batches):
        # compute length of longest sentence in batch
        batch_max_len = max([len(s) for s in batch])
        batch_data = 0 * np.ones((len(batch), batch_max_len))
        batch_labels = 0 * np.ones((len(batch), batch_max_len))
        batch_pos = 0 * np.ones((len(batch), batch_max_len))

        # copy the data to the numpy array
        for j in range(len(batch)):
            cur_len = len(batch[j])
            batch_data[j][:cur_len] = batch[j]
            batch_labels[j][:cur_len] = label[j]
            batch_pos[j][:cur_len] = pos[j]
        # since all data are indices, we convert them to torch LongTensors        
        batch_data, batch_labels, batch_pos = torch.LongTensor(batch_data), \
            torch.LongTensor(batch_labels), torch.LongTensor(batch_pos)
        # convert Tensors to Variables
        yield Batch(batch_data, batch_pos, batch_labels, 0)

        
class logicLoss:
    def __init__(self, opt=None):
        self.opt = opt
    def __call__(self, x, x_logic, y, norm, mask):
        # losses_flat: (batch * max_len, 1)
        losses_flat = -torch.log(torch.gather(x.contiguous(), dim=1, index=y.contiguous().view(-1, 1)))
        # losses: (batch, max_len)
        losses = losses_flat.view(*y.size())
        
        # mask: (batch, max_len)
        mask = mask.squeeze()
        (out_as, loss_rel, weights) = x_logic
        loss_logic = 0.0
        for i in range(mask.size(0)):
            [as_1, as_2, as_3] = out_as[i]
            x = x.view(mask.size(0), mask.size(1), -1)
            
            if as_1.nonzero().size(0) > 0:
                loss_logic += weights[0, 1] * ((as_1[as_1.nonzero().squeeze()] - x.contiguous()[i, as_1.nonzero().squeeze(), 2]) ** 2).sum() / (as_1.nonzero().size(0))
            if as_2.nonzero().size(0) > 0:
                dis_BA = (as_2[as_2.nonzero().squeeze()] - x.contiguous()[i, as_2.nonzero().squeeze(), 1]) ** 2
                dis_IA = (as_3[as_3.nonzero().squeeze()] - x.contiguous()[i, as_3.nonzero().squeeze(), 2]) ** 2
                loss_logic += weights[0, 2] * torch.min(dis_BA, dis_IA).sum() / (as_2.nonzero().size(0))
        
        loss = (losses * mask.float()).sum() / norm + 0.5 * (loss_logic + loss_rel)
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.zero_grad()
        return loss
        


criterion = nn.NLLLoss()
emb = cPickle.load(open("data/Aspect-Opinion/embedding300_laptop", "rb"))
idxs_train = cPickle.load(open("data/Aspect-Opinion/idx_laptop.train", "rb"))
idxs_test = cPickle.load(open("data/Aspect-Opinion/idx_laptop.test", "rb"))
labels_train, labels_type_train = cPickle.load(open("data/Aspect-Opinion/labels_chunk_laptop.train", "rb"))
labels_test, labels_type_test = cPickle.load(open("data/Aspect-Opinion/labels_chunk_laptop.test", "rb"))
pos_train, pos_test = cPickle.load(open("data/Aspect-Opinion/pos_laptop.tag", "rb"))
pos_dic = ['pad']
posind_train, posind_test = [], []
for pos_line in pos_train:
    pos_ind = []
    for pos in pos_line:
        if pos not in pos_dic:
            pos_dic.append(pos)
        pos_ind.append(pos_dic.index(pos))
    posind_train.append(pos_ind)
for pos_line in pos_test:
    pos_ind = []
    for pos in pos_line:
        if pos not in pos_dic:
            pos_dic.append(pos)
        pos_ind.append(pos_dic.index(pos))
    posind_test.append(pos_ind)

emb = torch.tensor(emb, dtype=torch.float)
model = make_model(emb.shape[0], len(pos_dic), emb, sch_k=1.0, N=2)
model = model.to(device)

f_out = open("result/Aspect-Opinion/laptop-transformer-logic.txt", "w")

optimizer = optim.Adadelta(model.parameters())

label_map = {0:'O', 1:'B-ASP', 2:'I-ASP'}
entity_map = {'Aspect':0}


for epoch in range(100):    
    model.train()
    loss = run_logic_epoch(data_batch(idxs_train, labels_train, posind_train, 25), model, 
        logicLoss(optimizer), entity=True, rel=False)
    
    model.eval()
    labels_predict = predict(data_batch(idxs_test, labels_test, \
                posind_test, 1), model, entity=True, rel=False)
        
    labels_test_map = [label_map[item] for sub in labels_test for item in sub]
    labels_predict_map = [label_map[t.item()] for sub in labels_predict for t in sub]

    print epoch    
    print eva.f1_score(labels_test_map, labels_predict_map)
    report = eva.classification_report(labels_test_map, labels_predict_map)
    print report
    print loss
    f_out.write("epoch: " + str(epoch) + "\n")
    f_out.write("performance on entity extraction:" + "\n")
    f_out.write("precision: " + str(eva.precision_score(labels_test_map, labels_predict_map)))
    f_out.write("\t" + "recall: " + str(eva.recall_score(labels_test_map, labels_predict_map)))
    f_out.write("\t" + "f1: " + str(eva.f1_score(labels_test_map, labels_predict_map)))
    f_out.write("\n" + report)
    f_out.write("\n")
    

f_out.close()
 
    