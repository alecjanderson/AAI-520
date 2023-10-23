
from flask import Flask, request, render_template

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
# Metrics
from nltk.translate.bleu_score import sentence_bleu
#from rouge_score import rouge_scorer

from datetime import datetime
import math

app = Flask(__name__)



movie_lines_path = 'movie_lines.txt'
movie_conversations_path = 'movie_conversations.txt'



class Vocab:
    def __init__(self):
        self.enum = {"PAD_token" : 0, "SOS_token" : 1, "EOS_token":2, "UNK":3}
        self.count = {}
        self.index = {}
        self.wordcount = 4
        self.min_freq = 3
    def addSentence(self,sentence):
        for word in sentence.split(' '):
            if word not in self.enum:
                if(word in self.count.keys()):
                    self.count[word] += 1
                    if(self.count[word] >= self.min_freq):
                        self.enum[word] = self.wordcount
                        self.index[self.wordcount] = word
                        self.wordcount += 1
                else:
                    self.count[word] = 1
            else:
                #print("Word already Added")
                self.count[word] += 1
    def __len__(self):
        return self.wordcount    
                
    ### This will be the class that handles the bag of words.
    
import string
def clean_String(stri):
    new_string = ''
    for i in stri:
        if i not in string.punctuation:
            new_string += i
    stri = new_string
    
    lower_string = stri.lower()
    no_number_string = re.sub(r'\d+','',lower_string)
    no_punc_string = re.sub(r'[^\w\s]','', no_number_string) 
    no_wspace_string = no_punc_string.strip()
    
    words = no_wspace_string.split()
    #filtered_words = [word for word in words if word not in stop_words]
    # I am unsure if removing stop words is correct on a chat bot for readability reasons
    #

        
    return ' '.join(words)
    

with open(movie_lines_path, encoding='iso-8859-1', errors='ignore') as my_file:
    all_lines = {}
    for line in my_file:
        split = line.split(' +++$+++ ')
        linemp = {}
        fields = ["lineID", "characterID", "movieID", "character", "text"]
        count = 0
        for field in (fields):
                linemp[field] = split[count]
                count +=1
        all_lines[linemp['lineID']] = linemp        
        

with open(movie_conversations_path, encoding='iso-8859-1', errors='ignore') as my_file:
    conv = []
    for line in my_file:
        split = line.split(' +++$+++ ')
        obj = {}
        fields = ["character1ID", "character2ID", "movieID", "utteranceIDs"]
        count = 0 
        for field in fields:
            obj[field] = split[count]
            count +=1
        ID = re.compile('L[0-9]+').findall(obj['utteranceIDs'])
        lines = []
        
        for id_ in ID:
            lines.append(all_lines[id_])
        obj['line'] = lines
        conv.append(obj)
        
pairs = []
for convrtsation in conv:
        for i in range(len(convrtsation['line'])):
            try:
                question = convrtsation['line'][i]['text'].strip()
                answer = convrtsation['line'][i+1]['text'].strip()
            except:
                pass
            if(question and answer):
                pairs.append([question, answer])
                
voc = Vocab()

pairs = pairs[:25000] #This is pretty much required?
for i in pairs:
    for j in i:
        cleaned = clean_String(j)
        voc.addSentence(cleaned)


class Attn(nn.Module):
    #Based off of https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html

    def scaled_dot_product(self, q, k, v, mask=None):
        
        sear = q.view(q.shape[0], -1, self.num_heads, self.divisor).permute(0, 2, 1, 3)   
        q = k.view(q.shape[0], -1, self.num_heads, self.divisor).permute(0, 2, 1, 3)  
        v = v.view(v.shape[0], -1, self.num_heads, self.divisor).permute(0, 2, 1, 3)  

        score = torch.matmul(sear, q.permute(0,1,3,2)) / math.sqrt(sear.size(-1))
        score = score.masked_fill(mask == 0, -1e9)    
        weights = F.softmax(score, dim = -1)          
        weights = self.dropout(weights)
        product = torch.matmul(weights, v)
        
        return product
        


    def __init__(self, num_heads, size):

        super(Attn, self).__init__()
        self.num_heads = num_heads
        self.dropout = nn.Dropout(0.1)
        self.query, self.key, self.value, self.concat  = nn.Linear(size, size), nn.Linear(size, size), nn.Linear(size, size), nn.Linear(size, size)
        self.divisor = int(size / num_heads)

    def forward(self, search, key, value, mask):
        search = self.query(search)
        value = self.value(value)
        search_key = self.key(key)
        product = self.scaled_dot_product(search, search_key, value, mask) 
    
        product = product.permute(0,2,1,3).contiguous().view(product.shape[0], -1, self.num_heads * self.divisor)
        interacted = self.concat(product)
        return interacted 
    
device = torch.device("cpu")


class Embedding(nn.Module):
        def __init__(self, voc_size , size, max_len = 30):
            super(Embedding, self).__init__()
            self.divs = 10000
            self.size = size
            self.dropout = nn.Dropout(0.1)
            self.embed = nn.Embedding(voc_size, size)
            self.out = self.pos_enc(max_len, self.size)
        
        def calc(self, out, size, pos, loc):
            out[pos, loc] = math.sin(pos / (self.divs ** ((2 * loc)/size)))
            out[pos, loc + 1] = math.cos(pos / (self.divs ** ((2 * (loc + 1))/size)))
            #print(out)
            return out
            
        def pos_enc(self, max_len, size):
            out = torch.zeros(max_len, size).to(device)
            
            for pos in range(max_len):  
                for loc in range(math.ceil(size/2)):
                    loc = loc * 2
                    out = self.calc(out, size, pos, loc)
            out = out.unsqueeze(0)   
            #print(out)
            return out
        def forward(self, enc_out):
            emb = self.embed(enc_out) * math.sqrt(self.size)
            emb += self.out[:, :emb.size(1)]  
            emb = self.dropout(emb)
            return emb
        

class Model(nn.Module):
        
    class Ann(nn.Module):

        def __init__(self, size):
            super(Model.Ann, self).__init__()
            
            self.fc1 = nn.Linear(size, 2048)
            self.fc3 = nn.Linear(2048 , size)
            self.dropout = nn.Dropout(0.1)

        def forward(self, x):
            out = F.relu(self.fc1(x))
            #out = self.fc2(self.dropout(out))
            out = self.fc3(self.dropout(out))
            return out

    class Encoder(nn.Module):

        def __init__(self, size, heads):
            super(Model.Encoder, self).__init__()
            self.layernorm = nn.LayerNorm(size)
            self.attn = Attn(heads, size)
            self.an = Model.Ann(size)
            self.dropout = nn.Dropout(0.1)

        def forward(self, emb, mask):
            atten = self.dropout(self.attn(emb, emb, emb, mask))
            atten = self.layernorm(atten + emb)
            out = self.dropout(self.an(atten))
            encoded = self.layernorm(out + atten)
            return encoded
    
    
    
    class Decoder(nn.Module):
        def __init__(self, size, heads):
            super(Model.Decoder, self).__init__()
            self.an = Model.Ann(size)
            self.dropout = nn.Dropout(0.1)
            
            self.layernorm = nn.LayerNorm(size)
            
            self.attn = Attn(heads, size)
            self.attb = Attn(heads, size)
            
        def forward(self, emb, encoded, s_mask, t_mask):
            search = self.dropout(self.attn(emb, emb, emb, t_mask))
            search = self.layernorm(search + emb)
            atten = self.dropout(self.attb(search, encoded, encoded, s_mask))
            atten = self.layernorm(atten + search)
            out = self.dropout(self.an(atten))
            decoded = self.layernorm(out + atten)
            return decoded
    
    def __init__(self, size, heads):
        super(Model, self).__init__()
        
        self.size = size
        self.vocab_size = len(voc)
        self.embed = Embedding(self.vocab_size, size)
        list_dec = []
        list_enc = []
        for i in range(3):
            list_enc.append(self.Encoder(size, heads))
            list_dec.append(self.Decoder(size, heads))
        
        self.encoder = nn.ModuleList(list_enc)
        self.decoder = nn.ModuleList(list_dec)
        
        self.logit = nn.Linear(size, self.vocab_size)
    
    def encode(self, src_words, src_mask):
        src_embeddings = self.embed(src_words)
        for layer in self.encoder:
            src_embeddings = layer(src_embeddings, src_mask)
        return src_embeddings
    
    def decode(self, target_words, target_mask, src_embeddings, src_mask):
        tgt_embeddings = self.embed(target_words)
        for layer in self.decoder:
            tgt_embeddings = layer(tgt_embeddings, src_embeddings, src_mask, target_mask)
        return tgt_embeddings
        
    def forward(self, word, s_mask, t_word, t_mask):
        s_emb = self.encode(word, s_mask)
        decoded = self.decode(t_word, t_mask, s_emb, s_mask)
        
        out = F.log_softmax(self.logit(decoded), dim = 2)
        return out


size = 512
heads = 8
device = torch.device("cpu")
epochs = 30
    

model = Model(size = size, heads = heads)

model.load_state_dict(torch.load("29model.pth"))

def evaluate(transformer, question, question_mask, max_len, word_map):
    
    rev_word_map = {v: k for k, v in word_map.enum.items()}
    transformer.eval()
    start_token = word_map.enum['SOS_token']
    encoded = transformer.encode(question, question_mask)
    words = torch.LongTensor([[start_token]]).to(device)
    
    for step in range(max_len - 1):
        size = words.shape[1]
        target_mask = torch.triu(torch.ones(size, size)).transpose(0, 1).type(dtype=torch.uint8)
        target_mask = target_mask.to(device).unsqueeze(0).unsqueeze(0)
        decoded = transformer.decode(words, target_mask, encoded, question_mask)
        predictions = transformer.logit(decoded[:, -1])
        _, next_word = torch.max(predictions, dim = 1)
        next_word = next_word.item()
        if next_word == word_map.enum['EOS_token']:
            break
        words = torch.cat([words, torch.LongTensor([[next_word]]).to(device)], dim = 1)   # (1,step+2)
        
    # Construct Sentence
    if words.dim() == 2:
        words = words.squeeze(0)
        words = words.tolist()
        
    sen_idx = [w for w in words if w not in {word_map.enum['SOS_token']}]
    sentence = ' '.join([rev_word_map[sen_idx[k]] for k in range(len(sen_idx))])
    
    return sentence


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
         text = request.form['question']
         
         max_len = 30
         enc_qus = [voc.enum.get(word, voc.enum['UNK']) for word in text.split()]
         question = torch.LongTensor(enc_qus).to(device).unsqueeze(0)
         question_mask = (question!=0).to(device).unsqueeze(1).unsqueeze(1)  
         sentence = evaluate(model, question, question_mask, int(max_len), voc)
            
         return render_template('page.html', input = sentence)
    if request.method == "GET":
        return render_template('page.html')
         
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)