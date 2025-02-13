import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)
#hyper-parameters
batch_size=64 #This is the size of one batch. 1 batch will contain 64 blocks
block_size=512 #This is the size of one block. Meaning that one block will contain 256 words
max_iters=50000
eval_interval=500
learning_Rate=3e-4
device='cuda' if torch.cuda.is_available() else 'cpu'
eval_iters=200
n_embd=384
n_head=6
n_layer=6
dropout=0.2


with open('input.txt','r',encoding='utf-8') as f:
    text=f.read()
print(len(text))

chars=sorted(list(set(text)))
vocab_size=len(chars)
print(chars)
print(vocab_size)

# Encoder and Decoder for integers
stoi={ch:i for i,ch in enumerate(chars)}
itos={i:ch for i,ch in enumerate(chars)}
encode= lambda s:[stoi[c] for c in s] # assign an integer to a char
decode=lambda l:"".join([itos[i] for i in l])

#Train test split
data=torch.tensor(encode(text), dtype=torch.long)
# print(data.shape,data.dtype)
n=int(0.9*len(data))
train_data=data[:n]
test_data=data[n:]

'''What we are doing now is that in a sentence encoded into [18,25,46,34], 
if the input is 18,output should be 25.If the input is [18,25], output should be 46 and so on
So we are trying to predict the next word in a sentence, based on the input tokens
'''
x=train_data[:block_size]
y=train_data[1:1+block_size]
print(x)
print(y)
# for t in range(block_size):          ## This code is to visualise the above described process
#     context=x[:t+1]
#     target=y[t]
#     print(f"when input is{context}, target is:{target}")

def get_batch(split):
    #generate a small batch of data inputs x and target y
    data=train_data if split=='train' else test_data
    ix=torch.randint(len(data)-block_size-1,(batch_size,))
    # print(f"This is ix: {ix}")
    x=torch.stack([data[i:i+block_size] for i in ix])
    y=torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y=x.to(device),y.to(device)
    return x,y

@torch.no_grad()
def estimate_loss():
    out={}
    m.eval()
    for split in ['train','test']:
        losses=torch.zeros(eval_iters)
        for k in range(eval_iters):
            x,y=get_batch(split)
            logits,loss=m(x,y)
            losses[k]=loss.item()
        out[split]=losses.mean()
    m.train()
    return out

#We'll start with super simple bigram model
'''
Bigram model is a very simple model.
It takes the last token in the input sequence and predicts the next token.
It doesn't take very large string or any string for that matter as I wrote above.
So it's output is mostly gibberish.
'''
class Bigram(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table=nn.Embedding(vocab_size,vocab_size)
    def forward(self,idx,targets=None):
        # idx is a tensor of shape (batch_size,block_size). Basically its input x
        # targets is a tensor of shape (batch_size,block_size)
        # We are trying to predict the next word in a sentence, based on the input tokens
        logits=self.token_embedding_table(idx)
        '''When idx is passed into this,
        logits will be produced which are basically the raw scores(before applying softmax)
        that represent model's confidence for each possible next token
        B(Batch size):The number of independent sequences processed in parallel
        T(Sequence length/block size): The number of tokens in each input sequence
        C(Vocab size): The total number of unique tokens in a dataset
        (B,T,C): The shape of the logit tensor,meaning each token in a batch has a prediction 
            over all possoble vocabulary tokens'''
        if targets is None:
            loss=None
        else:
            B,T,C=logits.shape
            logits=logits.view(B*T,C) #Basically the cross entropy function wants C as second input 
            # So we have merged 1st and 2nd to make C 2nd
            targets=targets.view(B*T)
            loss=F.cross_entropy(logits,targets)
        return logits,loss
    
    def generate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            logits,loss=self(idx)
            logits=logits[:,-1,:]
            probs=F.softmax(logits,dim=-1)
            idx_next=torch.multinomial(probs,num_samples=1)
            idx=torch.cat((idx,idx_next),dim=1)
        return idx
    
model=Bigram(vocab_size)
m=model.to(device)

optimizer=torch.optim.AdamW(m.parameters(),lr=learning_Rate)
for iter in range(max_iters):
    if iter%eval_interval==0:
        losses=estimate_loss()
        print(f"step {iter}:train loss {losses['train']:.4f},test loss {losses['test']:.4f}")
    xb,yb=get_batch('train')
    logits,loss=m(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
'''Basically in above loop, we are generating a batch with defined no of blocks and block size.
The blocks are randomly generated using randint.
These blocks are trained and loss is printed every 500 epochs.

The results are gibberish because the bigram model just takes the one char to check what will come next
So we'll use transformer model that will perform better'''      
context=torch.zeros((1,1),dtype=torch.long,device=device)
print(decode(m.generate(context,max_new_tokens=500)[0].tolist()))