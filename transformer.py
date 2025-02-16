import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
batch_size=64
block_size=512
max_iters=10000
eval_interval=10
learning_rate=3e-4
device='cuda' if torch.cuda.is_available() else 'cpu'
print(device)
eval_iters=200
n_embd=384 #Assumed each word is 384 dimension vector
n_head=6 #No of attention heads
n_layer=6 #No of Transformer blocks (stacked layers)
dropout=0.2
# (B,T,C)=(64,512,384)
'''
B= Batch_size
T= Block_size / No of words per sentence in a batch
C= embeddings size per word
'''

# head_size=n_embd/n_head=64
'''
Thus each token x will have a head_size=64 and it will projected into 64D vectors
==> K,Q,V for each token x will have 64 vectors
'''


torch.manual_seed(1337)

with open('input.txt','r',encoding='utf-8') as f:
    text=f.read()
vocab=sorted(list(set(text)))
size=len(vocab)
stoi={ch:i for i,ch in enumerate(vocab)}
itos={i:ch for i,ch in enumerate(vocab)}
encode=lambda x:[stoi[i] for i in x]
decode=lambda y: "".join([itos[i] for i in y])


data=torch.tensor(encode(text))
n=int(0.9*len(data))
train_data=data[:n]
test_data=data[n:]

def get_batch(split):
    x=train_data if split=='train' else test_data
    ix=torch.randint(len(x)-block_size-1,(batch_size,))
    x=torch.stack([data[i:i+block_size] for i in ix])
    y=torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y=x.to(device),y.to(device)
    return x,y

@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out

#Till here evrything was same as bigram.py
#Now we'll write code for GPTLanguagemodel using transformers

class Head(nn.Module): #This will be one head of self attention
    def __init__(self,head_size):
        super().__init__()
        #First 3 are key,query,value
        self.key=nn.Linear(n_embd,head_size,bias=False)
        self.query=nn.Linear(n_embd,head_size,bias=False)
        self.value=nn.Linear(n_embd,head_size,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        '''self.register_buffer = somethinf that is not a parameter so we create it as a buffer
        'tril'=name given to the matrix that we are creating(lower triangular matrix)
        torch.tril=torch.triu=torch.ones=torch.zeros are all functions that create a matrix of ones
        we create matrix of block_size,block_size because let's say our block size is 32 and we have to predict 33rd element
        so according to tril, first row will have 1 value i.e the influence of 1st char (incase to predict 2nd char)
        second row will have 2 values because of 1st and 2nd char(to predict 3rd)
        third row will have 3 values because of 1st,2nd and 3rd and so on 
        So the last row will have all the chars to take into context
        *** All the following values are zero becuase to calculate 4th char we need context of previous
        3 chars and not the following. So 4th row will have all places 0 except first four. 
        '''
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        # input of size(batch,time-step,channels)
        #output of size(batch,time-step.head size)
        B,T,C=x.shape
        k=self.key(x)  # of size (B,T,head_size)
        q=self.query(x)
        weights=(q @ k.transpose(-2,-1)) * q.shape[-1]**(-0.5) #calculating attention scores 
        # Division is done to kind of stabiise the values (normalise)
        # the output size will be (B,T,T)
        weights=weights.masked_fill(self.tril[:T,:T].bool()==0, float('-inf'))
        #we are replacing the 0s with -inf in attention scores because they don't matter
        weights=F.softmax(weights,dim=-1)
        '''
        The softmax function converts attention scores into probability distributions that tell 
        us which words contribute more to a word/ are more related to a word according to probability
        
        '''
        weights=self.dropout(weights)
        v=self.value(x)
        out=weights@v
        return out

'''
What we do with multi head attention is that if we have a token x of n_embd embeddings, it will be split into n_heads with
each head having head_size embeddings. This will make all heads have learning variation in attention patterns,reduces computational cost per head too!
'''
class MultiHeadAttention(nn.Module):
    def __init__(self,num_heads,head_size):    
        super().__init__()
        self.heads=nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj=nn.Linear(head_size*num_heads,n_embd)
        self.dropout=nn.Dropout(dropout)
    
    def forward(self,x):
        out=torch.cat([h(x) for h in self.heads],dim=-1)
        out=self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    '''A linear layer followed by non-linearity'''
    '''This is done to make actual prediciton of what will come next from the new values that we get after w@v'''
    def __init__(self,n_embd):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(n_embd,4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd,n_embd),
            nn.Dropout(dropout),
        )
    def forward(self,x):
        return self.net(x)
    
class Block(nn.Module):
# Transformer block:Joining all the pieces together
    def __init__(self,n_embd,n_head):
        super().__init__()
        head_size=n_embd//n_head
        self.sa=MultiHeadAttention(n_head,head_size)
        self.ffwd=FeedForward(n_embd)
        self.ln1=nn.LayerNorm(n_embd) # Layer Normalisation
        self.ln2=nn.LayerNorm(n_embd)
    def forward(self,x):
        # x=self.sa(self.ln1(x))
        x=x+self.sa(self.ln1(x)) # This is done because it creates some residual connections that helps minimise loss
                                # (more about residual connections in the nd if I read about them)
        # x=self.ffwd(self.ln2(x))
        x=x+self.ffwd(self.ln2(x))
        return x
'''
One thing to note: In the Attention is all you need paper, layer normalisation is done after self.sa and after self.ffwd
But here we are doing before because better.
'''        

class GPTLanguageModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.n_embd=n_embd 
        self.token_embedding_table=nn.Embedding(size,n_embd)
        self.position_embedding_table=nn.Embedding(block_size,n_embd)
        self.blocks=nn.Sequential(*[Block(n_embd,n_head=n_head) for _ in range(n_layer)])
        self.ln_f=nn.LayerNorm(n_embd)
        self.lm_head=nn.Linear(n_embd,size)
        self.apply(self.__init__weights)
    def __init__weights(self,module):
        if isinstance(module,nn.Linear):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
    def forward(self,idx,targets=None):
        B,T=idx.shape
        tok_emb=self.token_embedding_table(idx) #(B,T,C)
        pos_emb=self.position_embedding_table(torch.arange(T).expand(B,T).to(idx.device)) #(T,C)
        x=tok_emb+pos_emb
        x=self.blocks(x)
        x=self.ln_f(x)
        logits=self.lm_head(x)
        
        if targets is None:
            loss=None
        else:
            B,T,C=logits.shape
            logits=logits.view(B*T,C)
            targets=targets.view(B*T)
            loss=F.cross_entropy(logits,targets)
        return logits,loss     

# For training, comment everything after line 229
# For eval, comment everything from line 206 to 229 and uncomment after that

#Now everything is same too as bigram.py with minor differences
model = GPTLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
# # eval (uncomment this part and comment the above training loop to load the pretrained model and generate responses)
# # load the trained model
# checkpoint = torch.load('model_checkpoint.pth', map_location=device)
# # Initialize the model (if you haven't already)
# model = GPTLanguageModel()
# # Load the state_dict from the checkpoint into the model
# model.load_state_dict(checkpoint['state_dict'])
# # Set the model to evaluation mode
# model.eval()
# context = torch.zeros((1, 1), dtype=torch.long, device=device)
# print(decode(model.generate(context, max_new_tokens=1000)[0].tolist()))

'''
Residual Connections:

These connections (also called skip connections) adds the original x back.
This helps in preventing vanishing gradients adn makes learning easier

'''
