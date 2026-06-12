from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import tiktoken
import inspect
import torch.distributed as dist
import numpy as np
import os
from torch.distributed import init_process_group, destroy_process_group
import os
from torch.nn.parallel import DistributedDataParallel as DDP
import time


print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name())


""" 
A short description of the dataclasses model:
It provides decorator and functions for automatically adding 
generated special methods such as __init__() and __repr__() to 
user defined classes.
"""



class CausalSelfAttention(nn.Module):
    
    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_head == 0#return assertion error for the unmatched condition
    # key, query, value projectipns for all heads but in a batch
        self.c_attn = nn.Linear(config.n_embd,3*config.n_embd)
        
    #output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
    #regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        #not really a 'bias', more of a mask but follwing the OpenAI/HF naming though
        self.register_buffer("bias",torch.tril(torch.ones(config.block_size,config.block_size)).
                             view(1,1,config.block_size,config.block_size))
    def forward(self,x):
        B,T,C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        
        qkv = self.c_attn(x)
        
        q,k,v = qkv.split(self.n_embd, dim=2)
        k = k.view(B,T, self.n_head,C // self.n_head).transpose(1,2)
        q = q.view(B,T, self.n_head,C // self.n_head).transpose(1,2)
        v = v.view(B,T, self.n_head,C // self.n_head).transpose(1,2)
        
        # #attetion(materializes the large (T,T) matrix for all the queries and keys)
        # att = (q @ k.transpose(-2,-1))* (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill (self.bias[:,:,:T,:T] == 0,float('-inf'))
        # att = F.softmax(att,dim=-1)
        # y = att @ v
        ''' 
        Flash Attention ensures the matrix N*N for attention never materialzes
        in the HBM. This makes the 7.6 times speedup for the attention compuation.
        '''
        y = F.scaled_dot_product_attention(q,k,v, is_causal=True)
        y = y.transpose(1,2).contiguous().view(B,T,C)#re-assemble all heads outputs side by side
        #output projection
        y = self.c_proj(y)
        return y        
        


class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd,4*config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')#GELU(x) = x.psi(x, psi(x) is the cdg of standard normal distribution
        self.c_proj = nn.Linear(4*config.n_embd,config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)#multi layer perceptron
        #attention is a communication operation
        #the tokens communicate with each other in attention
        #mlp has no infromation exchange between the tokens
        
    
    def forward(self,x):
        # avoid in-place ops to preserve autograd graph
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x




@dataclass
class GPTConfig:
    block_size:int = 1024 #max sequence length.
    vocab_size:int = 50257 #number of token: 50,000 BPE merges + 256 bytes tokens+ 1>[end of text]
    ''' 
    BPE(Byte pari encoding) builds tokens from frequent subword pattern. eg 
    "un","happi","ness"
    GPT learns about 50,000 common words/subwords,
    256 byte level tokens means emojis, symbols, special texts, curropted text,
    then the 1 end of text to separate training documents.'''
    n_layer: int = 12 #number of layers
    n_head: int = 12 #number of heads
    n_embd: int = 768 #embedding dimension
    
    
#The model  
class GPT(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),#wrapper module output embedding
                wpe = nn.Embedding(config.block_size, config.n_embd),#postion embedding
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),#all the blocks
                ln_f = nn.LayerNorm(config.n_embd)#lm head is the new module 
            )
        )
        self.lm_head = nn.Linear(config.n_embd,config.vocab_size,bias=False)
        
        #weight tying
        self.transformer.wte.weight = self.lm_head.weight #copies the data pointer of the lm_head
        # thus we are left with the single tensor
        
        #init params
        self.apply(self._init_weights)
    # mirroring the weight initalization
    ''' 
    _init_weights receives one module at a time
    if the module is the instance of the linear module we initalize
    the weights such that each entry is sampled from N(0,0.0.^2)
    the biases are initalized to zero.
    we do the same for the embedding. 
    '''     
    def _init_weights(self,module):
        std = 0.02
        #reduce weight initalization scale in deep networks to prevantactivations from exploading
        # model is a stack of residal blocks thus each block adds x = x+F(x) thus 
        #the varience grows with layers and activation becomes unstable
        #training becomes noisy or diverges
        #scalign reduces initalization std 
        #varience adds up roughly linearly thus we need to compensate by scaling
        # 2*L is beacuse of the 2 residual addtions from attention path and MLP layer
        if hasattr(module,'NANOGPT_SCALE_INIT'):
            std *= (2*self.config.n_layer) ** -0.5
            
        if isinstance(module,nn.Linear):
            torch.nn.init.normal_(module.weight,mean = 0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        
        
    #forward function for the model.
        """
        idx contains token id of the shape B,T where
        B is the batch size and T is the sequence length
        idx.size returns the batch size and the token length.
        the assertion is to ensure sequenece fits in the context window.
        Then the torch.arange creates the value from 0 upto the T-1
        Then we get the postion embeddings and the token embeddings from the transformer and add
        then for the each blocks of the tranformer we sent the encoded x 
        then apply the ln_f
        then obtain the logits
        
        """
    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size) ie. (B,T,50257)
        loss = None
        if targets is not None:
            # view here is flattening the # dim tensor, logits, to the 2 dim tensor
            # then the targets are flattening to the 1dim
           loss = F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1)) 
        return logits ,loss#logits is the unnormalized score
    

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        #GPT-2 uses conv1D but we need to use linear thus the conv1D uses (in_features,out_features), linear uses (out_features,in_features)
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    ''' 
    we want the linear weights , embeddings weights to decay during training'''
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        ''' 
        first we get all the named parameters
        then we exclude the frozen parameters of whcich donot have the grad
        then we separate th decay and non decay parameters
        we check for 2 dim weights since the biases has 1 dim and other has more than 2 dim
        then we create the group of the optimizers
        the num_decay_params used numel because it does mul of the dimensions to get the total number of the elements
        check for fused adamW for newer versions since it uses one optimized cuda kernel which is faster and less memeory traffic'''
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
    


''' 
current position is the pointer where we currently are in the same tokeen,
suppose B=2, T=4
then B*T+1 = 9
then the buf becomes 
the +1 so the targets are shifted by one . 
then we build x and y
the intention of the advance position is to reset the current position
to zero after we reach the end.
The current position pointer is to ensure that 
we go to the next sequence after processing all the sequences in the 
current batch.'''
def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    def __init__(self,B,T,process_rank,num_process,split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_process = num_process
        assert split in {'train','val'}
        
        # at init load tokens from disk and store them in the momory
        
        data_root = 'data'
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root,s) for s in shards]
        self.shards = shards
        assert len(shards) >0,f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
            
        enc = tiktoken.get_encoding('gpt2')
        
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B*self.T *self.process_rank
        self.reset()
        
    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_position])
        self.current_position = self.B*self.T*self.process_rank
    
    def next_batch(self,device):
        
        B,T = self.B,self.T
        buf = self.tokens[self.current_position: self.current_position+B*T+1]
        buf = buf.to(device=device)
        #inputs
        x = (buf[:-1]).view(B,T)
        #targets
        y = (buf[1:]).view(B,T)
        
        #advance the position in the tensor
        self.current_position += B * T* self.num_process
        #if loading the next baatch would be out of bounds , advacne to next_shards
        if self.current_position + (B*T*self.num_process+1) > len(self.tokens):
            self.current_shard =(self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B*self.T*self.process_rank
        return x,y 
        
        
#set up DDP (distributed data parallel)
# torchrun command stes the env variables RANK, LOCAL_RANK and WORLD_SIZE
''' 
with DDP we want to employ multiple GPUs at once
'''
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:# running through mutliple GPUs
    #use of DDP atm demands CUDA, 
    assert torch.cuda.is_available()
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK']) #rank identifies proces
    ddp_local_rank = int(os.environ['LOCAL_RANK'])# local rank identifies the gpu number in local machine
    ddp_world_size = int(os.environ['WORLD_SIZE'])# world size gives the total numeber of processes
    device = f'cuda:{ddp_local_rank}' # we seletc the device if cuda:2 means GPU two
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing, only one proces becomes master process
    
else: 
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    #attempt to autodetect the device
    device = 'CPU'
    if torch.cuda.is_available():
        device = 'CUDA' # if only one gpus avilable
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = 'mps'
    print(f"using device :{device}")

device_type = "cuda" if device.startswith("cuda") else "cpu"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

enc = tiktoken.get_encoding('gpt2')
torch.manual_seed(1337)
total_batch_size = 524288 #2**19 , ~0.5M in number of tokens
B = 4 #micro batch size 
T = 1024 # sequence length
assert  total_batch_size % (B*T*ddp_world_size) == 0 
grad_accum_steps = total_batch_size // (B*T * ddp_world_size)
if master_process:
    print(f"total desired batch size :{total_batch_size}")
    print(f"calculated gradient accumatilation steps :{grad_accum_steps}")

print("GPU", torch.cuda.current_device())
train_loader = DataLoaderLite(B=B,T=T,process_rank = ddp_rank, num_process = ddp_world_size,split="train")
val_loader = DataLoaderLite(B=B,T=T,process_rank = ddp_rank, num_process = ddp_world_size,split="val")

torch.set_float32_matmul_precision('high')
# 50304 is power of 2 which we consider the nice number
# we are increasing the number of embedding but we will drive the prob
# to zero for those numbers.
# in cuda, kernals use block tiles which are power of two ,
# thus for ugly numbers the kernals first perform the nice parts and move to the worst part later to do calcualation
model = GPT(GPTConfig(vocab_size=50304))
# model = torch.compile(model=model) use if linux avialable or triton works
model.to(device=device)
if ddp:
    model = DDP(model ,device_ids = [ddp_local_rank])
    # DDP does is once the backward pass is over and call the all_reduce and 
    # average on all the rank and pass to each rank
raw_model = model.module if ddp else model
max_lr = 6e-4
warmup_steps = 95 #GPT-3 said warming to 350M tokens but we are taking 10% since we have so less data
min_lr = max_lr*0.1 #10% of the initial learning rate
max_steps = 950 # since we are procesing 500M tokens and roughly we need these much steps

def get_lr(it):
    #1) linear warmup fro warmup_ites steps
    if it<warmup_steps:
        return max_lr*(it+1)/warmup_steps
    # if it>lr_decay_iters,return min learning rate
    if it>max_steps:
        return min_lr
    #3) cosine learning rate decay to 10% of the initial learning rate over the course of training
    decay_ratio = (it-warmup_steps) / (max_steps-warmup_steps)
    assert 0 <= decay_ratio <= 1.0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (max_lr - min_lr) # cosine decay to min learning rate
    
# optimizer = torch.optim.AdamW(model.parameters(),lr = 3e-4, betas=(0.9,0.95),eps=1e-8)
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device)
for i in range(max_steps):
    t0 = time.time()
    if i%10==0:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x,y = val_loader.next_batch(device=device)
                with torch.autocast(device_type=device,dtype=torch.bfloat16):
                    logits,loss = model(x,y)
                loss = loss/val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum,op = dist.ReduceOp.AVG)
        if master_process:
            print(f"Validation loss: {val_loss_accum.item():.4f}")
    if i>0 and i %10 == 0:
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = torch.tensor("Hello , I am a language model")
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences,1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42+ddp_rank)
        while xgen.size(1) < max_length:
            with torch.no_grad():
                logits,loss = model(xgen)
                logits = logits[:,-1,:]
                probs = F.softmax(logits,dim = -1)
                topk_probs, topk_indices = torch.topk(probs,50,dim= -1)
                ix = torch.multinomial(topk_probs,1,generator=sample_rng)
                xcol = torch.gather(topk_indices,-1,ix)
                xgen = torch.cat((xgen,xcol),dim=1)
        for i in range(num_return_sequences):
            tokens = xgen[i,:max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")
                
    model.train()        
    optimizer.zero_grad()#starting with the zero gradient
    loss_accum = 0.0
    ''' 
    To replicate the batch size of the gpt-3 paper we are trying 
    to accumulate the gradient before we do the update'''
    for micro_step in range(grad_accum_steps):
        x,y = train_loader.next_batch(device=device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16): 
            logits,loss = model(x,y)
        loss = loss/grad_accum_steps #since the loss are just getting added without the normalization.
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss_accum += loss.detach()   # to keep track over the losses of the microbatch in each batch.
        if ddp:
            model.require_backward_grad_sync = (micro_step==grad_accum_steps-1)
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum,op=dist.ReduceOp.AVG)
    ''' 
    During backpropagtion every parameters gets a gradient
    sometimes the gradient becomes very large
    this may cause the training to become unstable
    The global norm
    ---------------
    sqrt(square of each of the grad)
    if we have the gobal gradient norm 8.7 then 
    scale = 1/8.7
    which we use to multiply each of the grad '''
    #clip the global norm o fthe gradient at 1.0
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(i)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()#updating the parameters
    
    torch.cuda.synchronize()  # wait for all CUDA kernels on the current device to finish before continuing
    t1 = time.time()
    dt = (t1-t0) * 1000 # time difference in milliseconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps*ddp_world_size
    tokesn_per_sec = tokens_processed / (t1-t0)
    if master_process:
        print(f"step {i}, loss : {loss_accum.item()},norm : {norm:.4f}, dt : {dt:.2f}ms, tok/sec :{tokesn_per_sec}")
if ddp:
    destroy_process_group()