import torch 
import torch.nn as nn 
from torch.nn import functional as F    

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# ------------

# -------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,) ) # 무작위로 선택된 인덱스의 배열 생성 
    # 0~len(data) - block_size 까지의 범위에서 정수 무작위 선택. 
    #(batch_size,)는 함수에게 생성할 무작위 정수의 개수를 알려줍니다.
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    # 결과적으로, 이 함수는 0부터 len(data) - block_size 범위 내에서 무작위로 batch_size 개수만큼의 정수를 선택
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split) 
            logits, loss = model(X, Y)  

# simple bigram bodel


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) # vocab_size x vocab_size (어휘의 크기 사용)
        # idx를 전달할 때 입력의 모든 단일 정수가 이 임베딩 테이블을 참조하고 뽑을 것임.
        # 인덱스에 해당하는 임베딩 테이블의 행을 꺼내므로 여기에서는 24를 임베딩 테이블로 이동하여 24번째 행을 뽑고 43번째 행을 차단

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of intergers
        logits = self.token_embedding_table(idx) #(B,T,C) (배치, 텐서, 채널) ex(4,8,65)
        # 여기서 logits은 신경망 모델이 생성한 예측값임. BigramLanguageModel을 통해 얻어진 각 단어에 대응하는 다음 단어의 로짓값을 나타냄.
        # logit이란 확률을 로그 오즈로 변환한 값으로, 신경망이 특정 클래스에 속할 원시 예측 확률을 나타내기 전의 값입니다. 

        if targets is None:
            loss = None
        else:
            # F.cross entropy 손실 함수에 적합한 형태로 만든다.
            B, T , C = logits.shape
            logits = logits.view(B*T, C) # (B*T, C) (32, 65)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits , loss

    def generate(self, idx, max_new_tokens):
        # idx는 (B,T) 사이즈의 정수 텐서입니다.
        for _ in range(max_new_tokens):
            # 예측값을 얻기 위해 마지막 토큰을 사용합니다.
            logits, loss = self(idx)
            # 마지막 타임 스텝에 집중합니다.
            logits = logits[:,-1,:] # (B,C)가 됩니다.
            # 로짓을 소프트맥스로 변환합니다.
            probs = F.softmax(logits, dim=-1) # (B,C)가 됩니다.
            # distribution에서 샘플링합니다.
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
            # 샘플링된 idx_next를 idx에 추가합니다.
            idx = torch.cat((idx, idx_next), dim=-1)  # (B, T+1)
        
        return idx


model = BigramLanguageModel(vocab_size)
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))