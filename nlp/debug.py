import pyprind
import torch
from torchtext.legacy import data 
from torchtext.vocab import GloVe
from torchtext.data import get_tokenizer
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def binary_accuracy(preds, y):
    #round predictions to the closest integer
    rounded_preds = torch.round(preds)

    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


def mytrain(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    #定义进度条
    bar = pyprind.ProgBar(len(iterator), bar_char='█')

    for batch in iterator:
        optimizer.zero_grad()

        input = torch.LongTensor(batch.text)

        predictions = model(input).squeeze()
        #target = batch.target
        target = batch.target.to(torch.float)

        loss = criterion(predictions, target)
        acc = binary_accuracy(predictions, target)
        #print('loss = ',loss)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        bar.update()

    return epoch_loss / len(iterator),epoch_acc /len(iterator)

def evaluate(model, iterator, criterion):

    epoch_loss = 0
    #epoch_acc = 0

    model.eval()

    with torch.no_grad():
        bar = pyprind.ProgBar(len(iterator), bar_char='█')
        for batch in iterator:

            input = torch.LongTensor(batch.text)

            predictions = model(input)
            target = (batch.target.reshape(32,1)).to(torch.float)
            
            loss = criterion(predictions, target)
            #print('loss = ',loss)
            #acc = binary_accuracy(predictions, batch.target)
            epoch_loss += loss.item()
            #epoch_acc += acc.item()
            bar.update()
    return epoch_loss / len(iterator)


class my_model(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim)
        self.dense = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #embedding
        x = self.embedding(x)
        '''
        #partial droupout 1D
        x = x.permute(0, 2, 1)   # convert to [batch, channels, time]
        x = F.dropout2d(x, 0.2, training=self.training)
        x = x.permute(0, 2, 1)   # back to [batch, time, channels]
        '''
        #LSTM
        x, (hidden,cell) = self.rnn(x)
        x = x[-1]

        #hidden = torch.cat((hidden[-2,:,:],hidden[-1,:,:]),dim = 1) 
        #Dense & Activation
        x = self.dense(x)
        x = self.sigmoid(x)
        return x


if __name__ == '__main__':

    tokenizer = get_tokenizer("basic_english")

    #定义数据处理方式
    TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True)
    LABEL = data.Field(sequential=False,
                    use_vocab=False,
                    preprocessing=data.Pipeline(
                        lambda x: int(float(x)),)
                    )

    #从表格导入数据集
    train1 = data.TabularDataset(
        path=r'./'+'usetrain.csv',
        format='csv',
        skip_header=True,  # 如果不加会把标题也读进去
        fields=[('', None), ('id', None), ('keyword', None), ('location', None),
                ('text', TEXT), ('target', LABEL)]  # 对应的列用对应处理方式
    )
    test = data.TabularDataset(
        path=r'./'+'usetest.csv',
        format='csv',
        skip_header=True,
        fields=[('', None), ('id', None), ('keyword', None), ('location', None),
                ('text', TEXT), ('target', None)]
    )

    #导入对应embedding，自动下载的文件存储在./.vector_cache
    TEXT.build_vocab(train1, vectors=GloVe(name='6B', dim=100))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #batch操作
    train_iter, test_iter = data.Iterator.splits(
        (train1, test),
        sort_key=lambda x: len(x.text),
        batch_sizes=(32, 32),
        device=device,
    )

    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 64
    OUTPUT_DIM = 1

    model = my_model(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)

    #导入glove权重
    model.embedding.weight.data = TEXT.vocab.vectors
    model.embedding.weight.requires_grad = False

    #选择优化器与损失函数
    optim = optim.Adam(model.parameters(),lr=1e-3)
    criterion = nn.BCELoss()

    N_EPOCHS = 2
    for epoch in range(N_EPOCHS):
        train_loss,train_acc = mytrain(model, train_iter, optim, criterion)
        
        print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f}| Train Acc: {train_acc*100:.3f}')

    test_loss,test_acc = evaluate(model,test_iter,criterion)
    print(f'| Epoch: {epoch+1:02} | Test Loss: {train_loss:.3f}| Test Acc: {train_acc*100:.3f}')
    

