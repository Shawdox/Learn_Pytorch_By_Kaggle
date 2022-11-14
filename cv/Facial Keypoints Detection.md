---
title : Facial Keypoints Detection
author: Shaw
categories: Kaggle
tags: [ "CV", "Pytorch"]
---

# Facial Keypoints Detection

>Time: 2022.11.11
>
>Author: Shaw
>
>[Facial Keypoints Detection | Kaggle](https://www.kaggle.com/competitions/facial-keypoints-detection/overview)

## 一、任务简介：

- ​	**任务**：正确标注所给图片的人脸关键位置；

<!--more-->

- ​    **关键位置（keypoint）**：

| 字段                    | 解释       |
| ----------------------- | ---------- |
| left_eye_center         | 左眼中心   |
| right_eye_center        | 右眼中心   |
| left_eye_inner_corner   | 左眼内眼角 |
| left_eye_outer_corner   | 左眼外眼角 |
| right_eye_inner_corner  | 右眼内眼角 |
| right_eye_outer_corner  | 右眼外眼角 |
| left_eyebrow_inner_end  | 左眉内侧   |
| left_eyebrow_outer_end  | 左眉外侧   |
| right_eyebrow_inner_end | 右眉内侧   |
| right_eyebrow_outer_end | 右眉外侧   |
| nose_tip                | 鼻尖       |
| mouth_left_corner       | 嘴左角     |
| mouth_right_corner      | 嘴右角     |
| mouth_center_top_lip    | 上唇中心   |
| mouth_center_bottom_lip | 下唇中心   |

- ​    **数据组成**：
  - **training.csv:** 包含7049张训练图片，每张图片有15个keypoint坐标（每个坐标用x,y两列数据表示，<u>有些数值缺失</u>），图像数据作为按行排序的像素列表。故表格中的数据标签有30列，图片数据一列共31列；
  - **test.csv:** 包含1783张测试图片；
  - **ubmissionFileFormat.csv:** 待提交的测试结果。

-    **样例**：

![](https://shaw-typora.oss-cn-beijing.aliyuncs.com/20221113142355.png)

## 二、数据分析处理

首先加载训练数据与测试数据：

```python
#Load the data
train_data = pd.read_csv(TRAIN_PATH)
test_data = pd.read_csv(TEST_PATH)
print(train_data.columns)
print(test_data.columns)
```

### 2.1 空值填补

因为数据中存在缺失值，故统计分析存在缺失值的列：

```python
train_data.isnull().any().value_counts()
#out:
'''
True     28
False     3
dtype: int64
'''
```

故28个属性列都存在缺失值，接下来填补缺失，‘ffill’方法表示用前面一列的值填补当前位置的值：

```python
train_data.fillna(method = 'ffill',inplace = True)
```

同时，Image列中的部分数值由‘ ’空格替代，这里将其替换为0：

```python
train_data['Image_new'] = train_data['Image'].map(lambda x:['0' if i == ' ' else i for i in x.split(' ')])

image_list = np.array(train_data['Image_new'].to_list(),dtype='float')
```

### 2.2 相关图片处理函数编写

```python
IMG_SIZE = 96  # image size 96 x 96 pixels

def show_keypoints(image, keypoints):
    '''
    Show image with keypoints
    Args:
        image (array-like or PIL image): The image data. (M, N)
        keypoints (array-like): The keypoits data. (N, 2)
    '''
    plt.imshow(image, cmap='gray')
    if len(keypoints):
        plt.scatter(keypoints[:, 0], keypoints[:, 1], s=24, marker='.', c='r')

def show_images(df, indxs, ncols=5, figsize=(15,10), with_keypoints=True):
    '''
    Show images with keypoints in grids
    Args:
        df (DataFrame): data (M x N)
        idxs (iterators): list, Range, Indexes
        ncols (integer): number of columns (images by rows)
        figsize (float, float): width, height in inches
        with_keypoints (boolean): True if show image with keypoints
    '''
    plt.figure(figsize=figsize)
    nrows = len(indxs) // ncols + 1
    for i, idx in enumerate(indxs):
        image = np.fromstring(df.loc[idx, 'Image'], sep=' ').astype(np.float32)\
                .reshape(-1, IMG_SIZE)
        if with_keypoints:
            keypoints = df.loc[idx].drop('Image').values.astype(np.float32)\
                        .reshape(-1, 2)
        else:
            keypoints = []
        plt.subplot(nrows, ncols, i + 1)
        plt.title(f'Sample #{idx}')
        plt.axis('off')                 #关闭坐标轴
        plt.tight_layout()              # tight_layout会自动调整子图参数，使之填充整个图像区域
        show_keypoints(image, keypoints)
    plt.show()
```



### 2.3 分割数据集

使用Dataset和DataLoader相关pytorch类来加载数据集。

定义数据集子类用于自动处理dataframe数据，返回{'image': image, 'keypoints': keypoints}类型的数据：

```python
class FaceKeypointsDataset(Dataset):
    '''Face Keypoints Dataset'''

    def __init__(self, dataframe, train=True, transform=None):
        '''
        Args:
            dataframe (DataFrame): data in pandas dataframe format.
            train (Boolean) : True for train data with keypoints, default is True
            transform (callable, optional): Optional transform to be applied on 
            sample
        '''
        self.dataframe = dataframe
        self.train = train
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image = np.fromstring(self.dataframe.iloc[idx, -1], sep=' ')\
            .astype(np.float32).reshape(-1, IMG_SIZE)
        if self.train:
            keypoints = self.dataframe.iloc[idx, :-1].values.astype(np.float32)
        else:
            keypoints = None
        sample = {'image': image, 'keypoints': keypoints}
        if self.transform:
            sample = self.transform(sample)
        return sample
```

将数据正则化，转化为tensor：

```python
class Normalize(object):
    '''Normalize input images'''
    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        return {'image': image / 255., # scale to [0, 1]
                'keypoints': keypoints}
                
class ToTensor(object):
    '''Convert ndarrays in sample to Tensors.'''
    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.reshape(1, IMG_SIZE, IMG_SIZE)
        image = torch.from_numpy(image)
        if keypoints is not None:
            keypoints = torch.from_numpy(keypoints)
            return {'image': image, 'keypoints': keypoints}
        else:
            return {'image': image}
```

利用SubsetRandomSampler构建采样器，并使用Dataloader构建数据加载器：

```python
from torch.utils.data.sampler import SubsetRandomSampler

def prepare_train_valid_loaders(trainset,valid_size = 0.2,batch_size = 128):

    # obtain training indices that will be used for validation
    # 这里保障了训练集和测试集的随机划分
    num_train = len(trainset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floot(valid_size*num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    # 这里保障了一轮epoch中每次生成的batch是随机的，避免模型学习到数据的顺序特征
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               sampler=valid_sampler)

    return train_loader, valid_loader
```



正式加载数据：

```python
from torchvision import transforms

# how many samples per batch to load
batch_size = 128
# percentage of training set to use as validation
valid_size = 0.2

# Define a transform to normalize the data
tsfm = transforms.Compose([Normalize(), ToTensor()])

# Load the training data and test data
trainset = FaceKeypointsDataset(train_data, transform=tsfm)
testset = FaceKeypointsDataset(test_data, train=False, transform=tsfm)

# prepare data loaders
train_loader, valid_loader = prepare_train_valid_loaders(trainset,
                                                         valid_size,
                                                         batch_size)

test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size)
```



## 三、模型搭建

```python
#Model
class CNN(nn.Module):
    def __init__(self,outputs = 30):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*12*12, 1024)
        self.fc2 = nn.Linear(1024, outputs)
        self.dropout = nn.Dropout(0.3)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64*12*12)
        x = F.relu(self.fc1(self.dropout(x)))
        x = self.fc2(self.dropout(x))
        return x
```

## 四、训练与测试

```python
#Train & Validate
device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN(outputs = 30)
model = model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

def train(train_loader,valid_loader,model,criterion,optimizer,n_epochs = 50,saved_model = 'cv_model.pt'):
    valid_loss_min = np.Inf
    train_losses = []
    valid_losses = []

    for epoch in range(n_epochs):

        train_loss = 0.0
        valid_loss = 0.0

        #training
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()

            output = model(batch['image'].to(device))
            loss = criterion(output,batch['keypoints'].to(device))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()*batch['image'].size(0)
        
        #validating
        model.eval()
        for batch in valid_loader:
            output = model(batch['image'].to(device))
            loss = criterion(output, batch['keypoints'].to(device))
            valid_loss += loss.item()*batch['image'].size(0)

        train_loss = np.sqrt(train_loss/len(train_loader.sampler.indices))
        valid_loss = np.sqrt(valid_loss/len(valid_loader.sampler.indices))

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'
              .format(epoch+1, train_loss, valid_loss))

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'
                  .format(valid_loss_min, valid_loss))
            torch.save(model.state_dict(), saved_model)
            valid_loss_min = valid_loss

    return train_losses, valid_losses
```

定义测试函数：

```python
def predict(data_loader, model):
    '''
    Predict keypoints
    Args:
        data_loader (DataLoader): DataLoader for Dataset
        model (nn.Module): trained model for prediction.
    Return:
        predictions (array-like): keypoints in float (no. of images x keypoints).
    '''

    model.eval()  # prep model for evaluation

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(batch['image'].to(device)).cpu().numpy()
            if i == 0:
                predictions = output
            else:
                predictions = np.vstack((predictions, output))

    return predictions
```

定义测试结果展示函数：

```python
def view_pred_df(columns, test_df, predictions, image_ids=range(1,6)):
    '''
    Display predicted keypoints
    Args:
        columns (array-like): column names
        test_df (DataFrame): dataframe with ImageId and Image columns
        predictions (array-like): keypoints in float (no. of images x keypoints)
        image_id (array-like): list or range of ImageIds begin at 1
    '''
    pred_df = pd.DataFrame(predictions, columns=columns)
    pred_df = pd.concat([pred_df, test_df], axis=1)
    pred_df = pred_df.set_index('ImageId')
    show_images(pred_df, image_ids)  # ImageId as index begin at 1
```

定义测试结果上交文件生成函数：

```python
def create_submission(predictions, pred_file='data/preds.csv', sub_file='data/submission.csv', columns=None):
    '''
    Create csv file for submission from predictions
    Args:
        predictions (array-like): prediction (no. fo images x 30 keypoints)
        pred_file (string): file path for prediction csv file
        sub_file (string): file path for submission csv file
        columns (dict): provided column names for submission file
    '''
    lookup = pd.read_csv('data/IdLookupTable.csv')
    if columns == None:
        columns = train_data.columns[:-1]
    preds = pd.DataFrame(predictions,
                         index=np.arange(1, len(predictions)+1),
                         columns=columns)
    preds.to_csv(pred_file)
    locations = [preds.loc[image_id, feature_name]
                 for image_id, feature_name
                 in lookup[['ImageId', 'FeatureName']].values]
    locations = [location if location <
                 IMG_SIZE else IMG_SIZE for location in locations]
    lookup.Location = pd.Series(locations)
    lookup[['RowId', 'Location']].to_csv(sub_file, index=False)
```

训练 and 预测：

```python
#Train & Validate
train(train_loader, valid_loader, model, criterion,  optimizer, n_epochs=50, saved_model='aug_cnn.pt')

#Predict
model.load_state_dict(torch.load('aug_cnn.pt'))
predictions = predict(test_loader,model)
create_submission(predictions,
                  pred_file='data/aug_cnn_preds.csv',
                  sub_file='data/aug_cnn_submission.csv')
columns = train_data.drop('Image', axis=1).columns
view_pred_df(columns, test_data, predictions,range(1,11))
```

预测结果：

![](https://shaw-typora.oss-cn-beijing.aliyuncs.com/20221113170335.png)

将submission文件上交到Kaggle评分:

![](https://shaw-typora.oss-cn-beijing.aliyuncs.com/20221113170259.png)

