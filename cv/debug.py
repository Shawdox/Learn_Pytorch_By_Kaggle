from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

IMG_SIZE = 96  # image size 96 x 96 pixels
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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


def show_images(df, indxs, ncols=5, figsize=(15, 10), with_keypoints=True):
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
        plt.axis('off')  # 关闭坐标轴
        plt.tight_layout()              # tight_layout会自动调整子图参数，使之填充整个图像区域
        show_keypoints(image, keypoints)
    plt.show()

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


def prepare_train_valid_loaders(trainset, valid_size=0.2, batch_size=128):

    # obtain training indices that will be used for validation
    num_train = len(trainset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size*num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               sampler=valid_sampler)

    return train_loader, valid_loader



if __name__ == '__main__':

    #Global Constants
    BASIC_PATH = r'./data/'
    TRAIN_PATH = os.path.join(BASIC_PATH,'training.csv')
    TEST_PATH = os.path.join(BASIC_PATH,'test.csv')

    #Load the data
    train_data = pd.read_csv(TRAIN_PATH)
    test_data = pd.read_csv(TEST_PATH)
    
    train_data.fillna(method='ffill', inplace=True)

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

