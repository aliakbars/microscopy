from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pickle
import matplotlib.pyplot as plt
import numpy as np

import readdata3 as readdata
from sklearn import metrics
from tqdm import tqdm

opts = {
  'img_dir': 'tb/',
  'annotation_dir': 'tb/',
  'detection_probability_threshold': 0.5,
  'detection_overlap_threshold': 0.3, 
  'gauss': 1,
  'patch_size': (160,160),
  'image_downsample' : 8,
  'detection_step': 5,
  'patch_creation_step': 40,
  'object_class': None,
  'negative_training_discard_rate': .9
 }
opts['patch_stride_training'] = int(opts['patch_size'][0]*.25)

batch_size = 128
nb_classes = 2
nb_epoch = 500

# input image dimensions
img_rows, img_cols = 20, 20
# number of convolutional filters to use
# nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

input_shape = (3, img_rows, img_cols)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 7, 3)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(7, 12, 2)
        self.hidden3 = nn.Linear(768, 100)
        self.output = nn.Linear(100, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = x.view(-1, 768)
        x = F.relu(self.hidden3(x))
        x = F.softmax(self.output(x), dim=1)
        return x

def create_data(balance_train=True, balance_test=False, augment_train=True, augment_test=True, opts=opts):
    trainfiles, valfiles, tesles = readdata.create_sets(opts['img_dir'],
                                                        train_set_proportion=.5, 
                                                        test_set_proportion=.5,
                                                        val_set_proportion=0)

    Y_train, X_train = readdata.create_patches(
        trainfiles,
        opts['annotation_dir'],
        opts['img_dir'],
        opts['patch_size'][0],
        opts['patch_stride_training'],
        grayscale=False,
        downsample=opts['image_downsample'],
        objectclass=opts['object_class'],
        negative_discard_rate=opts['negative_training_discard_rate']
    )
    Y_test, X_test = readdata.create_patches(tesles,
        opts['annotation_dir'],
        opts['img_dir'],
        opts['patch_size'][0],
        opts['patch_stride_training'],
        grayscale=False,
        downsample=opts['image_downsample'],
        objectclass=opts['object_class'],
        negative_discard_rate=opts['negative_training_discard_rate']
    )

    # Cut down on disproportionately large numbers of negative patches
    if balance_train:
        X_train, Y_train = readdata.balance(X_train, Y_train, mult_neg=100)
    if balance_test:
        X_test, Y_test = readdata.balance(X_test, Y_test, mult_neg=100)

    # Create rotated and flipped versions of the positive patches
    if augment_train:
        X_train, Y_train = readdata.augment_positives(X_train, Y_train)
    if augment_test:
        X_test, Y_test = readdata.augment_positives(X_test, Y_test)

    pickle.dump(((X_train, Y_train), (X_test, Y_test)), open('tb/tb.pkl', 'wb'))

def read_data():
    with open('tb/tb.pkl', 'rb') as f:
        (X_train, Y_train), (X_test, Y_test) = pickle.load(f)

    print()
    print('%d positive training examples, %d negative training examples' % (sum(Y_train), len(Y_train)-sum(Y_train)))
    print('%d positive testing examples, %d negative testing examples' % (sum(Y_test), len(Y_test)-sum(Y_test)))
    print('%d patches (%.1f%% positive)' % (len(Y_train)+len(Y_test), 100.*(float(sum(Y_train)+sum(Y_test))/(len(Y_train)+len(Y_test)))))

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    return X_train, Y_train, X_test, Y_test

def show_patches(X_train, Y_train, N_samples_to_display=10):
    pos_indices = np.where(Y_train)[0]
    pos_indices = pos_indices[np.random.permutation(len(pos_indices))]
    for i in range(N_samples_to_display):
        plt.subplot(2,N_samples_to_display,i+1)
        example_pos = X_train[pos_indices[i],:,:,:]
        example_pos = np.swapaxes(example_pos,0,2)
        plt.imshow(example_pos)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)

    neg_indices = np.where(Y_train==0)[0]
    neg_indices = neg_indices[np.random.permutation(len(neg_indices))]
    for i in range(N_samples_to_display,2*N_samples_to_display):
        plt.subplot(2,N_samples_to_display,i+1)
        example_neg = X_train[neg_indices[i],:,:,:]
        example_neg = np.swapaxes(example_neg,0,2)
        plt.imshow(example_neg)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    plt.gcf().set_size_inches(1.5*N_samples_to_display,3)

    plt.show()

def main():
    torch.manual_seed(91) # for reproducibility

    create_data(balance_train=True, balance_test=False,
                augment_train=True, augment_test=False, opts=opts)
        
    X_train, Y_train, X_test, Y_test = read_data()
    
    x_train = torch.from_numpy(X_train).float()
    x_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(Y_train).long()
    y_test = torch.from_numpy(Y_test).long()

    lr = 1e-4
    momentum = 0.9

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net().to(device)

    optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=0.1)
    criterion = nn.CrossEntropyLoss()

    train_data = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

    val_data = TensorDataset(x_test, y_test)
    val_loader = DataLoader(val_data, shuffle=False, batch_size=batch_size)

    def train(epoch):
        net.train()
        total_loss = 0
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        return total_loss/(i+1)

    def validate(epoch):
        net.eval()
        total_loss = 0
        with torch.no_grad():
            for i, (data, target) in enumerate(val_loader):
                data, target = data.to(device), target.to(device)
                output = net(data)
                loss = criterion(output, target)
                total_loss += loss.item()
            return total_loss/(i+1)

    loss = {
        'train': [],
        'valid': []
    }

    for epoch in tqdm(range(1, num_epochs+1)):
        loss['train'].append(train(epoch))
        loss['valid'].append(validate(epoch))

    # # To show learning curve
    # plt.plot(range(1, num_epochs+1), loss['train'], label='train')
    # plt.plot(range(1, num_epochs+1), loss['valid'], label='valid')
    # plt.xlabel('epoch')
    # plt.ylabel('cross entropy loss')
    # plt.legend();

    net.eval()
    y_pred = net.forward(X_test).numpy()
    fpr, tpr, thresholds = metrics.roc_curve(y_test[:,1], y_pred[:,1])
    roc_auc = metrics.auc(fpr, tpr)

    precision, recall, thresholds = metrics.precision_recall_curve(y_test[:,1], y_pred[:,1])
    average_precision = metrics.average_precision_score(y_test[:,1], y_pred[:,1])

    plt.title('ROC: AUC = %0.2f' % roc_auc)
    plt.plot(fpr, tpr, 'b')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylim([-.05, 1.05])
    plt.xlim([-.05, 1.0])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    plt.savefig('tb-test-{}.png'.format(i))

if __name__ == '__main__':
    main()
