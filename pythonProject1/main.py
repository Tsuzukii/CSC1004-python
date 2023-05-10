from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import multiprocessing as mp
import copy

from utils.config_utils import read_args, load_config, Dict2Object

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(args, model, device, train_loader, optimizer, epoch):
    """
    tain the model and return the training accuracy
    :param args: input arguments
    :param model: neural network model
    :param device: the device where model stored
    :param train_loader: data loader
    :param optimizer: optimizer
    :param epoch: current epoch
    :return:
    """
    model.train()
    total_loss = 0
    total_correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target) #calculate the loss function
        loss.backward() #gradient moves backward
        optimizer.step() #optimizer update
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        total_correct += pred.eq(target.view_as(pred)).sum().item()
        if (batch_idx) % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
    '''Fill your code'''
    #calculate training accuracy and loss

    training_acc, training_loss = total_correct / len(train_loader.dataset), total_loss / len(train_loader.dataset)
    return training_acc, training_loss


def test(model, device, test_loader):
    """
    test the model and return the tesing accuracy
    :param model: neural network model
    :param device: the device where model stored
    :param test_loader: data loader
    :return:
    """
    model.eval()
    test_loss = 0
    total_correct = 0
    with torch.no_grad(): #we can avoid storing the computations done producing the output of our network in the computation graph.
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            total_correct += pred.eq(target.view_as(pred)).sum().item()
            #'''Fill your code'''
            pass
    test_loss /= len(test_loader.dataset)
    testing_acc = total_correct / len(test_loader.dataset)    
    testing_acc, testing_loss = testing_acc, test_loss
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, total_correct, len(test_loader.dataset),
        100. * total_correct / len(test_loader.dataset)))
    return testing_acc, testing_loss

def save_results_to_txt(file_name, data):
    """
    save the accuracy and loss to txt file
    """
    with open(file_name, 'w') as f : # open a file for writing, and creates the file if it does not exist.
        for item in data :
            f.write(f"{item}\n")

def read_results_from_txt(file_name):
    """
    read the accuracy and loss from txt file
    """
    with open(file_name, 'r') as f:
        data = [float(line.strip()) for line in f.readlines()] # eliminate the line break and transform them into float
    return data


def plot(title, x_label, y_label, x_data, y_data, file_name):
    """
    plot the model peformance
    :param epoches: recorded epoches
    :param performance: recorded performance
    :return:
    """
    plt.figure()
    plt.plot(x_data, y_data)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(file_name)
    plt.close()

def calculate_mean_results(train_loss_files, train_acc_files, test_loss_files, test_acc_files):
    """
    Calculate the mean training and testing results from multiple runs
    """
    mean_train_loss = []
    mean_train_acc = []
    mean_test_loss = []
    mean_test_acc = []

    for i in range(len(train_loss_files)):
        train_loss = read_results_from_txt(train_loss_files[i])
        train_acc = read_results_from_txt(train_acc_files[i])
        test_loss = read_results_from_txt(test_loss_files[i])
        test_acc = read_results_from_txt(test_acc_files[i])

        if i == 0:
            mean_train_loss = train_loss
            mean_train_acc = train_acc

            mean_test_loss = test_loss
            mean_test_acc = test_acc
        else:
            mean_train_loss = [sum(x) for x in zip(mean_train_loss, train_loss)]
            mean_train_acc = [sum(x) for x in zip(mean_train_acc, train_acc)]
            mean_test_loss = [sum(x) for x in zip(mean_test_loss, test_loss)]
            mean_test_acc = [sum(x) for x in zip(mean_test_acc, test_acc)]

    mean_train_loss = [x / len(train_loss_files) for x in mean_train_loss] #
    mean_train_acc = [x / len(train_acc_files) for x in mean_train_acc]
    mean_test_loss = [x / len(test_loss_files) for x in mean_test_loss]
    mean_test_acc = [x / len(test_acc_files) for x in mean_test_acc]

    return mean_train_loss, mean_train_acc, mean_test_loss, mean_test_acc


def run(config):
    seed = config.seed
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    use_mps = not config.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': config.batch_size}
    test_kwargs = {'batch_size': config.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # download data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=config.lr)

    """record the performance"""
    epoches = []
    training_accuracies = []
    training_loss = []
    testing_accuracies = []
    testing_loss = []

    scheduler = StepLR(optimizer, step_size=1, gamma=config.gamma)
    for epoch in range(1, config.epochs + 1):
        train_acc, train_loss = train(config, model, device, train_loader, optimizer, epoch)
        test_acc, test_loss = test(model, device, test_loader) #seemingly a typo here
        """record training info and record testing info"""
        training_loss.append(train_loss)
        testing_loss.append(test_loss)
        training_accuracies.append(train_acc)
        testing_accuracies.append(train_acc)

        scheduler.step()
        """update the records"""

    #"""plotting training performance with the records"""
    #plot(epoches, training_loss)

    #"""plotting testing performance with the records"""
    #plot(epoches, testing_accuracies)
    #plot(epoches, testing_loss)


    plot(f"Training Loss - Seed {seed}", "Epochs", "Loss", list(range(1, config.epochs + 1)), training_loss,
         f"training_loss_seed_{seed}.png")
    plot(f"Testing Loss - Seed {seed}", "Epochs", "Loss", list(range(1, config.epochs + 1)), testing_loss,
         f"testing_loss_seed_{seed}.png")
    plot(f"Testing Accuracy - Seed {seed}", "Epochs", "Accuracy", list(range(1, config.epochs + 1)), testing_accuracies,
         f"testing_accuracy_seed_{seed}.png")
    plot(f"Training Accuracy - Seed {seed}", "Epochs", "Accuracy", list(range(1, config.epochs + 1)),training_accuracies,
         f"training_accuracy_seed_{seed}.png")

    if config.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

    train_loss_file = f"train_loss_seed_{seed}.txt"
    train_acc_file = f"train_acc_seed_{seed}.txt"
    test_loss_file = f"test_loss_seed_{seed}.txt"
    test_acc_file = f"test_acc_seed_{seed}.txt"

    save_results_to_txt(train_loss_file, training_loss)
    save_results_to_txt(train_acc_file, training_accuracies)
    save_results_to_txt(test_loss_file, testing_loss)
    save_results_to_txt(test_acc_file, testing_accuracies)

def run_single(config):
    run(config)

def main(config):
    with mp.Pool(processes=config.num_processes) as pool: #create pool and processes from the config file
        configs = [copy.deepcopy(config) for _ in range (config.num_processes)]
        seeds = config.seed
        for i, conf in enumerate(configs):
            conf.seed = seeds [i]
        pool.map(run_single, configs) #get the seed from the config file and set them for every single run
        train_loss_files = [] # initialize array for files
        train_acc_files = []
        test_loss_files = []
        test_acc_files = []
        for seed in seeds:
            train_loss_file = f"train_loss_seed_{seed}.txt" #set the file names in the format of "file name + current seed for every run"
            train_acc_file = f"train_acc_seed_{seed}.txt"
            test_loss_file = f"test_loss_seed_{seed}.txt"
            test_acc_file = f"test_acc_seed_{seed}.txt"
            train_loss_files.append(train_loss_file)
            train_acc_files.append(train_acc_file)
            test_loss_files.append(test_loss_file)
            test_acc_files.append(test_acc_file)

    mean_train_loss, mean_train_acc, mean_test_loss, mean_test_acc = calculate_mean_results(
        train_loss_files, train_acc_files, test_loss_files, test_acc_files
    )

    plot("Mean Training Loss", "Epochs", "Loss", list(range(1, config.epochs + 1)), mean_train_loss,
         "mean_train_loss.png")
    plot("Mean Training Accuracy", "Epochs", "Accuracy", list(range(1, config.epochs + 1)), mean_train_acc,
         "mean_train_accuracy.png")
    plot("Mean Testing Loss", "Epochs", "Loss", list(range(1, config.epochs + 1)), mean_test_loss,
         "mean_test_loss.png")
    plot("Mean Testing Accuracy", "Epochs", "Accuracy", list(range(1, config.epochs + 1)), mean_test_acc,
         "mean_test_accuracy.png")



if __name__ == '__main__':
    arg = read_args()
    # load training settings
    config = load_config(arg)
    main(config)
