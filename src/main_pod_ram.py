from podkopaev_ramdas.baseline_alg import pod_ram_mnist
from podkopaev_ramdas.tests import Drop_tester
from utils import *
import argparse
import torch.optim as optim
import torch.nn.functional as F

import pdb


class MLP(nn.Module):
    """
    A simple 3-layer MLP for MNIST/CIFAR-10 classification.
    Input: (N, 1, 28, 28)
    Output: (N, 10) for 10 classes (digits 0..9)
    """
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (N, 1, 28, 28)
        # Flatten to (N, 784)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
    

def train_one_epoch(model, device, train_loader, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Compute accuracy
        _, predicted = outputs.max(dim=1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, device, data_loader):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            running_loss += loss.item() * images.size(0)

            _, predicted = outputs.max(dim=1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def fit(model, epochs, train_loader, optimizer, setting, device):
    """
    Train the model on the training set.
    """
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, device, train_loader, optimizer)
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")

    torch.save(model.state_dict(), os.getcwd() + '/../pkl_files/best_model_' + setting + '.pth')
    return model


def main():
    parser = argparse.ArgumentParser(description='Run Podkopaev & Ramdas Baseline.')
    parser.add_argument('--source_conc_type', type=str, default='betting', help='Type of concentration inequliity for source UCB')
    parser.add_argument('--target_conc_type', type=str, default='conj-bern', help='Type of concentration inequliity for target LCB')
    parser.add_argument('--num_of_repeats', type=int, default=50, help='Number of repetitions for the algorithm')
    parser.add_argument('--eps_tol', type=float, default=0.05, help='Epsilon tolerance')
    parser.add_argument('--data_name', type=str, default='mnist', help='Dataset name')
    parser.add_argument('--val_set_size', type=int, default=1000, help='Validation set size')
    parser.add_argument('--corruption_type', type=str, default='fog', help='Type of corruption to apply to MNIST/CIFAR dataset')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='Maximum number of epochs to train the model')

    args = parser.parse_args()
    source_conc_type = args.source_conc_type
    target_conc_type = args.target_conc_type
    num_of_repeats = args.num_of_repeats
    eps_tol = args.eps_tol
    data_name = args.data_name
    val_set_size = args.val_set_size
    corruption_type = args.corruption_type
    lr = args.lr
    epochs = args.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    setting = 'pod_ram-s_conc {}-t_conc {}-repeat {}-eps_tol {}-data {}-corruption {}-lr {}'.format(
        source_conc_type,
        target_conc_type,
        num_of_repeats,
        eps_tol,
        data_name,
        corruption_type,
        lr
    )
    print('Setting:\n', setting)
    if data_name == 'mnist':
        loaders = get_mnist_data(val_set_size=val_set_size)
        loader_1 = get_mnist_c_data(corruption_type=corruption_type, val_set_size=val_set_size)
    elif data_name == 'cifar10':
        loaders = get_cifar10_data(val_set_size=val_set_size)
        loader_1 = get_cifar10_c_data(corruption_type=corruption_type, val_set_size=val_set_size)
    if data_name == 'mnist':
        model = MLP(input_size=784, hidden_size=256, num_classes=10).to(device)
    elif data_name == 'cifar10':
        model = MLP(input_size=3*32*32, hidden_size=1024, num_classes=10).to(device)
    
    loader = loaders[0] # train loader
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model = fit(model, epochs, loader, optimizer, setting, device)
    tester = Drop_tester()
    tester.eps_tol = eps_tol
    tester.source_conc_type = source_conc_type
    tester.target_conc_type = target_conc_type

    if data_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        clean_data = torchvision.datasets.MNIST(
            root='/cis/home/xhan56/code/wtr/data',
            train=False,
            transform=transform
        )
        mnist_c_path = os.path.join('/cis/home/xhan56/code/wtr/data/mnist_c', corruption_type)
        corrupted_data = NpyDataset(os.path.join(mnist_c_path, 'test_images.npy'), 
                                os.path.join(mnist_c_path, 'test_labels.npy'), transform=transform)
    
    pod_ram_mnist(
        model=model,
        ds_clean=clean_data,
        ds_corrupted=corrupted_data,
        tester=tester,
        device=device,
        num_of_repeats=num_of_repeats,
        setting=setting,
        corruption_type=corruption_type
    )
    print('Done!')


if __name__ == '__main__':
    main()