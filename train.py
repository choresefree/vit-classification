import torch
from tqdm import tqdm
from PIL import Image
import torch.utils.data
from torch import optim, nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from timm.loss import LabelSmoothingCrossEntropy
from timm.models.vision_transformer import vit_small_patch16_224


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class_dict = {
        0: 'Black_footed_Albatross',
        1: 'Laysan_Albatross',
        2: 'Sooty_Albatross',
        3: 'Groove_billed_Ani',
        4: 'Crested_Auklet'
    }

    # define data tfs
    data_transform = {
        "train":
            transforms.Compose([transforms.Resize(250, Image.BILINEAR),
                                transforms.RandomCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "test":
            transforms.Compose([transforms.Resize(250, Image.BILINEAR),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }

    # load train_dataset
    train_root = 'dataset/train'
    train_dataset = ImageFolder(root=train_root, transform=data_transform['train'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    train_number = len(train_dataset)

    # load test_dataset
    test_root = 'dataset/test'
    test_dataset = ImageFolder(root=test_root, transform=data_transform['test'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)
    test_number = len(test_dataset)

    # define and load model
    net = vit_small_patch16_224(pretrained=True)
    net = net.to(device)

    # define training details
    lr = 1e-3
    criterion = nn.CrossEntropyLoss()
    # criterion = LabelSmoothingCrossEntropy()
    optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=5e-4, momentum=0.9)
    epochs = 120
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=0, last_epoch=-1)
    best_acc = 0
    for epoch in range(epochs):
        loss_train = 0
        acc_test = 0
        net.train()
        for _, data in enumerate(tqdm(train_loader, desc='Train {}'.format(epoch + 1))):
            images, labels = data[0].to(device), data[1].to(device)
            output = net(images)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        scheduler.step()
        net.etest()
        with torch.no_grad():
            for _, data in enumerate(tqdm(test_loader, desc='test {}'.format(epoch + 1))):
                images, labels = data[0].to(device), data[1].to(device)
                output = net(images)
                prediction = output.argmax(dim=1)
                acc_test += torch.eq(prediction, labels).sum().float().item()
        loss_train = loss_train / train_number
        acc_test = acc_test / test_number
        print('epoch{} loss: '.format(epoch + 1), loss_train, 'acc-test: ', acc_test)
        # store the best model on test set
        if acc_test > best_acc:
            best_acc = acc_test
            torch.save(net.state_dict(), 'checkpoint.pth')


def main():
    train()


if __name__ == '__main__':
    main()
