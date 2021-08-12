
from functools import partial

import torch
import torchvision
from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule
from torch import nn, Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms, models
from torchvision.transforms import RandomResizedCrop, ColorJitter, RandomHorizontalFlip, RandomGrayscale, \
    RandomSolarize, Normalize, RandomApply

from vibcreg.losses import VIbCRegLoss
from vibcreg.modules import VIbCReg


class VIbCRegLight(LightningModule):
    def __init__(self,):
        super().__init__()
        self.vibcreg = VIbCReg(partial(models.resnet18, pretrained=False), 1000)
        self.augs = transforms.Compose([RandomResizedCrop(32, (0.5, 0.9)),
                                   RandomHorizontalFlip(),
                                   RandomGrayscale(0.5),
                                   RandomApply([ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.5),
                                   RandomSolarize(threshold=0.8, p=0.1),
                                   Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.criterion = VIbCRegLoss()

    def forward(self, x: Tensor) -> Tensor:
        return self.vibcreg(x)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        aug1, aug2 = self.augs(x), self.augs(x)
        z1, z2 = self(aug1), self(aug2)
        loss = self.criterion(z1, z2)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)


class LinearEval(LightningModule):
    def __init__(self, encoder: nn.Module):
        super(LinearEval, self).__init__()
        self.encoder = encoder
        for par in self.encoder.parameters():
            par.requires_grad = False
        self.class_head = nn.Linear(1000, 10)
        self.criterion = nn.CrossEntropyLoss()
        self.transform = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def forward(self, x: Tensor) -> Tensor:
        x = self.transform(x)
        with torch.no_grad():
            x = self.encoder(x)
        return self.class_head(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.criterion(out, y)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-4)


if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.ToTensor()])

    batch_size = 256

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

    testloader = DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    combined_dataset = ConcatDataset([trainset, testset])
    combined_loader = DataLoader(combined_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    model = VIbCRegLight()
    trainer = Trainer(gpus=1, max_epochs=5)
    trainer.fit(model, combined_loader)

    linear_eval = LinearEval(model.vibcreg.encoder)
    trainer_linear = Trainer(gpus=1, max_epochs=5)
    trainer_linear.fit(linear_eval, trainloader)
