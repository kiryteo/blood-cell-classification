from dataset import GlobulesDataset, pad_collate
from model import RecurrentConvNet
from sampler import BalancedBatchSampler
import time

from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import WeightedRandomSampler, BatchSampler

from torch.optim import Adam
import torch
import torch.nn as nn

from catalyst.dl import SupervisedRunner, AccuracyCallback, AUCCallback, ConfusionMatrixCallback

path = "tmp/train1.dat"
#path = "tmp/single_file.dat"
batch_size = 384
num_epoch = 128
logdir = f'logs/{time.time()}'


dataset = GlobulesDataset(path)

train_dataset, validation_dataset = random_split(dataset, lengths=[int(len(dataset)*0.8), len(dataset) - int(len(dataset)*0.8)])

train_sampler = BalancedBatchSampler(train_dataset, [x[1] for x in train_dataset])
validation_sampler = BalancedBatchSampler(validation_dataset, [x[1] for x in validation_dataset])

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=pad_collate)
valid_loader = DataLoader(validation_dataset, batch_size=batch_size, collate_fn=pad_collate, sampler=validation_sampler)

loaders = {"train": train_loader, "valid": valid_loader}

model = RecurrentConvNet(device='cuda').to('cuda')
criterion = nn.NLLLoss()
optimizer = Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

runner = SupervisedRunner(device='cuda')

runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=loaders,
    logdir=logdir,
    num_epochs=num_epoch,
    verbose=True,
    callbacks=[AccuracyCallback(), ConfusionMatrixCallback(num_classes=3)]
)





