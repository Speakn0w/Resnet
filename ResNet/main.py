import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from torchvision import models
import Dataset
import config
import numpy as np

def train(model, loader, optimizer, lossfunc, epoch, testloader, scheduler):

    total = 0
    correct = 0

    for step,(images, labels) in enumerate(loader):
        images = images.to(config.DEVICE)
        labels = list(map(float, labels))
        labels = np.array(labels)
        labels = torch.from_numpy(labels)
        labels = labels.to(config.DEVICE)

        with torch.cuda.amp.autocast():
            optimizer.zero_grad()
            logits = model(images)
            #print(logits.device)
            loss = lossfunc(logits, labels.long())
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            _,predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    train_accuracy = correct/total
    #np.save(config.TRAIN_ACC_PATH + '/epoch_{}'.format(epoch),train_accuracy)

    saveloss = loss.cpu()
    saveloss = np.array(saveloss.detach())
    #np.save(config.LOSS_PATH + '/epoch_{}'.format(epoch), saveloss)


#calculate testloss
    total = 0
    correct = 0

    for step,(images, labels) in enumerate(testloader):
        images = images.to(config.DEVICE)
        labels = list(map(float, labels))
        labels = np.array(labels)
        labels = torch.from_numpy(labels)
        labels = labels.to(config.DEVICE)

        with torch.cuda.amp.autocast():
            logits = model(images)
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        test_accuracy = correct / total
        #np.save(config.TEST_ACC_PATH + '/epoch_{}'.format(epoch), test_accuracy)

    print("epoch_{}".format(epoch)+"  trainloss="+str(loss.item())+"  trainacc=" + str(train_accuracy) + "  testacc=" + str(test_accuracy))

# def test(model, testloader):


def main():
    model = models.resnet34(pretrained = False)
    if config.LOAD_MODEL:
        model.load_state_dict(torch.load(config.MODEL_PATH))

    #print(model.is_cuda)
    print(torch.cuda.is_available())
    inchannel = model.fc.in_features
    model.fc = nn.Linear(inchannel, 16)
    for param in model.parameters():
        param.requires_grad = True
    model = model.to(config.DEVICE)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, mode='min')



    train_dataset = Dataset.ClassificationDataset(
        root_dir = config.ROOT_DIR,
        json_path = config.TRAINJSON_PATH,
        transform = config.transforms,
    )

    class_sample_counts = [315, 186, 581, 726, 382, 178, 167, 70, 73, 456, 170, 72, 228, 425, 151, 543]
    weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)

    train_targets = train_dataset.get_classes_for_all_imgs()
    samples_weights = weights[train_targets]

    sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size = config.BATCH_SIZE,
        shuffle = False,
        sampler = sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    test_dataset = Dataset.ClassificationDataset(
        root_dir = config.ROOT_DIR,
        json_path = config.TESTJSON_PATH,
        transform = config.transforms,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    for epoch in range(config.NUM_EPOCHS):
        train(model, train_loader, optimizer, loss_function, epoch, test_loader, scheduler)
        if epoch == 100:
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.LEARNING_RATE * 0.1

if __name__ == "__main__":
    main()