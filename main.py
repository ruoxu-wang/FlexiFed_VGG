#!/usr/bin/env python3
# use Apple Silicon as https://developer.apple.com/metal/pytorch/
# --------------------
import torch
import torch.nn as nn
from torch.nn.functional import layer_norm
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from vgg import *
import copy
import json


# loading DataSet
def data_loader(data_dir, batch_size, random_seed=42, valid_size=0.1, test=False):
    # define transforms
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])

    if test:
        test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
        return test_dataset

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    return train_dataset

# training
def train_local_model(train_loader, client_model, num_epochs, learning_rate):
    client_model.to(device)
    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(client_model.parameters(), lr=learning_rate, weight_decay = 0.0005, momentum = 0.9)

    # training loop
    client_model.train()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Take the Tensors onto the device
            images = images.to(device)
            labels = labels.to(device)

            # Forward
            outputs = client_model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return client_model.state_dict()

# test
def evaluate_model(test_loader, client_model):
    client_model.eval()
    client_model.to(device)
    total_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = client_model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

if __name__ == '__main__':
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using mps device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using cuda device")
    else:
        device = torch.device("cpu")
        print("Using cpu device")

    # init training conf
    training_round = 2
    num_epochs = 10
    batch_size = 64
    learning_rate = 0.01
    num_client = 40 # init dataset for 40 client, every 10 client share one model
    round_results = {"loss": [], "accuracy": []}

    # init CIFAR10 dataset
    train_dataset = data_loader(data_dir='./data', batch_size=64)   # DataLoader init
    test_dataset = data_loader(data_dir='./data', batch_size=64, test=True)

    # for iid purpose
    data_len = len(train_dataset)
    data_idx = np.arange(data_len)
    np.random.shuffle(data_idx)

    test_data_len = len(test_dataset)
    test_data_idx= np.arange(test_data_len)
    np.random.shuffle(test_data_idx)

    # allocate to clients
    client_datasets = []
    split_size = data_len // num_client
    for i in range(num_client):
        client_idx = data_idx[i * split_size: (i + 1) * split_size]
        client_datasets.append(Subset(train_dataset, client_idx))

    client_data_splits = []
    for i in range(num_client):
        client_data = client_datasets[i]
        split_sizes = [len(client_data) // training_round] * training_round
        client_split = random_split(client_data, split_sizes)
        client_data_splits.append(client_split)

    client_test_datasets = []
    test_split_size = data_len // num_client
    for i in range(num_client):
        client_test_idx = data_idx[i * test_split_size: (i + 1) * test_split_size]
        client_test_datasets.append(Subset(test_dataset, client_test_idx))

    client_test_data_splits = []
    for i in range(num_client):
        client_test_data = client_test_datasets[i]
        split_sizes = [len(client_test_data) // training_round] * training_round
        client_test_split = torch.utils.data.random_split(client_test_data, split_sizes)
        client_test_data_splits.append(client_test_split)

    # init global model
    global_model = {}
    model_mapping = [(range(0, 10), vgg11), (range(10, 20), vgg13), (range(20, 30), vgg16), (range(30, 40), vgg19)]
    for model_range, model_fn in model_mapping:
        for current_idx in model_range:
            global_model[current_idx] = model_fn()

    # federated learning
    for round in range(training_round):
        print(f"Training Round {round + 1}")
        round_loss = []
        round_accuracy = []

        # local training
        # for client_id in range(num_client):
        #     print(f"Client {client_id + 1} Local Training...")
        #     # model distribution
        #     client_model = copy.deepcopy(global_model[client_id])
        #     # local training
        #     train_loader = DataLoader(client_data_splits[client_id][round], batch_size=batch_size, shuffle=True)
        #     client_state_dict = train_local_model(train_loader, client_model, num_epochs, learning_rate)
        #     global_model[client_id].load_state_dict(client_state_dict)

        # train 10 usr every time for ram limitation
        batch_size = 10
        for start in range(0, num_client, batch_size):
            end = min(start + batch_size, num_client)
            for client_id in range(start, end):
                print(f"Client {client_id + 1} Local Training...")
                client_model = copy.deepcopy(global_model[client_id]).to(device)
                train_loader = DataLoader(client_data_splits[client_id][round], batch_size=batch_size, shuffle=True)
                client_state_dict = train_local_model(train_loader, client_model, num_epochs, learning_rate)
                global_model[client_id].load_state_dict(client_state_dict)
            torch.cuda.empty_cache()

        # server model aggregation and averaging
        # vgg11-19
        state_dict = global_model[0].state_dict()
        layer_weight_sum = state_dict['features.0.weight'].clone().to(device)
        layer_bias_sum = state_dict['features.0.bias'].clone().to(device)

        for client_id in range(1, num_client):
            state_dict = global_model[client_id].state_dict()
            weight = state_dict['features.0.weight']
            bias = state_dict['features.0.bias']

            layer_weight_sum += weight
            layer_bias_sum += bias

        avg_weight = layer_weight_sum / num_client
        avg_bias = layer_bias_sum / num_client

        for client_id in range(num_client):
            state_dict = global_model[client_id].state_dict()
            state_dict['features.0.weight'].copy_(avg_weight)
            state_dict['features.0.bias'].copy_(avg_bias)
            global_model[client_id].load_state_dict(state_dict)

        # vgg13-19
        for layer_idx in [2, 5, 7, 10, 12]:
            layer_weight_sum = None
            layer_bias_sum = None
            num_clients = len(range(10, 40))

            for client_id in range(10, 40):
                state_dict = global_model[client_id].state_dict()

                weight_name = f'features.{layer_idx}.weight'
                bias_name = f'features.{layer_idx}.bias'

                weight = state_dict[weight_name]
                bias = state_dict[bias_name]

                if layer_weight_sum is None:
                    layer_weight_sum = weight.clone().to(device)
                    layer_bias_sum = bias.clone().to(device)
                else:
                    layer_weight_sum += weight
                    layer_bias_sum += bias

            avg_weight = layer_weight_sum / num_clients
            avg_bias = layer_bias_sum / num_clients

            for client_id in range(10, 40):
                state_dict = global_model[client_id].state_dict()
                state_dict[weight_name].copy_(avg_weight)
                state_dict[bias_name].copy_(avg_bias)
                global_model[client_id].load_state_dict(state_dict)

        # vgg16-19
        layer_weight_sum = None
        layer_bias_sum = None
        num_clients = len(range(20, 40))

        for client_id in range(20, 40):
            state_dict = global_model[client_id].state_dict()

            weight = state_dict['features.14.weight']
            bias = state_dict['features.14.bias']

            if layer_weight_sum is None:
                layer_weight_sum = weight.clone().to(device)
                layer_bias_sum = bias.clone().to(device)
            else:
                layer_weight_sum += weight
                layer_bias_sum += bias

        avg_weight = layer_weight_sum / num_clients
        avg_bias = layer_bias_sum / num_clients

        for client_id in range(20, 40):
            state_dict = global_model[client_id].state_dict()
            state_dict['features.14.weight'].copy_(avg_weight)
            state_dict['features.14.bias'].copy_(avg_bias)
            global_model[client_id].load_state_dict(state_dict)

        # model evaluation
        print(f"Evaluating client models after round {round + 1}")
        for client_id in range(num_client):
            test_loader = DataLoader(client_test_data_splits[client_id][round], batch_size=batch_size, shuffle=False)
            avg_loss, accuracy = evaluate_model(test_loader, client_model[client_id])
            round_loss.append(avg_loss)
            round_accuracy.append(accuracy)
            print(f"Client {client_id + 1} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        avg_round_loss = sum(round_loss) / len(round_loss)
        avg_round_accuracy = sum(round_accuracy) / len(round_accuracy)
        round_results["loss"].append(avg_round_loss)
        round_results["accuracy"].append(avg_round_accuracy)

        print(f"Round {round + 1} - Average Loss: {avg_round_loss:.4f}, Average Accuracy: {avg_round_accuracy:.2f}%")
        print("Round complete.\n")

    with open("federated_results.json", "w") as f:
        json.dump(round_results, f)

    print("Training complete.")




