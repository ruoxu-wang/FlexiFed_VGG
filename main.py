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
    client_model.to('cpu')
    return client_model.state_dict()

# test
def evaluate_model(test_loader, client_model):
    client_model.eval()
    client_model.to(device)
    # total_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = client_model(images)
            # loss = criterion(outputs, labels)
            # total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    # avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * correct / total
    client_model.to('cpu')
    return accuracy


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
    training_round = 50
    num_epochs = 10
    batch_size = 64
    learning_rate = 0.01
    num_client = 40 # init dataset for 40 client, every 10 client share one model
    # round_results = {"loss": [], "accuracy": []}
    # round_results = {"accuracy": []}
    # round_results = {"VGG11": [], "VGG13": [], "VGG16": [], "VGG19": []}
    round_results = []

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
    test_split_size = test_data_len // num_client
    for i in range(num_client):
        client_test_idx = test_data_idx[i * test_split_size: (i + 1) * test_split_size]
        client_test_datasets.append(Subset(test_dataset, client_test_idx))

    client_test_data_splits = []
    for i in range(num_client):
        client_test_data = client_test_datasets[i]
        split_sizes = [len(client_test_data) // training_round] * training_round
        client_test_split = random_split(client_test_data, split_sizes)
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
        # round_loss = []
        # round_accuracy = []
        vgg11_accuracies = []
        vgg13_accuracies = []
        vgg16_accuracies = []
        vgg19_accuracies = []
        client_accuracies = []

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
        temp_batch_size = 10
        for start in range(0, num_client, temp_batch_size):
            end = min(start + temp_batch_size, num_client)
            for client_id in range(start, end):
                print(f"Client {client_id + 1} Local Training...")
                client_model = copy.deepcopy(global_model[client_id]).to(device)
                train_loader = DataLoader(client_data_splits[client_id][round], batch_size=batch_size, shuffle=True)
                client_state_dict = train_local_model(train_loader, client_model, num_epochs, learning_rate)
                global_model[client_id].load_state_dict(client_state_dict)
                global_model[client_id].to('cpu')
                del client_model
            torch.cuda.empty_cache()

        # server model aggregation and averaging
        # vgg11-19
        layer_weight_sum = None
        layer_bias_sum = None

        for client_id in range(0, num_client):
            state_dict = global_model[client_id].state_dict()

            weight = state_dict['features.0.weight'].to(device)
            bias = state_dict['features.0.bias'].to(device)

            if layer_weight_sum is None:
                layer_weight_sum = weight.clone()
                layer_bias_sum = bias.clone()
            else:
                layer_weight_sum += weight
                layer_bias_sum += bias

        avg_weight = layer_weight_sum / num_client
        avg_bias = layer_bias_sum / num_client
        del layer_weight_sum, layer_bias_sum
        torch.cuda.empty_cache()

        for client_id in range(num_client):
            state_dict = global_model[client_id].state_dict()
            state_dict['features.0.weight'].copy_(avg_weight)
            state_dict['features.0.bias'].copy_(avg_bias)
            global_model[client_id].load_state_dict(state_dict)
            global_model[client_id].to('cpu')

        # vgg13-19
        for layer_idx in [2, 5, 7, 10, 12]:
            layer_weight_sum = None
            layer_bias_sum = None
            num_clients = len(range(10, 40))

            for client_id in range(10, 40):
                state_dict = global_model[client_id].state_dict()

                weight_name = f'features.{layer_idx}.weight'
                bias_name = f'features.{layer_idx}.bias'

                weight = state_dict[weight_name].to(device)
                bias = state_dict[bias_name].to(device)

                if layer_weight_sum is None:
                    layer_weight_sum = weight.clone()
                    layer_bias_sum = bias.clone()
                else:
                    layer_weight_sum += weight
                    layer_bias_sum += bias

            avg_weight = layer_weight_sum / num_clients
            avg_bias = layer_bias_sum / num_clients
            del layer_weight_sum, layer_bias_sum
            torch.cuda.empty_cache()

            for client_id in range(10, 40):
                state_dict = global_model[client_id].state_dict()
                state_dict[weight_name].copy_(avg_weight)
                state_dict[bias_name].copy_(avg_bias)
                global_model[client_id].load_state_dict(state_dict)
                global_model[client_id].to('cpu')

        # vgg16-19
        layer_weight_sum = None
        layer_bias_sum = None
        num_clients = len(range(20, 40))

        for client_id in range(20, 40):
            state_dict = global_model[client_id].state_dict()

            weight = state_dict['features.14.weight'].to(device)
            bias = state_dict['features.14.bias'].to(device)

            if layer_weight_sum is None:
                layer_weight_sum = weight.clone()
                layer_bias_sum = bias.clone()
            else:
                layer_weight_sum += weight
                layer_bias_sum += bias

        avg_weight = layer_weight_sum / num_clients
        avg_bias = layer_bias_sum / num_clients
        del layer_weight_sum, layer_bias_sum
        torch.cuda.empty_cache()

        for client_id in range(20, 40):
            state_dict = global_model[client_id].state_dict()
            state_dict['features.14.weight'].copy_(avg_weight)
            state_dict['features.14.bias'].copy_(avg_bias)
            global_model[client_id].load_state_dict(state_dict)
            global_model[client_id].to('cpu')

        # model evaluation
        print(f"Evaluating client models after round {round + 1}")
        for client_id in range(num_client):
            test_loader = DataLoader(client_test_data_splits[client_id][round], batch_size=batch_size, shuffle=False)
            # avg_loss, accuracy = evaluate_model(test_loader, global_model[client_id])
            accuracy = evaluate_model(test_loader, global_model[client_id])
            # round_loss.append(avg_loss)
            # round_accuracy.append(accuracy)
            if 0 <= client_id < 10:
                vgg11_accuracies.append(accuracy)
                model_name = 'VGG11'
            elif 10 <= client_id < 20:
                vgg13_accuracies.append(accuracy)
                model_name = 'VGG13'
            elif 20 <= client_id < 30:
                vgg16_accuracies.append(accuracy)
                model_name = 'VGG16'
            elif 30 <= client_id < 40:
                vgg19_accuracies.append(accuracy)
                model_name = 'VGG19'

            client_accuracies.append({
                'client_id': client_id + 1,
                'model': model_name,
                'accuracy': accuracy
            })

            # print(f"Client {client_id + 1} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
            print(f"Client {client_id + 1} - Accuracy: {accuracy:.2f}%")
            global_model[client_id].to('cpu')

        # avg_round_loss = sum(round_loss) / len(round_loss)
        # avg_round_accuracy = sum(round_accuracy) / len(round_accuracy)
        avg_accuracy_vgg11 = sum(vgg11_accuracies) / len(vgg11_accuracies)
        avg_accuracy_vgg13 = sum(vgg13_accuracies) / len(vgg13_accuracies)
        avg_accuracy_vgg16 = sum(vgg16_accuracies) / len(vgg16_accuracies)
        avg_accuracy_vgg19 = sum(vgg19_accuracies) / len(vgg19_accuracies)
        # round_results["loss"].append(avg_round_loss)
        # round_results["accuracy"].append(avg_round_accuracy)
        round_results.append({
            'round': round + 1,
            'client_accuracies': client_accuracies,
            'model_averages': {
                'VGG11': avg_accuracy_vgg11,
                'VGG13': avg_accuracy_vgg13,
                'VGG16': avg_accuracy_vgg16,
                'VGG19': avg_accuracy_vgg19
            }
        })

        # print(f"Round {round + 1} - Average Loss: {avg_round_loss:.4f}, Average Accuracy: {avg_round_accuracy:.2f}%")
        # print(f"Round {round + 1} - Average Accuracy: {avg_round_accuracy:.2f}%")
        print(f"Round {round + 1} - Average Accuracy for VGG11: {avg_accuracy_vgg11:.2f}%")
        print(f"Round {round + 1} - Average Accuracy for VGG13: {avg_accuracy_vgg13:.2f}%")
        print(f"Round {round + 1} - Average Accuracy for VGG16: {avg_accuracy_vgg16:.2f}%")
        print(f"Round {round + 1} - Average Accuracy for VGG19: {avg_accuracy_vgg19:.2f}%")
        print("Round complete.\n")


    with open("federated_results_1123.json", "w") as f:
        json.dump(round_results, f, indent=4)

    print("Training complete.")

