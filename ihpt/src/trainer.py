# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

trainer

@author: tadahaya
"""
import torch

from .utils import save_checkpoint, make_optimizer, make_loss_fn

class Trainer:
    def __init__(self, model, config):
        # config
        default_config = {
            "epochs": 20,
            "save_model_every": 10,
            "optimizer": {
                "name": "AdamW",
                "lr": 1e-3,
                "weight_decay": 1e-2,
            },
            "loss_fn": {
                "name": "CrossEntropyLoss",
                "label_smoothing": 0.1,
            },
        }
        self.config = {**default_config, **config} # merge the two dictionaries
        # for readability
        self.exp_name = self.config["exp_name"]
        self.base_dir = self.config["base_dir"]
        self.epochs = self.config["epochs"]
        self.save_model_every = self.config["save_model_every"]
        # models
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.model = model.to(self.device)
        self.optimizer = make_optimizer(self.model.parameters(), **self.config["optimizer"])
        self.loss_fn = make_loss_fn(**self.config["loss_fn"])


    def get_model(self):
        return self.model
    

    def set_model(self, model):
        self.model = model
        # update
        self.optimizer = make_optimizer(
            self.model.parameters(), **self.config["optimizer"]
            )


    def train(self, trainloader, testloader):
        """
        train the model for the specified number of epochs.
        No save version

        """
        # keep track of the losses and accuracies
        train_losses, test_losses, accuracies = [], [], []
        # training
        for i in range(self.epochs):
            train_loss = self.train_epoch(trainloader)
            accuracy, test_loss = self.evaluate(testloader) # return accuracy and loss
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracies.append(accuracy)
            print(
                f"Epoch: {i + 1}, Train_loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}"
                )
            if self.save_model_every > 0 and (i + 1) % self.save_model_every == 0 and i + 1 != self.epochs:
                print("> Save checkpoint at epoch", i + 1)
                save_checkpoint(self.exp_name, self.model, i + 1, self.base_dir)
        return train_losses, test_losses, accuracies # accuracyは空


    def train_epoch(self, trainloader):
        """ train the model for one epoch """
        self.model.train()
        total_loss = 0
        for data, label in trainloader:
            # batchをdeviceへ
            data, label = data.to(self.device), label.to(self.device)
            # 勾配を初期化
            self.optimizer.zero_grad()
            # forward
            output, _ = self.model(data) # x, critical indices
            # loss
            loss = self.loss_fn(output, label)
            # backpropagation
            loss.backward()
            # パラメータ更新
            self.optimizer.step()
            total_loss += loss.item() * len(data) # loss is averaged over the batch
        return total_loss / len(trainloader.dataset) # 全データセットのうちのいくらかという比率になっている


    @torch.no_grad()
    def evaluate(self, testloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        for data, label in testloader:
            # batchをdeviceへ
            data, label = data.to(self.device), label.to(self.device)
            # 予測
            output, _ = self.model(data)
            # # lossの計算
            loss = self.loss_fn(output, label)
            total_loss += loss.item() * len(data)
            # # accuracyの計算
            predictions = torch.argmax(output, dim=1)
            correct += torch.sum(predictions == label).item()
        accuracy = correct / len(testloader.dataset) # SSLでは不要
        avg_loss = total_loss / len(testloader.dataset)
        # return accuracy, avg_loss
        return accuracy, avg_loss