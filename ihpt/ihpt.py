# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

ihvit module

@author: tadahaya
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import yaml
from tqdm.auto import tqdm

from .src.model import PointNetClassHead
from .src.utils import save_experiment, load_experiment
from .src.trainer import Trainer
from .src.data_handler import prep_data


class IhPointNet:
    def __init__(self, config=None, config_path=None):
        # config
        if config is None:
            if config_path is not None:
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
            else:
                config = dict()
        default_config = {
            # backbone config
            "input_dim": 3,
            "num_points": 256,
            "dim_global_feats": 128, # 1024
            "local_feats": False,
            "dropout_ratio": 0.3,
            # trainer config
            "exp_name": "experiment",
            "base_dir": None,
            "epochs": 20,
            "batch_size": 64,
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
        self.config = {**default_config, **config}
        # model
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.model = PointNetClassHead(self.config)
        self.trainer = Trainer(self.model, self.config)


    def prep_data(
            self, x_train, y_train=None, x_test=None, y_test=None,
            train:bool=True
            ):
        """
        data preparation
        
        Parameters
        ----------
        x_train: np.array
            training data or all data, (batch_size, num_points, input_dim)
        
        """
        train_loader, test_loader = prep_data(
            x_train, y_train, x_test, y_test,
            num_points=self.config["num_points"], batch_size=self.config["batch_size"],
            train=train
            )
        return train_loader, test_loader


    def fit(self, train_loader, test_loader):
        """ training """
        # training
        train_losses, test_losses, accuracies = self.trainer.train(train_loader, test_loader)
        # save experiment
        save_experiment(
            self.config["exp_name"], self.config["base_dir"], self.config,
            self.model, train_losses, test_losses, accuracies
            )


    @torch.no_grad()
    def evaluate(self, test_loader, exp_name:str=None, base_dir:str=None):
        """ evaluation """
        test_losses, accuracies = self.trainer.evaluate(test_loader)
        # save experiment
        if exp_name is None:
            exp_name = "evaluation"
        if base_dir is None:
            base_dir = self.config["base_dir"]
        save_experiment(
            exp_name, self.config["base_dir"], self.config,
            self.model, [], test_losses, accuracies
            )


    def load_model(self, exp_name, base_dir):
        """
        load model
        
        Parameters
        ----------
        exp_name: str
            experiment name
        
        base_dir: str
            base directory path
        
        """
        self.config, cpfile, _, _, _ = load_experiment(
            exp_name, base_dir
            )
        self.model = PointNetClassHead(self.config)
        self.model.load_state_dict(torch.load(cpfile))
        self.trainer = Trainer(self.model, self.config)


    @torch.no_grad()
    def predict(self, X):
        """
        predict the class labels
        
        Parameters
        ----------
        X: np.array
            input data, (batch_size, num_points, input_dim)
                
        Returns
        -------
        predictions: np.array
            predicted class labels
        
        probs: np.array
            predicted probabilities
        
        indices: np.array
            critical indices
                
        """
        self.model.eval()
        # data loading
        testloader, _ = prep_data(
            X, num_points=self.config["num_points"], batch_size=self.config["batch_size"],
            train=False
            )
        # prediction
        predictions = []
        probs = []
        indices = []
        for data in testloader:
            # batchをdeviceへ
            data = data.to(self.device)
            # 予測
            output, idx = self.model(data)
            probs.append(output)
            pred = torch.argmax(output, dim=1)
            predictions.append(pred)
            indices.append(idx)
        predictions = torch.cat(predictions, dim=0).cpu().numpy()
        probs = torch.cat(probs, dim=0).cpu().numpy()
        return predictions, probs, indices


    @torch.no_grad()
    def get_latent(self, X, return_idx=False):
        raise NotImplementedError("Not implemented yet")