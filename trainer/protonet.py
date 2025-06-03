# coding=utf-8
import sys

sys.path.append('..')

import os
from tqdm import tqdm
import time
import numpy as np
import torch
import torchinfo
from torchvision import transforms

from model.customNet import protoNet_ResNet18_
from utils.draw import draw_confusion_matrix, draw_result_and_save
from utils.init_set import *
from utils.loss_function import prototypical_loss

from utils.our_dataset import OceanDataset, FewshotDataset

import eval_ocean_voting

"""
Task ProtoNet:
    ProtoNet 

    1. Dataset: 
    2. Model: protoNet_ResNet18_
    3. Train: 300 epochs
    4. Loss: prototypical_loss
"""

class ProtoNetTrainer:
    def __init__(self):
        self.TASK_NAME = f'protonet_{time.strftime("%m%d-%H%M")}'
        self.TRAIN_EPOCH_N = 300  # ----------------------------> 300
        self.TEST_N = 2000 # ----------------------------> 2000
        self.WAY_N = 5
        self.SHOT_N = 5 
        self.QUERY_N = 5 
        self.TEST_QUERY_N = 5 

        self.TRAINSET = 'tiered_tr_ts'
        self.VALSET = 'tiered_val'
        self.TESTSET  = 'fewshot_coralset'

        self.CORALSET_PATH = 'datasets/fewshot_coralset'
        
        self.LOSSFUNC = prototypical_loss
        self.SIZE = 160


    def save_list_to_file(self, path, thelist):
        with open(path, 'w') as f:
            for item in thelist:
                f.write("%s\n" % item)

    

    def training(
            self,
            model, optim, lr_scheduler,
            tr_dataloader, val_dataloader=None, 
            epochs=300, device=None, experiment_root = '/output',
            loss_func=prototypical_loss,
            ):
        '''
        args:
            model: model to be trained
            optim: optimizer
            lr_scheduler: learning rate scheduler
            tr_dataloader: training dataloader
            val_dataloader: validation dataloader
            epochs: number of epochs to train
            device: device to run the model on
            experiment_root: root directory to save the model
        
        return:
            {
                'best_state': best_state,
                'best_acc': best_acc,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            }
        '''
        if device is None:
            raise Exception('device is None')

        if val_dataloader is None:
            best_state = None
        
        best_model_path = os.path.join(experiment_root, 'best_model.pth')
        last_model_path = os.path.join(experiment_root, 'last_model.pth')

        trainAcc_list, trainLoss_list = [], []
        valAcc_list, valLoss_list = [], []
        best_acc = 0

        for epoch in range(epochs):
            # Train
            train_acc, train_loss = self.train_loop(
                model, optim, lr_scheduler, tr_dataloader, device, loss_func=loss_func)
            trainAcc_list.append(train_acc)
            trainLoss_list.append(train_loss)

            print(f"[Epoch: {epoch}] [Train] Acc:{train_acc:.5f}, TotalLoss:{train_loss:.5f}")
            
            if val_dataloader is None:
                continue
            # Validation
            val_acc, val_loss = self.test_loop(model, device, val_dataloader, loss_func=loss_func)
            valAcc_list.append(val_acc)
            valLoss_list.append(val_loss)
            postfix = f' (Best:{best_acc:.5f})'
            if val_acc >= best_acc:
                postfix = ' (Best)'
                best_acc = val_acc
                best_state = model.state_dict()
                torch.save(model.state_dict(), best_model_path)
            print(f'[Validation] Loss:{val_loss:.5f}, Acc:{val_acc:.5f}{postfix}')
            

            torch.save(model.state_dict(), last_model_path)
        
        for name in ['trainAcc_list', 'trainLoss_list', 'valAcc_list', 'valLoss_list']:
            self.save_list_to_file(os.path.join(experiment_root, name + '.txt'), locals()[name])
        
        result = {
            'best_state': best_state,
            'best_acc': best_acc,
            'train_acc': trainAcc_list,
            'train_loss': trainLoss_list,
            'val_acc': valAcc_list,
            'val_loss': valLoss_list,
        }
        return result
    

    def train_loop(
            self,
            model, optim, lr_scheduler, 
            tr_dataloader, 
            device, 
            loss_func
            ):
        
        tr_iter = iter(tr_dataloader)
        e_acc, e_loss = [], []
        e_floss, e_simloss, e_varloss, e_eucloss = [], [], [], []
        model.train()
        for batch in tqdm(tr_iter):
            optim.zero_grad()
            x, y = batch

            # proto step
            x, y = x.to(device), y.to(device)
            x_output = model(x)
            loss_result = loss_func(x_output, target=y, n_support=5)

            total_loss = loss_result['loss']

            total_loss.backward()
            optim.step()

            e_acc.append(loss_result['acc'].item())
            e_loss.append(total_loss.item())
            e_floss.append(0)
            e_simloss.append(0)
            e_eucloss.append(0)
            e_varloss.append(0)
            # break

        lr_scheduler.step()
        avg_acc, avg_loss  = np.mean(e_acc), np.mean(e_loss)
        
        return avg_acc, avg_loss

    def test_loop(self, model, device, val_dataloader, loss_func):
        val_iter = iter(val_dataloader)
        model.eval()
        with torch.no_grad():
            e_acc, e_loss = [], []
            for batch in val_iter:
                x, y = batch
                x, y = x.to(device), y.to(device)

                model_output = model(x)
                loss_result = loss_func(model_output, target=y, n_support=5)
                e_acc.append(loss_result['acc'].item())
                e_loss.append(loss_result['loss'].item())

            avg_acc, avg_loss = np.mean(e_acc), np.mean(e_loss)
        return avg_acc,avg_loss
    

    def run(self):
        # Initialize seed
        init_seed()

        if not torch.cuda.is_available():
            raise Exception('CUDA not available, but required')
        device = torch.device('cuda:0')
        print(f"CUDA is Used, Device: {device}")

        experiment_root = f'output/{self.TASK_NAME}'
        if not os.path.exists(experiment_root):
            os.makedirs(experiment_root)
        print (f'Experiment Root: {experiment_root}')

        # ======================= Transform =======================
        train_transform = transforms.Compose([
            transforms.Resize((self.SIZE, self.SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ])
        test_transform = transforms.Compose([
            transforms.Resize((self.SIZE, self.SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ])


        # ======================= Dataset, Dataloader =======================
        train_csv_path = f'datasets/{self.TRAINSET}.csv'
        train_dataset = FewshotDataset( csv_path=train_csv_path, transform=train_transform)
        train_sampler = init_sampler(labels=train_dataset.y, way=self.WAY_N, support=self.SHOT_N, query=self.QUERY_N, iterations=100)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler)

        val_csv_path = f'datasets/{self.VALSET}.csv'
        val_dataset = FewshotDataset( csv_path=val_csv_path, transform=test_transform)
        val_sampler = init_sampler(labels=val_dataset.y, way=self.WAY_N, support=self.SHOT_N, query=self.TEST_QUERY_N, iterations=100)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_sampler=val_sampler)

        test_csv_path = f'datasets/{self.TESTSET}.csv'
        test_dataset = FewshotDataset( csv_path=test_csv_path, transform=test_transform)
        test_sampler = init_sampler(labels=test_dataset.y, way=self.WAY_N, support=self.SHOT_N, query=self.TEST_QUERY_N, iterations=100)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_sampler)

        # ================ Model, Optimizer, LR_Scheduler ================
        backbone = protoNet_ResNet18_(isPretrained=True).to(device)
        optim = init_optim( backbone)
        lr_scheduler = init_lr_scheduler(optim=optim)


        # =========================== Print Info ===========================
        print (f'[Backbone] {backbone.__class__.__name__}')
        print (f'[Train Dataset] {self.TRAINSET}, categories: {len (train_dataset.wnids)}, size: {len(train_dataset)}')
        print (f'[Val Dataset] {self.VALSET}, categories: {len (val_dataloader.dataset.wnids)}, size: {len(val_dataloader.dataset)}')
        print (f'[Test Dataset] {self.TESTSET}, categories: {len (test_dataloader.dataset.wnids)}, size: {len(test_dataloader.dataset)}')
        
        star_time = time.time()


        # =========================== Training ===========================
        res = self.training(
            model=backbone,
            optim=optim,
            lr_scheduler=lr_scheduler,
            tr_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            experiment_root=experiment_root,
            device=device,
            epochs=self.TRAIN_EPOCH_N,
            loss_func=self.LOSSFUNC
        )


        # =========================== Testing ===========================


        # test with LAST model
        _, _, _, test_last_acc = eval_ocean_voting.voting_system(
            # N-way K-shot Q-query
            ways = self.WAY_N,   K = self.SHOT_N,    Q = self.TEST_QUERY_N,
            isCrop_query= False, isCrop_support= False,
            dataset_path=self.CORALSET_PATH,
            run_times=self.TEST_N,
            device=device,
            model=backbone
        )
        # test with BEST model
        backbone.load_state_dict(torch.load(os.path.join(experiment_root, 'best_model.pth')))
        classes_label, truth, voted, test_best_acc = eval_ocean_voting.voting_system(
            # N-way K-shot Q-query
            ways = self.WAY_N,   K = self.SHOT_N,    Q = self.TEST_QUERY_N,
            isCrop_query= False, isCrop_support= False,
            dataset_path=self.CORALSET_PATH,
            run_times=self.TEST_N,
            device=device,
            model=backbone
        )
        draw_confusion_matrix(truth, voted, classes_label, save_path=os.path.join(experiment_root, f'cm_best-model_{self.WAY_N}W-{self.SHOT_N}S.png'))

        # 輸出訓練紀錄圖
        draw_result_and_save(experiment_root, res, test_best_acc, test_last_acc, task_name=self.TASK_NAME)


        print (f'Eclipsed time: {(time.time() - star_time)//3600} hours {(time.time() - star_time)%3600//60} minutes')
        print(f'Done! Check out results in {experiment_root}')
