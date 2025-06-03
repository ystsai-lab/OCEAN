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

from model.customNet import protoNet_ResNet34, protoNet_ResNet18_
from utils.draw import draw_confusion_matrix, draw_result_and_save
from utils.init_set import *
from utils.our_dataset import OceanDataset, FewshotDataset
from utils.loss_function import ocean_loss, compare_ssl_loss
import eval_ocean_voting


"""
Task OCEAN
    - Backbone: ResNet18
    - Dataset: 
    - Loss Function: OCEAN Loss
    - Fusion Mode: 
    - BEST ALPHA: 0.5 BETA: 0.5

"""
class OceanTrainer:
    def __init__(self):
        self.TASK_NAME = f'ocean_{time.strftime("%m%d-%H%M")}'
        self.TRAIN_EPOCH_N = 300  # ----------------------------> 300
        self.TEST_N = 2000 # ----------------------------> 2000
        self.WAY_N = 5
        self.SHOT_N = 5 # 5
        self.QUERY_N = 15 #q值低
        self.TEST_QUERY_N = 5

        self.TRAINSET = 'tiered_tr_ts'
        self.VALSET = 'tiered_val'
        self.TESTSET  = 'fewshot_coralset'

        self.VALSET_PATH = 'datasets/tieredImagenet/val'
        self.CORALSET_PATH = 'datasets/fewshot_coralset'

        self.LOSSFUNC = ocean_loss
        self.FUSIONMODE = 'mul'

        self.SIZE = 160
        self.ALPHA = 0.5
        self.BETA = 0.5

    # === Train function ===
    def save_list_to_file(self, path, thelist):
        with open(path, 'w') as f:
            for item in thelist:
                f.write("%s\n" % item)

    def training_model_with_ssl(
            self,
            model, optim, lr_scheduler,
            tr_dataloader, val_dataloader=None, 
            epochs=300, device=None, experiment_root = '/output',
            loss_func=ocean_loss,
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
        voting_best_model_path = os.path.join(experiment_root, 'best_voting_model.pth')
        last_model_path = os.path.join(experiment_root, 'last_model.pth')

        trainAcc_list, trainLoss_list = [], []
        train_eucL_list, train_cosL_list, train_varL_list, train_sslL_list = [], [], [], []
        
        valAcc_list, valLoss_list = [], []
        votingAcc_list = []
        best_acc, best_voting_acc = 0, 0

        for epoch in range(epochs):
            # Train
            train_acc, train_loss = self.train_loop(
                model, optim, lr_scheduler, tr_dataloader, device, loss_func=loss_func)
            trainAcc_list.append(train_acc)
            trainLoss_list.append(train_loss['loss'])
            train_eucL_list.append(train_loss['euc_loss'])
            train_cosL_list.append(train_loss['cos_loss'])
            train_sslL_list.append(train_loss['ssl_loss'])
            train_varL_list.append(train_loss['var_loss'])
            print(f"[Epoch: {epoch}] [Train] Acc:{train_acc:.5f}, TotalLoss:{train_loss['loss']:.5f}, sslL:{train_loss['ssl_loss']:.5f}")
            
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
            
            # Validation with Voting
            if epoch%5 == 0:
                classes_label, truth, voted, voting_acc = eval_ocean_voting.voting_system(
                    # N-way K-shot Q-query
                    ways = self.WAY_N,    K = self.SHOT_N,    Q = self.QUERY_N,
                    dataset_path=self.VALSET_PATH,
                    run_times=50,  # ----------------------------> 100
                    device=device,
                    model=model
                )

                voting_postfix = f'(Best: {best_voting_acc:.5f})'
                if voting_acc >= best_voting_acc:
                    best_voting_acc = voting_acc
                    voting_postfix = ' (Best Voting) '+'-'*50
                    torch.save(model.state_dict(), voting_best_model_path)
                print (f'[Voting Val] Acc:{voting_acc}{voting_postfix}')
            votingAcc_list.append(voting_acc)

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
            
            'train_eucloss':train_eucL_list,
            'train_cosloss':train_cosL_list,
            'train_sslloss':train_sslL_list,
            'train_varloss':train_varL_list,

            'voting_acc': votingAcc_list
        }
        return result


    def train_loop(self, model, optim, lr_scheduler, tr_dataloader, device, loss_func):
        """
        每個epoch的訓練過程
        return:
            avg_acc: 平均準確率
            loss_dict: 平均損失 {loss, euc_loss, cos_loss, ssl_loss, var_loss}
        """

        tr_iter = iter(tr_dataloader)
        acc_list, loss_list = [], []
        e_cosloss, e_sslloss, e_varloss, e_eucloss = [], [], [], []
        model.train()
        for batch in tqdm(tr_iter):
            optim.zero_grad()
            x, x_sub1, x_sub2, y = batch

            # proto step
            x, y = x.to(device), y.to(device)
            x_output = model(x)
            main_task_loss = loss_func(x_output, target=y, n_support=5, fusionMode=self.FUSIONMODE)

            # img vs subimg step
            x_sub1, x_sub2 = x_sub1.to(device), x_sub2.to(device)
            x_sub1, x_sub2 = model(x_sub1), model(x_sub2)

            # 子圖之間的相似度
            ssl_sub12 = compare_ssl_loss(x_sub1, x_sub2)
            # 子圖1與原圖之間的相似度
            ssl_sub1o = compare_ssl_loss(x_sub1, x_output)
            # 子圖2與原圖之間的相似度
            ssl_sub2o = compare_ssl_loss(x_sub2, x_output)
            ssl_loss = (ssl_sub12 + ssl_sub1o + ssl_sub2o) / 3

            # # 子圖之間的相似度
            # ssl_loss = compare_ssl_loss_V2(x_output, x_sub1, x_sub2)


            total_loss = self.ALPHA*main_task_loss['loss'] + self.BETA*ssl_loss

            total_loss.backward()
            optim.step()

            acc_list.append(main_task_loss['acc'].item())
            loss_list.append(total_loss.item())
            e_eucloss.append(main_task_loss['eucli_loss_val'].item())
            e_cosloss.append(main_task_loss['cosine_loss_val'].item())
            e_varloss.append(main_task_loss['variance'].item())
            e_sslloss.append(ssl_loss.item())
            # break

        lr_scheduler.step()
        avg_acc = np.mean(acc_list)*100
        loss_dict = {
            'loss': np.mean(loss_list),
            'euc_loss': np.mean(e_eucloss),
            'cos_loss': np.mean(e_cosloss),
            'ssl_loss': np.mean(e_sslloss),
            'var_loss': np.mean(e_varloss)
        }
        return avg_acc, loss_dict


    def test_loop(self, model, device, val_dataloader, loss_func):
        val_iter = iter(val_dataloader)
        model.eval()
        with torch.no_grad():
            acc_list, loss_list = [], []
            for batch in val_iter:
                x, y = batch
                x, y = x.to(device), y.to(device)

                model_output = model(x)
                result = loss_func(model_output, target=y, n_support=5, fusionMode=self.FUSIONMODE)

                acc_list.append(result['acc'].item())
                loss_list.append(result['loss'].item())

            avg_acc, avg_loss = np.mean(acc_list)*100, np.mean(loss_list)
        return avg_acc, avg_loss

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
            # transforms.RandomResizedCrop(size = (size, size), scale=(0.6, 1.0), ratio=(0.8, 1.1)),
            transforms.Resize((self.SIZE, self.SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ])
        train_subimg_transform = transforms.Compose([
            # transforms.RandomResizedCrop(size = (size, size), scale=(0.6, 1.0), ratio=(0.8, 1.1)),
            transforms.Resize((self.SIZE, self.SIZE)),
            transforms.CenterCrop(size=(((self.SIZE*3)//4, (self.SIZE*3)//4))),
            transforms.RandomCrop(size=((self.SIZE//2, self.SIZE//2))),
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
        train_dataset = OceanDataset( csv_path=train_csv_path, transform=train_transform, sub_transform=train_subimg_transform)
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
        # 引入訓練過的模型 ------> # 這裡有需要才用 # 
        # backbone.load_state_dict(torch.load('output/Ocean_A8-B2/last_model.pth')) # ----------------------------> None
        optim = init_optim( backbone)
        lr_scheduler = init_lr_scheduler(optim=optim)

        # =========================== Print Info ===========================
        print (f'[Backbone] {backbone.__class__.__name__}')
        print (f'[Train Dataset] {self.TRAINSET}, categories: {len (train_dataset.wnids)}, size: {len(train_dataset)}')
        print (f'[Val Dataset] {self.VALSET}, categories: {len (val_dataloader.dataset.wnids)}, size: {len(val_dataloader.dataset)}')
        print (f'[Test Dataset] {self.TESTSET}, categories: {len (test_dataloader.dataset.wnids)}, size: {len(test_dataloader.dataset)}')
        
        star_time = time.time()
        
        # =========================== Training ===========================
        res = self.training_model_with_ssl(
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
            dataset_path=self.CORALSET_PATH,
            run_times=self.TEST_N,
            device=device,
            model=backbone
        )
        draw_confusion_matrix(truth, voted, classes_label, save_path=os.path.join(experiment_root, f'cm_best-model_{self.WAY_N}W-{self.SHOT_N}S.png'))

        # 輸出訓練紀錄圖
        draw_result_and_save(experiment_root, res, test_best_acc, test_last_acc, task_name=self.TASK_NAME)


        # Test with BEST Voting model
        backbone.load_state_dict(torch.load(os.path.join(experiment_root, 'best_voting_model.pth')))
        classes_label, truth, voted, acc = eval_ocean_voting.voting_system(
            # N-way K-shot Q-query
            ways = self.WAY_N,   K = self.SHOT_N,    Q = self.TEST_QUERY_N,
            dataset_path=self.CORALSET_PATH,
            run_times=self.TEST_N,
            device=device,
            model=backbone
        )
        draw_confusion_matrix(truth, voted, classes_label, save_path=os.path.join(experiment_root, f'cm_best-voting_{self.WAY_N}W-{self.SHOT_N}S.png'))


        print (f'Eclipsed time: {(time.time() - star_time)//3600} hours {(time.time() - star_time)%3600//60} minutes')
        print(f'Done! Check out results in {experiment_root}')
