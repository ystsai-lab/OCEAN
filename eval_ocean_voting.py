
import os
import random
from PIL import Image

import torch
from torchvision import transforms

from model.customNet import protoNet_ResNet18_, protoNet_ResNet34
from utils.loss_function import calculate_cosine_similarity, euclidean_dist


# 把珊瑚prototype投票改好


def random_choice_classes(dataset_path, numbers=5) -> dict:
    """
        隨機選擇數個類別
        Parameters:
        dataset_path(str): 資料集路徑
        numbers(int): 選擇的類別數量, default=5

        Returns:
        labels(list): 選擇的類別
    """
    labels = random.sample(
        [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))],
        numbers
    )
    return labels


def random_choice_classes_sample(dataset_path, labels, numbers=5) -> dict:
    """
        隨機選擇數個類別的樣本
        Parameters:
        dataset_path(str): 資料集路徑
        labels(list): 要選擇的類別
        numbers(int): 每個類別選擇的樣本數量, default=5

        Returns:
        samples(dict{str: list}): 樣本 {label: [img1, img2, ...]}
    """

    samples = {}

    for label in labels:
        classes_dir = os.path.join(dataset_path, label)
        image_paths = random.sample(os.listdir(classes_dir), numbers)
        samples[label] = [
            Image.open(os.path.join(classes_dir, path)) for path in image_paths
        ]
    return samples

def crop_local_image(image, num_patch = 4):
    """
    裁切局部圖塊，分成多種模式 4 塊或 5 塊

    Parameters:
    image(PIL.Image): 圖片
    split_numbers(int): 切分數量, default=4

    Returns:
    images(list): 切分後的圖片 [Origin_image, ,,,]
    
    """
    
    width, height = image.size

    # 計算分割線
    center_x, center_y = width // 2, height // 2

    # 切割圖片
    top_left = image.crop((0, 0, center_x, center_y))
    top_right = image.crop((center_x, 0, width, center_y))
    bottom_left = image.crop((0, center_y, center_x, height))
    bottom_right = image.crop((center_x, center_y, width, height))

    if num_patch == 1:
        left = center_x - center_x // 2
        top = center_y - center_y // 2
        right = center_x + center_x // 2
        bottom = center_y + center_y // 2
        center = image.crop((left, top, right, bottom))
        return image, center
    
    if num_patch == 4:
        return image, top_left, top_right, bottom_left, bottom_right
    
    if num_patch == 5:
        left = center_x - center_x // 2
        top = center_y - center_y // 2
        right = center_x + center_x // 2
        bottom = center_y + center_y // 2
        center = image.crop((left, top, right, bottom))
        return image, top_left, top_right, bottom_left, bottom_right, center

def handel_set_crop(samples, num_patch = 5, isCrop = True):
    """
    處理資料集的切割
    
    """
    if isCrop:
        for label in samples:
            imgs = []
            for image in samples[label]:
                imgs += crop_local_image(image, num_patch)
            samples[label]=imgs
    
    return samples

def get_features(model, images, transform, device):
    """
    取得圖片特徵
    
    Parameters:
    model(torch.nn.Module): 模型
    images(list): 圖片
    transform(torchvision.transforms): 轉換
    device(torch.device): 設備
    
    Returns:
    features(torch.Tensor): 特徵
    """
    model.eval()
    features = []
    with torch.no_grad():
        for image in images:
            image = transform(image).unsqueeze(0).to(device)
            feature = model(image)
            features.append(feature)
    return torch.cat(features, dim=0)


def voting_system( 
        model , 
        device, 
        dataset_path,
        ways = 5, 
        K = 5, 
        Q = 5, 
        run_times = 100, 
        isCrop_support = True, 
        isCrop_query = True, 
        sim_method = 'euclidean-cosine',
        num_patch = 5
    ):
    print (f"[voting_system] run_times: {run_times}, dataset_path: {dataset_path}")
    print (f"[voting_system] way: {ways}, Kshot: {K}, query: {Q}")
    print (f"[voting_system] isCrop_support: {isCrop_support}, isCrop_query: {isCrop_query}")
    print (f"[voting_system] sim_method: {sim_method}")

    all_classes_label = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    # 定義轉換
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
    truth = []
    voted = []
    correct = 0

    for _run in range(run_times):
        # 選擇N個類別
        test_labels = random.sample(all_classes_label, ways)
        # 支持集：所有類別隨機選擇 5張
        support = random_choice_classes_sample(dataset_path, labels=test_labels, numbers=K)
        # 查詢集：所有類別隨機選擇 1張
        query = random_choice_classes_sample(dataset_path, labels=test_labels, numbers=Q)
        # 裁切圖片： origin -> origin + 5 patch
        support = handel_set_crop(support, num_patch=num_patch, isCrop=isCrop_support)

        # 取得特徵
        support_features = {}
        for label in support:
            support_features[label] = get_features(model, support[label], transform, device)
        
        # 計算 prototype
        classes_proto = {}
        for label in support_features:
            classes_proto[label] = support_features[label].mean(dim=0)
        prototypes = torch.stack(list(classes_proto.values()))
        
        # 投票評估
        for label in query:
            for anchor_image in query[label]:
                ballots = []
                if isCrop_query:
                    images = crop_local_image(anchor_image, num_patch=num_patch)
                else:
                    images = [anchor_image]
                img_features = get_features(model, images, transform, device)

                for feature in img_features:
                    feature = feature.unsqueeze(0)
                    if sim_method == 'cosine':
                        sim = calculate_cosine_similarity(prototypes, feature)
                        _, simest_idx = torch.max(sim, dim=1)

                    elif sim_method == 'euclidean':
                        sim = euclidean_dist(prototypes, feature)
                        _, simest_idx = torch.min(sim, dim=0)
                        

                    elif sim_method == 'euclidean-cosine':
                        sim_cos = calculate_cosine_similarity(prototypes, feature)
                        sim_euc = euclidean_dist(prototypes, feature).view(1, -1)

                        # 把sim_cos 轉換為 "其他類別"的概率
                        sim_cos = 1-torch.softmax(sim_cos, dim=1)

                        sim = sim_cos * sim_euc
                        _, simest_idx = torch.min(sim, dim=1)
                        
                    else:
                        sim = euclidean_dist(prototypes, feature)
                        _, simest_idx = torch.min(sim, dim=0)
                    
                    ballots.append(simest_idx.item())
                # 最多票的
                elected = max(set(ballots), key=ballots.count)
                truth.append(label)
                voted.append(test_labels[elected])
                if label == test_labels[elected]:
                    correct += 1
        if _run % 5 == 0:
            acc = (correct / ((_run+1)*Q*ways))*100
        print (f"Run: {_run+1}, Acc:{acc:.4f}%, Classes: {test_labels}", end="\r")
    
    return all_classes_label, truth, voted, acc
