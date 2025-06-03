import torch
import torch.nn.functional as F



def prototypical_loss(input, target, n_support, isVar=False):
    '''
    prototypical loss function

    '''
    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')

    def supp_idxs(c):
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)
    
    classes = torch.unique(target_cpu)
    n_classes = len(classes)
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support

    support_idxs = list(map(supp_idxs, classes))
    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)

    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])

    query_samples = input.to('cpu')[query_idxs]

    target_inds = torch.zeros(n_classes*n_query, n_classes)
    for i in range(n_classes):
        target_inds[i*n_query:(i+1)*n_query, i] = 1

    # 每個類別的樣本與該類別的原型之間的歐幾里德距離
    # 計算完歐幾里德距離後，使用log_softmax函數將其轉換為概率，並將其添加到我們的損失中。
    dists = euclidean_dist(query_samples, prototypes)
    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)
    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()
    eucli_loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

    _, predicted = log_p_y.max(2)
    acc = predicted.eq(target_inds.squeeze(2)).float().mean()

    variance = torch.var(prototypes, dim=0).mean()
    # 計算 "總損失"
    cosine_loss_val = 0
    total_loss = eucli_loss_val+cosine_loss_val
    if isVar:
        # 計算變異數，dim=0 表示沿著堆疊的維度（即第一維度）計算
        total_loss = total_loss/variance
    
    return {
        'loss': total_loss, 
        'eucli_loss_val': eucli_loss_val, 
        'feature_loss_val': cosine_loss_val, 
        'variance': variance,
        'acc': acc, 
        'classes': classes,
        'y_hat': predicted, 
        'prototypes': prototypes
    }

def ocean_loss(input, target, n_support, isVar=False, fusionMode='add'):
    '''
    OCEAN 使用歐幾里德距離與餘弦相似度混合做為準確度指標的損失函數

    loss = -log(softmax(-euclidean_dist * (1-cosine_prob)))

    '''
    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')

    def supp_idxs(c):
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)
    
    classes = torch.unique(target_cpu)
    n_classes = len(classes)
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support

    support_idxs = list(map(supp_idxs, classes))
    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)
    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])

    query_samples = input.to('cpu')[query_idxs]

    # 計算每個查詢樣本與每個原型向量之間的餘弦相似度與歐基理德距離。cosine_sim:[40, 8], euclidean_sim[40, 8]
    cosine_sim = calculate_cosine_similarity(prototypes, query_samples)
    euclidean_sim = euclidean_dist(query_samples, prototypes)

    # 把cosine_sim 轉換為"機率分佈"
    cosine_prob = torch.softmax(cosine_sim, dim=1)
    # 轉換成 "別類"的機率
    cosine_prob = 1 - cosine_prob

    # euclidean_sim與cosine_prob相乘
    eucXcos_sim = (euclidean_sim * cosine_prob)
    sim = F.log_softmax(-eucXcos_sim, dim=1).view(n_classes, n_query, -1)

    target_idxs = torch.arange(0, n_classes)
    target_idxs = target_idxs.view(n_classes, 1, 1)
    target_idxs = target_idxs.expand(n_classes, n_query, 1).long()

    total_loss = -sim.gather(2, target_idxs).squeeze().view(-1).mean()

    _, predicted = sim.max(2)
    acc = predicted.eq(target_idxs.squeeze(2)).float().mean()

    variance = torch.tensor (0)
    cosine_loss_val = torch.tensor (0)
    eucli_loss_val = torch.tensor (0)

    return {
        'loss': total_loss, 
        'eucli_loss_val': eucli_loss_val, 
        'cosine_loss_val': cosine_loss_val, 
        'variance': variance,
        'acc': acc, 
        'classes': classes,
        'y_hat': predicted, 
        'prototypes': prototypes
    }

def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors (samples and prototypes)

    Parameters:
    x: Sample tensor [N, D]
    y: Prototypes tensor [M, D]
    
    Returns:
    dist: Euclidean distance [N, M]

    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - y, 2).sum(2)

def calculate_cosine_similarity(prototypes, query_samples):
    '''
    calculate cosine similarity between prototypes and query samples
    args:
        prototypes: [N, D]
        query_samples: [N, D]
    return:
        cosine_sim: [N, N]
    '''
    # 標準化原型向量和查詢樣本
    prototypes_norm = F.normalize(prototypes, p=2, dim=1)
    query_samples_norm = F.normalize(query_samples, p=2, dim=1)

    # 計算餘弦相似度
    cosine_sim = torch.mm(query_samples_norm, prototypes_norm.t())

    return cosine_sim


def compare_subimg_loss(x1, x2):
    """
        x1 和 x2 是模型的輸出特徵 [B, N]
        log_softmax( eculudean)
    """
    devive = x1.device
    dists = euclidean_dist(x1, x2)
    
    log_p_y = F.log_softmax(-dists, dim=1)
    target_inds = torch.arange(0, log_p_y.shape[0])
    target_inds = target_inds.view(log_p_y.shape[0], 1).to(devive)
    eucli_loss_val = -log_p_y.gather(1, target_inds).squeeze().view(-1).mean()
    
    return eucli_loss_val


def compare_ssl_loss(x1, x2):
    """
        x1 和 x2 是模型的輸出特徵 [B, N]
        log_softmax( eculudean)
    """
    n_query = x1.shape[0]
    
    cosine_sim = calculate_cosine_similarity(x1, x2)
    euclidean_sim = euclidean_dist(x1, x2)

    # 把cosine_sim 轉成"機率分佈"
    cosine_prob = torch.softmax(cosine_sim, dim=1)
    # 轉換成 "別類"的機率
    cosine_prob = 1 - cosine_prob
    # euclidean_sim與cosine_prob相乘
    eucXcos_sim = (euclidean_sim * cosine_prob)
    sim = F.log_softmax(-eucXcos_sim, dim=1).view(n_query, n_query, -1)

    # 取得sim對角線的值
    diagonal_values = -torch.diagonal(sim, offset=0, dim1=0, dim2=1)
    loss_val = diagonal_values.mean()

    return loss_val


def compare_ssl_loss_V2(origin, x1, x2):
    """
        x1 和 x2 是模型的輸出特徵 [B, N]
        log_softmax( eculudean)
        
        
        使用原型的概念
    """
    n_query = x1.shape[0]
    # 子圖的原型 x1與x2的平均
    sub_prototype = (x1 + x2) / 2
    
    cosine_sim = calculate_cosine_similarity(origin, sub_prototype)
    euclidean_sim = euclidean_dist(origin, sub_prototype)

    # 把cosine_sim 轉成"機率分佈"
    cosine_prob = torch.softmax(cosine_sim, dim=1)
    # 轉換成 "別類"的機率
    cosine_prob = 1 - cosine_prob
    # euclidean_sim與cosine_prob相乘
    eucXcos_sim = (euclidean_sim * cosine_prob)
    sim = F.log_softmax(-eucXcos_sim, dim=1).view(n_query, n_query, -1)

    # 取得sim對角線的值
    diagonal_values = -torch.diagonal(sim, offset=0, dim1=0, dim2=1)
    loss_val = diagonal_values.mean()

    return loss_val