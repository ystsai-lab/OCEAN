import os
import torch
from model.customNet import protoNet_ResNet18_
import eval_ocean_voting
from utils.draw import draw_confusion_matrix

# experiment_root = 'output/ocean_0807-1631' #'/home/lab705/code/Few-shot/output/OCEAN'

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# backbone = protoNet_ResNet18_().to(device)
# backbone.load_state_dict(torch.load(os.path.join(experiment_root, 'best_model.pth')))

# classes_label, truth, voted, test_best_acc = eval_ocean_voting.voting_system(
#     # N-way K-shot Q-query
#     ways = 5,   K = 5,    Q = 5,
#     isCrop_query= True, isCrop_support= True,
#     dataset_path='datasets/fewshot_coralset',
#     run_times=100,
#     device=device,
#     model=backbone
# )
# draw_confusion_matrix(truth, voted, classes_label, save_path=os.path.join(experiment_root, f'cm_best-model_{self.WAY_N}W-{self.SHOT_N}S.png'))



experiment_root = '/output/ocean_...'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
backbone = protoNet_ResNet18_().to(device)
backbone.load_state_dict(torch.load(os.path.join(experiment_root, 'best_model.pth')))

classes_label, truth, voted, test_best_acc = eval_ocean_voting.voting_system(
    # N-way K-shot Q-query
    ways = 5,   K = 5,    Q = 5,
    isCrop_query= True, isCrop_support= True, #isCrop_query= False, isCrop_support= False,投票裁切
    dataset_path='datasets/fewshot_coralset',
    run_times=1000,
    device=device,
    model=backbone,
    sim_method='euclidean' #eudlidean
)


# 計算準確度，先初始化
class_acc = {}
tatal_correct = 0
for label in classes_label:
    class_acc[label]={ k:0 for k in classes_label}
for t, v in zip(truth, voted):
    class_acc[t][v]+=1
    if t == v:
        tatal_correct += 1
        
for label in ['acro', 'agar', 'euph', 'fung', 'lobo', 'meli', 'meru', 'poci', 'sarc', 'sidr']:
    precision = class_acc[label][
        label] / sum(class_acc[label].values())
    recall = class_acc[label][
        label] / sum([class_acc[k][label] for k in classes_label])
    f1 = 2 * (precision * recall) / (precision + recall)
    print (f'{label}: Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    

draw_confusion_matrix(truth, voted, classes_label, save_path=os.path.join(experiment_root, f'cm_test_best-model_5W-5S.png'))
