
from trainer.ocean import OceanTrainer
from trainer.protonet import ProtoNetTrainer
from model.resNet import ResNet18

backbone = ResNet18(pretrained=False)

if __name__ == '__main__':
    train_task = OceanTrainer()
    
    # train_task = ProtoNetTrainer()
    train_task.TRAIN_EPOCH_N = 300 #300
    train_task.TEST_N = 2000 #2000
    train_task.run()
