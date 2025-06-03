import os
import matplotlib.pyplot as plt
import numpy as np


def draw_result_and_save(experiment_root, result, best_test, last_test, task_name):
    """
    
    """
    print (f'[draw_result_and_save] experiment_root: {experiment_root}')

    train_loss = result['train_loss']
    train_acc = result['train_acc']
    val_loss = result['val_loss']
    val_acc = result['val_acc']
    fig = plt.figure(dpi=400)
    plt.rcParams.update({'font.size': 12})
    ax = fig.add_axes([0.1,0.3,0.85,0.6])
    ax.plot(train_acc, label='Train accuracy', color="b")
    ax.plot(val_acc, label='Val accuracy', color="r")
    # ax.plot(v_loss,label='Validation Loss')
    ax.text(
        0.6, -0.2, 
        f'Train Acc: ({max(train_acc):.4f}, { np.mean(train_acc):.4f})', 
        # horizontalalignment='center',
        verticalalignment='center',
        transform=ax.transAxes, fontsize=10
    )
    ax.text(
        0.6, -0.26, 
        f'Validation Acc: ({max(val_acc):.4f}, { np.mean(val_acc):.4f})', 
        # horizontalalignment='center',
        verticalalignment='center',
        transform=ax.transAxes, fontsize=10
    )
    ax.text(
        0.6, -0.34,  
        f'B/L Model Test: { best_test:.4f}/{ last_test:.4f}', 
        # horizontalalignment='center',
        verticalalignment='center',
        transform=ax.transAxes, fontsize=10
    )

    if 'voting_acc' in result:
        ax.plot(result['voting_acc'], label='voting accuracy', color="#785578")
        ax.text(
            0.2, -0.26,
            f'B/L Val Voting Acc: {max(result["voting_acc"]):.4f}/{np.mean(result["voting_acc"]):.4f}',
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes, fontsize=10
        )

    ax.legend()
    ax.set_title(f'{task_name} Train & Val Accuracy')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    # save train accuracy
    fig.savefig( os.path.join(experiment_root, 'accuracy.png'))


    fig = plt.figure(dpi=400)
    ax = plt.axes()
    ax.plot(train_loss[1:], label='Train loss', color="b")
    ax.plot(val_loss[1:], label='Val loss', color="r")

    # ax.plot(v_loss,label='Validation Loss')
    ax.legend()
    ax.set_title(f'{task_name} Train & Val loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    fig.savefig( os.path.join(experiment_root, 'loss.png'))


    if 'train_sslloss' in result:
        fig = plt.figure(dpi=400)
        ax = plt.axes()
        ax.plot(result['train_sslloss'], label='sslloss', color="#789958")
        ax.plot(result['train_eucloss'], label='eucloss', color="#995868")
        ax.plot(result['train_cosloss'], label='cosloss', color="#C95868")
        # 將圖的y軸範圍設為平均值加減兩倍標準差
        mean = np.mean(result['train_sslloss']+result['train_eucloss']+result['train_cosloss'])
        std = np.std(result['train_sslloss']+result['train_eucloss']+result['train_cosloss'])
        ax.set_ylim(mean-1.5*std, mean+2*std)
        ax.legend()
        ax.set_title(f'{task_name} Train loss detail')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Accuracy')
        fig.savefig( os.path.join(experiment_root, 'loss_detail.png'))

        fig = plt.figure(dpi=400)
        ax = plt.axes()
        ax.plot(result['train_varloss'], label='var', color="#785568")
        # 將圖的y軸範圍設為平均值加減兩倍標準差
        mean = np.mean(result['train_varloss'])
        std = np.std(result['train_varloss'])
        ax.set_ylim(mean-2*std, mean+2*std)
        ax.legend()
        ax.set_title(f'{task_name} Train loss detail')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Accuracy')
        fig.savefig( os.path.join(experiment_root, 'varloss.png'))



from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    """Plot the confusion matrix."""
    fig, ax = plt.subplots(figsize=(len(classes), len(classes)), dpi=600)
    plt.rcParams.update({'font.size': 12})
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    return fig, ax

def compute_metrics(y_true, y_pred):
    """Compute accuracy and kappa scores."""
    accuracy = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    return accuracy, kappa

def draw_confusion_matrix(y_true, y_pred, classes, save_path=None, title='Confusion Matrix', msg='Test', cmap=plt.cm.Blues):
    """Create a confusion matrix plot and return the figure."""
    print (f'[plot_confusion_matrix] title: {title}')
    print (f'[plot_confusion_matrix] save_path: {save_path}')

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    accuracy, kappa = compute_metrics(y_true, y_pred)

    fig, ax = plot_confusion_matrix(cm, classes, title=title, cmap=cmap)

    # Display the metrics
    metrics_text = f'Accuracy: {accuracy:.4f}\nKappa: {kappa:.4f}'
    fig.text(0.01, 0.01, metrics_text, fontsize=12, color='black', va='bottom', ha='left')

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    
    return fig
