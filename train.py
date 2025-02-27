from Transmil_MLP import TransMIL_MLP
from MMIL import MultipleMILTransformer
from detr import MILModel
import os
import torch
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
from dataset_196patch import get_dataloader
from train_val_196patch import train_one_epoch,validate_one_epoch
import torch
import torch.nn as nn
import torch.nn.functional as F
from TransMIL import TransMIL  # 确保你已经正确安装或导入了 TransMIL 模型
from tqdm import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from plt import plot_metrics
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
# 路径设置
# class Args:
#     def __init__(self):
#         # 数据相关
#         self.in_chans = 1024           # 输入特征维度
#         self.embed_dim = 512           # 嵌入维度
#         self.n_classes = 5             # 分类类别数
#         self.num_msg = 10              # 每组的消息令牌数量
#         self.num_subbags = 4           # 子袋（sub-bags）的数量
#         self.mode = 'random'           # 分组模式：'random', 'coords', 'seq', 'embed', 'idx'
#         self.num_layers = 2            # Transformer 层数
#         self.ape = True                # 是否使用绝对位置编码
#         self.type = 'camelyon16'       # 数据类型，用于特定的前处理
#         # 其他可能需要的参数
#         self.max_size = 4300           # 分组的最大大小（根据需要调整）

# args = Args()
features_directory = '/root/autodl-tmp/ubc-ocean/features/'
labels_csv = '/root/autodl-tmp/ubc-ocean/train.csv'
# 设置保存模型的目录
save_directory = 'detr_model/detr_196patch_random2'
os.makedirs(save_directory, exist_ok=True)  # 如果目录不存在，则创建它
# 读取标签文件
label_df = pd.read_csv(labels_csv)

# 创建标签映射，将标签字符串映射到整数 ID
label_mapping = {label: idx for idx, label in enumerate(label_df['label'].unique())}
#label_mapping_reverse = {v: k for k, v in label_mapping.items()}
label_df['label_id'] = label_df['label'].map(label_mapping)

# 获取所有图像的 ID 和标签
image_ids = label_df['image_id'].astype(str).tolist()
labels = label_df['label_id'].tolist()

# 创建一个字典，键为图像 ID，值为标签 ID
#image_label_dict = dict(zip(image_ids, labels))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_classes = len(label_mapping)
num_epochs = 30
save_every = 5  # 每隔多少个 epoch 保存一次模型
batch_size = 64  # 由于每个样本的特征长度不同，batch_size 设置为 1
learning_rate = 1e-4

    
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# 准备存储每个折的数据索引
folds = []

for train_index, val_index in skf.split(image_ids, labels):
    folds.append((train_index, val_index))
    
# 存储所有折的结果

fold_results = {}

for fold, (train_index, val_index) in enumerate(folds):
    print(f'\n===== Fold {fold+1} / {n_splits} =====')
    best_val_acc = 0.0
    # 初始化每个折的结果存储
    fold_results[fold] = {
        'train_losses': [],
        'val_losses': [],
        'train_accs': [],
        'val_accs': [],
        'train_bal_accs': [],
        'val_bal_accs': [],
        'train_f1s': [],
        'val_f1s': [],
        'best_val_bal_acc': 0.0,  # 新增，用于存储最高的 val_bal_acc
        'best_epoch': 0,          # 新增，用于存储达到最高 val_bal_acc 的 epoch
    }
     # 获取数据加载器
    
    train_loader, val_loader = get_dataloader(
        train_index, val_index, image_ids, labels, features_directory, batch_size)
    
    # 初始化模型、损失函数和优化器
  #  model = TransMIL_MLP(n_classes=n_classes).to(device)
    model = MILModel().to(device)

    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, num_epochs + 1):
        print(f'\nEpoch {epoch}/{num_epochs}')

        # 训练一个 epoch
        train_loss, train_acc, train_bal_acc, train_f1 = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # 验证一个 epoch
        val_loss, val_acc, val_bal_acc, val_f1 = validate_one_epoch(model, val_loader, criterion, device)

        # 更新 fold_results
        fold_results[fold]['train_losses'].append(train_loss)
        fold_results[fold]['val_losses'].append(val_loss)
        fold_results[fold]['train_accs'].append(train_acc)
        fold_results[fold]['val_accs'].append(val_acc)
        fold_results[fold]['train_bal_accs'].append(train_bal_acc)
        fold_results[fold]['val_bal_accs'].append(val_bal_acc)
        fold_results[fold]['train_f1s'].append(train_f1)
        fold_results[fold]['val_f1s'].append(val_f1)

        print(f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Bal Acc: {train_bal_acc:.4f}, F1: {train_f1:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Bal Acc: {val_bal_acc:.4f}, F1: {val_f1:.4f}')

        # 检查并更新最高的 val_bal_acc
        if val_bal_acc > fold_results[fold]['best_val_bal_acc']:
            fold_results[fold]['best_val_bal_acc'] = val_bal_acc
            fold_results[fold]['best_epoch'] = epoch

            # 保存最佳模型
            best_model_path = os.path.join(save_directory, f'best_model_fold{fold+1}.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f'Best model saved to {best_model_path}')

        # 每隔一定 epoch 保存模型
        if epoch % save_every == 0:
            model_save_path = os.path.join(save_directory, f'model_fold{fold+1}_epoch{epoch}.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f'Model saved to {model_save_path}')

# 在所有折训练完成后
results_file_path = os.path.join('/root/detr_model', 'training_results.txt')
with open(results_file_path, 'w') as f:
    for fold in fold_results:
        best_val_bal_acc = fold_results[fold]['best_val_bal_acc']
        best_epoch = fold_results[fold]['best_epoch']
        result_str = f'Fold {fold+1}: Best Val Bal Acc: {best_val_bal_acc:.4f} at epoch {best_epoch}\n'
        print(result_str)  # 可以保留打印输出
        f.write(result_str)

print(f'Results saved to {results_file_path}')
plot_metrics(fold_results, num_epochs, save_dir='detr_model')
