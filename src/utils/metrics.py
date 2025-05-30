import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, classification_report


def compute_accuracy(image_embeds, text_embeds, labels):
    # 计算相似度矩阵
    similarities = torch.matmul(image_embeds, text_embeds.T)
    
    # 图像到文本检索
    i2t_predictions = torch.argmax(similarities, dim=1)
    i2t_accuracy = (i2t_predictions == labels).float().mean().item()
    
    # 文本到图像检索
    t2i_predictions = torch.argmax(similarities.T, dim=1)
    t2i_accuracy = (t2i_predictions == labels).float().mean().item()
    
    return i2t_accuracy, t2i_accuracy


def compute_recall_at_k(image_embeds, text_embeds, labels, k_values=[1, 5, 10]):
    similarities = torch.matmul(image_embeds, text_embeds.T)
    
    recalls = {}
    
    # 图像到文本检索的Recall@K
    i2t_sorted_indices = torch.argsort(similarities, dim=1, descending=True)
    for k in k_values:
        correct = 0
        for i, label in enumerate(labels):
            if label in i2t_sorted_indices[i, :k]:
                correct += 1
        recalls[f'i2t_recall@{k}'] = correct / len(labels)
    
    # 文本到图像检索的Recall@K
    t2i_sorted_indices = torch.argsort(similarities.T, dim=1, descending=True)
    for k in k_values:
        correct = 0
        for i, label in enumerate(labels):
            if label in t2i_sorted_indices[i, :k]:
                correct += 1
        recalls[f't2i_recall@{k}'] = correct / len(labels)
    
    return recalls


class AverageMeter:
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def evaluate_model(model, dataloader, device):
    model.eval()
    
    all_image_embeds = []
    all_text_embeds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label']
            
            image_embeds, text_embeds = model(images, input_ids, attention_mask)
            
            all_image_embeds.append(image_embeds.cpu())
            all_text_embeds.append(text_embeds.cpu())
            all_labels.extend(labels.tolist())
    
    all_image_embeds = torch.cat(all_image_embeds, dim=0)
    all_text_embeds = torch.cat(all_text_embeds, dim=0)
    all_labels = torch.tensor(all_labels)
    
    i2t_acc, t2i_acc = compute_accuracy(all_image_embeds, all_text_embeds, all_labels)
    recalls = compute_recall_at_k(all_image_embeds, all_text_embeds, all_labels)
    
    metrics = {
        'i2t_accuracy': i2t_acc,
        't2i_accuracy': t2i_acc,
        'average_accuracy': (i2t_acc + t2i_acc) / 2,
        **recalls
    }
    
    return metrics 