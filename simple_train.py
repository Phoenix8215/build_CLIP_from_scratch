import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.models.clip_model import SimpleCLIPModel
from src.data.dataset import create_simple_dataloaders


def compute_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            texts = batch['text']
            labels = batch['label'].to(device)
            
            image_embeds, text_embeds = model(images, texts=texts)
            
            similarities = torch.matmul(image_embeds, text_embeds.T)
            predictions = torch.argmax(similarities, dim=1)
            
            correct += (predictions == torch.arange(len(predictions)).to(device)).sum().item()
            total += len(predictions)
    
    return correct / total


def train_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        texts = batch['text']
        
        optimizer.zero_grad()
        
        image_embeds, text_embeds = model(images, texts=texts)
        
        loss = model.compute_loss(image_embeds, text_embeds)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg Loss': f'{total_loss/num_batches:.4f}',
            'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
    
    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description='训练简化版CLIP模型')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--embed_dim', type=int, default=256, help='嵌入维度')
    parser.add_argument('--num_workers', type=int, default=2, help='数据加载线程数')
    parser.add_argument('--save_dir', type=str, default='./simple_checkpoints', help='模型保存目录')
    parser.add_argument('--device', type=str, default='auto', help='训练设备')
    
    args = parser.parse_args()
    
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"使用设备: {device}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("正在创建数据加载器...")
    train_loader, val_loader = create_simple_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")
    
    print("正在创建简化CLIP模型...")
    model = SimpleCLIPModel(embed_dim=args.embed_dim)
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    train_losses = []
    val_accuracies = []
    best_accuracy = 0.0
    
    print("开始训练...")
    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")
        
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        train_losses.append(train_loss)
        
        print("正在评估模型...")
        val_accuracy = compute_accuracy(model, val_loader, device)
        val_accuracies.append(val_accuracy)
        
        print(f"训练损失: {train_loss:.4f}")
        print(f"验证准确率: {val_accuracy:.4f}")
        
        scheduler.step()
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_accuracy': best_accuracy,
                'train_losses': train_losses,
                'val_accuracies': val_accuracies,
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"保存最佳模型 (准确率: {best_accuracy:.4f})")
        
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_accuracy': best_accuracy,
                'train_losses': train_losses,
                'val_accuracies': val_accuracies,
            }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth'))
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('train_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.title(' accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'training_curves.png'))
    print(f"训练曲线已保存到: {os.path.join(args.save_dir, 'training_curves.png')}")
    
    print(f"训练完成！最佳验证准确率: {best_accuracy:.4f}")


if __name__ == '__main__':
    main() 