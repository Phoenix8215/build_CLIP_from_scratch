#!/usr/bin/env python3
"""
Simplified CLIP inference script with visualization
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.models.clip_model import SimpleCLIPModel
from src.data.dataset import SimpleCIFARDataset, get_transforms


class SimpleInference:
    
    def __init__(self, model_path, device='auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        self.model = self._load_model(model_path)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.cifar_classes = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
    
    def _load_model(self, model_path):
        """Load trained model"""
        print("Loading model...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        model = SimpleCLIPModel(embed_dim=256)  # Use training embedding dimension
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"✅ Model loaded successfully, best accuracy: {checkpoint.get('best_accuracy', 'N/A')}")
        return model
    
    def _denormalize_image(self, tensor):
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        
        denorm_tensor = tensor * std[:, None, None] + mean[:, None, None]
        denorm_tensor = torch.clamp(denorm_tensor, 0, 1)
        
        return denorm_tensor.permute(1, 2, 0).cpu().numpy()
    
    def visualize_prediction(self, image, pred_class, pred_prob, all_probs, true_class=None, save_path=None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image.squeeze(0)
            image_np = self._denormalize_image(image)
        elif isinstance(image, str):
            image_np = np.array(Image.open(image).convert('RGB').resize((224, 224))) / 255.0
        else:
            image_np = np.array(image.resize((224, 224))) / 255.0
        
        ax1.imshow(image_np)
        
        title = f'Predicted: {pred_class} ({pred_prob:.1%})'
        if true_class:
            title += f'\nActual: {true_class}'
            color = 'green' if pred_class == true_class else 'red'
            ax1.add_patch(patches.Rectangle((0, 0), 224, 224, linewidth=3, edgecolor=color, facecolor='none'))
        
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        classes = list(all_probs.keys())
        probs = list(all_probs.values())
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))
        bars = ax2.barh(classes, probs, color=colors)
        
        pred_idx = classes.index(pred_class)
        bars[pred_idx].set_color('red')
        bars[pred_idx].set_alpha(0.8)
        
        if true_class and true_class in classes:
            true_idx = classes.index(true_class)
            if true_idx != pred_idx:
                bars[true_idx].set_color('green')
                bars[true_idx].set_alpha(0.6)
        
        ax2.set_xlabel('Probability', fontsize=12)
        ax2.set_title('Class Probability Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, max(probs) * 1.1)
        
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            ax2.text(prob + max(probs) * 0.01, i, f'{prob:.3f}', 
                    va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def predict_image_class(self, image_path, visualize=True):
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        class_descriptions = [f"a photo of a {class_name}" for class_name in self.cifar_classes]
        
        with torch.no_grad():
            image_embed, _ = self.model(image_tensor, texts=["dummy"])
            
            dummy_images = torch.zeros(len(class_descriptions), 3, 224, 224).to(self.device)
            _, text_embeds = self.model(dummy_images, texts=class_descriptions)
            
            similarities = torch.matmul(image_embed, text_embeds.T)
            probabilities = F.softmax(similarities, dim=-1)
            
            pred_idx = torch.argmax(probabilities, dim=-1).item()
            pred_class = self.cifar_classes[pred_idx]
            pred_prob = probabilities[0, pred_idx].item()
            
            all_probs = {
                self.cifar_classes[i]: probabilities[0, i].item()
                for i in range(len(self.cifar_classes))
            }
        
        if visualize:
            self.visualize_prediction(image, pred_class, pred_prob, all_probs)
        
        return pred_class, pred_prob, all_probs
    
    def encode_text(self, text):
        dummy_images = torch.zeros(1, 3, 224, 224).to(self.device)
        
        with torch.no_grad():
            _, text_embed = self.model(dummy_images, texts=[text])
        
        return text_embed.cpu().numpy()
    
    def demo_with_cifar(self, num_samples=5, visualize=True):
        print("Loading CIFAR test data...")
        dataset = SimpleCIFARDataset(train=False, transform=get_transforms(False))
        
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        
        correct = 0
        results = []
        
        for i, idx in enumerate(indices):
            sample = dataset[idx]
            image = sample['image'].unsqueeze(0).to(self.device)
            true_label = sample['label']
            true_class = self.cifar_classes[true_label]
            
            with torch.no_grad():
                class_descriptions = [f"a photo of a {class_name}" for class_name in self.cifar_classes]
                dummy_images = torch.zeros(len(class_descriptions), 3, 224, 224).to(self.device)
                
                image_embed, _ = self.model(image, texts=["dummy"])
                _, text_embeds = self.model(dummy_images, texts=class_descriptions)
                
                similarities = torch.matmul(image_embed, text_embeds.T)
                probabilities = F.softmax(similarities, dim=-1)
                
                pred_idx = torch.argmax(probabilities, dim=-1).item()
                pred_class = self.cifar_classes[pred_idx]
                pred_prob = probabilities[0, pred_idx].item()
                
                all_probs = {
                    self.cifar_classes[j]: probabilities[0, j].item()
                    for j in range(len(self.cifar_classes))
                }
            
            is_correct = pred_class == true_class
            if is_correct:
                correct += 1
            
            results.append({
                'image': sample['image'],
                'true_class': true_class,
                'pred_class': pred_class,
                'pred_prob': pred_prob,
                'all_probs': all_probs,
                'is_correct': is_correct
            })
            
            print(f"\n=== Sample {i+1}/{num_samples} ===")
            print(f"True class: {true_class}")
            print(f"Predicted class: {pred_class}")
            print(f"Prediction probability: {pred_prob:.4f}")
            print(f"Prediction result: {'✅ Correct' if is_correct else '❌ Incorrect'}")
            
            if visualize:
                save_path = f"demo_sample_{i+1}.png"
                self.visualize_prediction(
                    sample['image'], pred_class, pred_prob, all_probs, 
                    true_class=true_class, save_path=save_path
                )
        
        accuracy = correct / num_samples
        print(f"\n=== Demo Summary ===")
        print(f"Total samples: {num_samples}")
        print(f"Correct predictions: {correct}")
        print(f"Accuracy: {accuracy:.2%}")
        
        if visualize and num_samples <= 6:
            self.create_summary_visualization(results)
        
        return results
    
    def create_summary_visualization(self, results):
        num_samples = len(results)
        cols = min(3, num_samples)
        rows = (num_samples + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if num_samples == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, result in enumerate(results):
            row, col = i // cols, i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            image_np = self._denormalize_image(result['image'])
            ax.imshow(image_np)
            
            title = f"Pred: {result['pred_class']}\nActual: {result['true_class']}"
            color = 'green' if result['is_correct'] else 'red'
            
            ax.set_title(title, fontsize=10, color=color, fontweight='bold')
            ax.add_patch(patches.Rectangle((0, 0), 224, 224, linewidth=3, 
                                         edgecolor=color, facecolor='none'))
            ax.axis('off')
        
        for i in range(num_samples, rows * cols):
            row, col = i // cols, i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.axis('off')
        
        plt.suptitle('CLIP Model Prediction Results Overview', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('demo_summary.png', dpi=150, bbox_inches='tight')
        print("Demo summary visualization saved to: demo_summary.png")
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Simplified CLIP inference with visualization')
    parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--image_path', type=str, help='Image path')
    parser.add_argument('--text', type=str, help='Text to encode')
    parser.add_argument('--demo', action='store_true', help='CIFAR demo mode')
    parser.add_argument('--device', type=str, default='auto', help='Device')
    parser.add_argument('--no_viz', action='store_true', help='Disable visualization')
    
    args = parser.parse_args()
    
    inference = SimpleInference(args.model_path, args.device)
    visualize = not args.no_viz
    
    if args.demo:
        print("Running CIFAR demo mode...")
        inference.demo_with_cifar(num_samples=5, visualize=visualize)
    
    elif args.image_path:
        if not os.path.exists(args.image_path):
            print(f"Error: Image file {args.image_path} does not exist")
            return
        
        print(f"Predicting image: {args.image_path}")
        pred_class, pred_prob, all_probs = inference.predict_image_class(args.image_path, visualize=visualize)
        
        print(f"\nPrediction results:")
        print(f"Class: {pred_class}")
        print(f"Probability: {pred_prob:.4f}")
        
        if not visualize:
            print(f"\nAll class probabilities:")
            for class_name, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
                print(f"  {class_name}: {prob:.4f}")
    
    elif args.text:
        print(f"Encoding text: '{args.text}'")
        text_embed = inference.encode_text(args.text)
        print(f"Text embedding shape: {text_embed.shape}")
        print(f"Embedding vector norm: {np.linalg.norm(text_embed):.4f}")
    
    else:
        print("Please specify --demo, --image_path or --text parameter")
        print("Examples:")
        print("  python simple_inference.py --model_path ./simple_checkpoints/best_model.pth --demo")
        print("  python simple_inference.py --model_path ./simple_checkpoints/best_model.pth --text 'a cat'")
        print("  python simple_inference.py --model_path ./simple_checkpoints/best_model.pth --image_path image.jpg")
        print("\nAdd --no_viz parameter to disable visualization")


if __name__ == '__main__':
    main() 