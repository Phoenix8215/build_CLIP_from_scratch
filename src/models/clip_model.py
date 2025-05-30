import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np


class SimpleTextEncoder(nn.Module):
    
    def __init__(self, vocab_size=10000, embed_dim=512, hidden_dim=256):
        super(SimpleTextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.projection = nn.Linear(hidden_dim * 2, embed_dim)
        
    def forward(self, input_ids):
        embeds = self.embedding(input_ids)
        lstm_out, (hidden, _) = self.lstm(embeds)
        
        # 使用最后一个时间步的输出
        text_embeds = self.projection(lstm_out[:, -1, :])
        return F.normalize(text_embeds, dim=-1)


class ImageEncoder(nn.Module):
    
    def __init__(self, embed_dim=512):
        super(ImageEncoder, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, embed_dim)
        
    def forward(self, x):
        return F.normalize(self.backbone(x), dim=-1)


class SimpleCLIPModel(nn.Module):
    
    def __init__(self, embed_dim=512, vocab_size=10000, temperature=0.07):
        super(SimpleCLIPModel, self).__init__()
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = SimpleTextEncoder(vocab_size, embed_dim)
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        
        self.create_simple_vocab()
        
    def create_simple_vocab(self):

        words = [
            'a', 'photo', 'of', 'picture', 'image', 'this', 'is', 'clear', 'good',
            'airplane', 'automobile', 'bird', 'cat', 'deer', 
            'dog', 'frog', 'horse', 'ship', 'truck',
            'flying', 'driving', 'animal', 'vehicle', 'transportation',
            'the', 'in', 'on', 'with', 'and', 'or'
        ]
        
        self.vocab = ['[PAD]', '[UNK]'] + words
        self.word_to_id = {word: i for i, word in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        
    def tokenize_text(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        
        tokenized = []
        max_len = 20  # 最大序列长度
        
        for text in texts:
            words = text.lower().replace(',', '').replace('.', '').split()
            token_ids = []
            
            for word in words[:max_len-1]:
                token_id = self.word_to_id.get(word, 1)  # 1是[UNK]的ID
                token_ids.append(token_id)
            
            # 填充到固定长度
            while len(token_ids) < max_len:
                token_ids.append(0)  # 0是[PAD]的ID
            
            tokenized.append(token_ids)
        
        return torch.tensor(tokenized, dtype=torch.long)
        
    def forward(self, images, texts=None, input_ids=None):
        image_embeds = self.image_encoder(images)
        
        if texts is not None:
            input_ids = self.tokenize_text(texts).to(images.device)
        elif input_ids is None:
            raise ValueError("必须提供texts或input_ids")
        
        text_embeds = self.text_encoder(input_ids)
        
        return image_embeds, text_embeds
    
    def compute_loss(self, image_embeds, text_embeds):
        # 计算相似度矩阵
        logits = torch.matmul(image_embeds, text_embeds.T) * self.temperature.exp()
        
        # 创建标签（对角线为正样本）
        batch_size = image_embeds.shape[0]
        labels = torch.arange(batch_size, device=image_embeds.device)
        
        # 计算交叉熵损失
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        
        return (loss_i2t + loss_t2i) / 2 