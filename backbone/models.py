import torch
from torchvision import models
from torchvision.models import ResNet50_Weights
import torch.nn as nn

def build_backbone(model_name=None, num_classes=7, freeze=False):
    if model_name == 'resnet_50':
        model = MyResNet50(num_classes=num_classes, freeze=freeze)
        
        model_sequence = []
    else:
        assert model_name == 'sequence_model', "Here must be Sequence_Model"
        model = MyResNet50(num_classes=num_classes, freeze=freeze)
        
        model_sequence = SequenceModel()

    return model, model_sequence


class MyResNet50(nn.Module):
    def __init__(self, num_classes=7, freeze=False, dropout_rate=0.5):
        super(MyResNet50, self).__init__()
        # 加载预训练的 ResNet-50 模型
        self.resnet_50 = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        # 获取 ResNet-50 的特征提取部分，即去掉最后一层全连接层之前的部分
        self.features = nn.Sequential(*list(self.resnet_50.children())[:-1], nn.Dropout(p=dropout_rate))

        if freeze:
            ## Freeze stem, layer1, layer2 these 3 parts
            for params in self.resnet_50.parameters():
                params.requires_grad = False
            
            for param in self.resnet_50.layer3.parameters():
                param.requires_grad = True
            for param in self.resnet_50.layer4.parameters():
                param.requires_grad = True

        num_feats = self.resnet_50.fc.in_features
        self.fc = nn.Linear(num_feats, num_classes)
        ## Initialization for new linear layer
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        # 将输入通过 ResNet-50 的特征提取部分传递
        features = self.features(x).squeeze()
        output = self.fc(features)
        return output, features



class SequenceModel(nn.Module):
    def __init__(self, input_size=2048, hidden_size=128, output_size=7, num_layers=2, dropout=0.2):
        super(SequenceModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)
        return out




# 训练模型（假设你有训练数据和标签）
# 这里假设你有数据 x 和标签 y
# x 的形状是 (batch_size, sequence_length, input_size)，y 的形状是 (batch_size,)
# batch_size 是你的训练批次大小
# 你需要将 x 和 y 转换为张量，然后将它们传递给模型和损失函数
# 之后通过优化器更新模型参数
