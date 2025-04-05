import torch
from transformers.modeling_outputs import ModelOutput
from transformers import EsmModel, EsmTokenizer
import torch.nn as nn
from dataclasses import dataclass, asdict
import torch.nn.functional as F
import math


@dataclass
class Config:
    tokenizer_path: str = 'pretrain_models/esm150'
    pretrain_model_path: str = 'pretrain_models/esm150'
    database: str = 'database.pt'
    conv_in: int = 640
    conv_mid: int = 1280
    conv_out: int = 640
    num_labels: int = 12

    def to_dict(self):
        output = asdict(self)
        return output


class MultScaleConv1d(nn.Module):
    def __init__(self, in_channels=640, mid_channels=1280, out_channels=640):
        super(MultScaleConv1d, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(mid_channels, mid_channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(mid_channels, out_channels, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(mid_channels)
        self.bn2 = nn.BatchNorm1d(mid_channels)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(0.0)  # Adjust dropout rate
        # Residual connections
        self.residual_conv1 = nn.Conv1d(in_channels, mid_channels, kernel_size=1)
        self.residual_conv2 = nn.Conv1d(mid_channels, mid_channels, kernel_size=1)
        self.residual_conv3 = nn.Conv1d(mid_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=False)  # Use non-inplace ReLU

    def forward(self, x):
        x = x.permute(0, 2, 1)

        # First conv block
        residual = self.residual_conv1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x + residual

        # Second conv block
        residual = self.residual_conv2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x + residual

        # Third conv block
        residual = self.residual_conv3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x + residual
        # Add residual

        return x


class ClassificationHead(nn.Module):
    def __init__(self, dim, num_labels):
        super().__init__()
        self.dense = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(0.0)
        self.out_proj = nn.Linear(dim, num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class MRLAT(nn.Module):
    def __init__(self, config: Config):
        super(MRLAT, self).__init__()
        self.values_conv = MultScaleConv1d(config.conv_in, config.conv_mid, config.conv_out)
        self.weights_conv = MultScaleConv1d(config.conv_in, config.conv_mid, config.conv_out)
        self.classificationHead = ClassificationHead(config.conv_out*2, config.num_labels)
        self.ln = nn.LayerNorm(config.conv_out)

    def forward(self, embeddings):
        values = self.values_conv(embeddings)
        weights = self.weights_conv(embeddings)
        x_sum = torch.sum(values * weights, dim=-1)  # Attention-weighted pooling
        x_sum = self.ln(x_sum)
        x_max, _ = torch.max(values, dim=-1)  # Max pooling
        x_max = self.ln(x_max)
        x = torch.cat([x_sum, x_max], dim=1)
        logits = self.classificationHead(x)
        return logits





class Model(nn.Module):
    def __init__(self, config: Config):
        super(Model, self).__init__()
        self.config = config
        self.pretrain_model = EsmModel.from_pretrained(config.pretrain_model_path, ignore_mismatched_sizes=True)
        self.mrlat = MRLAT(config)
        self.loss_fct = nn.CrossEntropyLoss()

    def reweight(self, class_weights):
        self.loss_fct = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, input_ids, attention_mask, labels=None):
        with torch.no_grad():
            embeddings = self.pretrain_model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = embeddings.last_hidden_state

        mrlat_logits = self.mrlat(embeddings)

        logits = mrlat_logits

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)
        return ModelOutput(loss=loss, logits=logits)


class Collator:
    def __init__(self, tokenizer_path):
        self.tokenizer = EsmTokenizer.from_pretrained(tokenizer_path)

    def __call__(self, batch):
        seqs = [_.sequence for _ in batch]
        labels = [int(_.ph)-1 for _ in batch]
        labels = torch.tensor(labels).long()
        inputs = self.tokenizer(seqs, return_tensors="pt", padding='max_length', truncation=True, max_length=1000)
        inputs['labels'] = labels
        return inputs

