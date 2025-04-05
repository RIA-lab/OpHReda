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
    k: int = 3
    num_layers: int = 3
    num_heads: int = 8
    retrieval_ratio: float = 0.2
    conv_in: int = 640
    conv_mid: int = 1280
    conv_out: int = 640
    num_labels: int = 12
    calibrator_num_heads: int = 8
    calibrator_num_layers: int = 6
    calibrator_embed_dim: int = 640

    def to_dict(self):
        output = asdict(self)
        return output


class SearchLayer(nn.Module):
    def __init__(self, initial_weight, k=3):
        super(SearchLayer, self).__init__()
        weight = initial_weight[:, :-1]
        weight = weight / weight.norm(dim=-1, keepdim=True)
        weight = weight.T
        label = initial_weight[:, -1]
        self.weight = nn.Parameter(weight.contiguous(), requires_grad=False)
        self.label = nn.Parameter(label.contiguous(), requires_grad=False)
        self.k = k

    def forward(self, queries):
        queries_norm = queries / queries.norm(dim=-1, keepdim=True)
        similarities = torch.matmul(queries_norm, self.weight)
        top_k_scores, top_k_indices = torch.topk(similarities, k=self.k, dim=1)
        top_k_labels = self.label[top_k_indices]
        top_k_seqs = self.weight.T[top_k_indices.flatten()]
        return top_k_scores, top_k_indices, top_k_seqs, top_k_labels


class MEAClassificationHead(nn.Module):
    def __init__(self, dim, num_labels, k):
        super().__init__()
        self.k = k
        self.num_labels = num_labels
        self.dense = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(0.0)
        self.out_proj = nn.Linear(dim, num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        x = x.view(-1, self.k, self.num_labels)
        x = torch.mean(x, dim=1)
        return x


class MEALayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Linear projections for Q, K, V
        self.W_Q = nn.Linear(embed_dim, embed_dim)
        self.W_K = nn.Linear(embed_dim, embed_dim)
        self.W_V = nn.Linear(embed_dim, embed_dim)
        self.W_O = nn.Linear(embed_dim, embed_dim)

    def forward(self, H_cxr):
        # Compute Q, K, V
        Q = self.W_Q(H_cxr)
        K = self.W_K(H_cxr)
        V = self.W_V(H_cxr)

        # Compute attention scores
        d_k = self.embed_dim // self.num_heads
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        A = torch.softmax(attention_scores, dim=-1)

        # Split A into A_x and A_r
        H_x = H_cxr[:, :2, :]
        H_r = H_cxr[:, 2:, :]
        L = H_x.shape[1]
        A_x, A_r = A[:, :, :L], A[:, :, L:]

        # Compute attention output
        attn_output = torch.matmul(A_x, self.W_V(H_x)) + torch.matmul(A_r, self.W_V(H_r))
        output = self.W_O(attn_output)

        return output


class MEATransformer(nn.Module):
    def __init__(self, database, embed_dim, num_heads, num_layers, num_labels, k, retrieval_ratio):
        super().__init__()
        self.retrieval_ratio = retrieval_ratio
        self.k = k
        self.embed_dim = embed_dim
        self.num_labels = num_labels
        self.search_layer = SearchLayer(torch.load(database), k)
        self.mea_layers = nn.ModuleList([MEALayer(embed_dim, num_heads) for _ in range(num_layers)])
        self.classification_head = MEAClassificationHead(embed_dim, num_labels, k)

    def retrieval(self, top_k_labels, num_classes=12):
        # Convert top_k_labels to one-hot encoding
        one_hot_labels = F.one_hot(top_k_labels.to(torch.int64), num_classes=num_classes).float()
        # Sum the one-hot encoded labels along the top-k dimension
        aggregated_labels = one_hot_labels.sum(dim=1)
        # Normalize the aggregated labels
        normalized_labels = aggregated_labels / aggregated_labels.sum(dim=-1, keepdim=True)
        return normalized_labels

    def forward(self, queries):
        similarity_scores, indices, rsa_seqs, rsa_label = self.search_layer(queries)
        H_r = rsa_seqs.unsqueeze(1)
        H_x = queries.repeat_interleave(self.k, 0).unsqueeze(1)
        CLS = F.one_hot(rsa_label.to(torch.int64), num_classes=self.embed_dim)
        CLS = CLS.view(-1, 1, self.embed_dim)
        CLS = CLS * similarity_scores.unsqueeze(-1).repeat(1, 1, self.embed_dim).view(-1, 1, self.embed_dim)
        H_cxr = torch.cat([CLS, H_x, H_r], dim=1)
        for mea_layer in self.mea_layers:
            H_cxr = mea_layer(H_cxr)
        logits = self.classification_head(H_cxr)
        retrieval_logits = self.retrieval(rsa_label, num_classes=self.num_labels)
        logits = logits*(1-self.retrieval_ratio) + retrieval_logits*self.retrieval_ratio
        return logits


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


class Calibrator(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super().__init__()
        self.embed_dim = embed_dim
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(embed_dim, 1)

    def forward(self, embeddings, logits):
        logits = torch.cat([logits, torch.zeros(logits.shape[0], self.embed_dim-12).to(logits.device)], dim=1)
        logits = logits.unsqueeze(1)
        out = self.decoder(logits, embeddings)
        out = self.linear(out.squeeze(1))
        return out


class RMSELoss:
    def __init__(self):
        self.mse = nn.MSELoss()

    def __call__(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true))


class Model(nn.Module):
    def __init__(self, config: Config):
        super(Model, self).__init__()
        self.config = config
        self.pretrain_model = EsmModel.from_pretrained(config.pretrain_model_path, ignore_mismatched_sizes=True)
        self.mrlat = MRLAT(config)
        self.mea = MEATransformer(config.database, config.conv_in, config.num_heads, config.num_layers, config.num_labels, config.k, config.retrieval_ratio)
        self.calibrator = Calibrator(self.config.calibrator_embed_dim, self.config.calibrator_num_heads, self.config.calibrator_num_layers)
        # self.loss_fct = nn.CrossEntropyLoss()
        self.loss_fct = RMSELoss()

    def reweight(self, class_weights):
        self.loss_fct = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, input_ids, attention_mask, cls=None, labels=None):
        embeddings = self.pretrain_model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = embeddings.last_hidden_state
        mrlat_logits = self.mrlat(embeddings)
        queries = torch.mean(embeddings, dim=1)
        mea_logits = self.mea(queries)


        logits = mrlat_logits + mea_logits
        logits = torch.softmax(logits, dim=-1)
        cls_pred = torch.argmax(logits, dim=-1) + 1

        bias = self.calibrator(embeddings, logits)
        bias = torch.squeeze(bias)
        bias = torch.nan_to_num(bias, nan=0.0)

        pH = cls_pred + bias

        loss = None
        if labels is not None:
            # loss = self.loss_fct(logits.view(-1, self.config.num_labels), cls.view(-1))
            loss = self.loss_fct(pH, labels)
        # weights = torch.sum(weights.permute(0, 2, 1), dim=-1)
        # weights = weights.masked_fill(attention_mask == 0, -1e6)
        # weights = torch.softmax(weights, dim=-1)
        # return ModelOutput(loss=loss, logits=logits)
        return ModelOutput(loss=loss, logits=pH)


class Collator:
    def __init__(self, tokenizer_path):
        self.tokenizer = EsmTokenizer.from_pretrained(tokenizer_path)

    def __call__(self, batch):
        seqs = [_.sequence for _ in batch]
        cls = [_.cls for _ in batch]
        cls = torch.tensor(cls).long()
        labels = [_.ph_label for _ in batch]
        labels = torch.tensor(labels).float()
        inputs = self.tokenizer(list(seqs), return_tensors="pt", padding='max_length', truncation=True, max_length=1000)
        inputs['cls'] = cls
        inputs['labels'] = labels
        return inputs

