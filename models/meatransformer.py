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
    num_labels: int = 12

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


class Model(nn.Module):
    def __init__(self, config: Config):
        super(Model, self).__init__()
        self.config = config
        self.pretrain_model = EsmModel.from_pretrained(config.pretrain_model_path, ignore_mismatched_sizes=True)
        self.mea = MEATransformer(config.database, config.conv_in, config.num_heads, config.num_layers, config.num_labels, config.k, config.retrieval_ratio)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        with torch.no_grad():
            embeddings = self.pretrain_model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = embeddings.last_hidden_state

        queries = torch.mean(embeddings, dim=1)
        mea_logits = self.mea(queries)

        logits = mea_logits

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)
        return ModelOutput(loss=loss, logits=logits)


class Collator:
    def __init__(self, tokenizer_path):
        self.tokenizer = EsmTokenizer.from_pretrained(tokenizer_path)

    def __call__(self, batch):
        seqs = [_.sequence for _ in batch]
        labels = [_.cls for _ in batch]
        labels = torch.tensor(labels).long()
        inputs = self.tokenizer(seqs, return_tensors="pt", padding='max_length', truncation=True, max_length=1000)
        inputs['labels'] = labels
        return inputs

