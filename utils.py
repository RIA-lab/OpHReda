import torch
from safetensors import safe_open
import numpy as np
import json
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, average_precision_score, precision_recall_curve
from sklearn.preprocessing import label_binarize
from scipy.stats import rankdata
from sklearn.metrics import roc_curve, auc
from itertools import cycle
import matplotlib.pyplot as plt
import yaml
from transformers import EsmTokenizer
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    # Ensure lr is treated as a float
    config['training']['lr'] = float(config['training']['lr'])
    return config

def read_fasta(fasta, return_as_dict=False):
    headers, sequences = [], []
    with open(fasta, 'r') as fast:

        for line in fast:
            if line.startswith('>'):
                head = line.replace('>', '').strip()
                headers.append(head)
                sequences.append('')
            else:
                seq = line.strip()
                if len(seq) > 0:
                    sequences[-1] += seq

    if return_as_dict:
        return dict(zip(headers, sequences))
    else:
        return (headers, sequences)


def write_fasta(headers, seqdata, path):
    with open(path, 'w') as pp:
        for i in range(len(headers)):
            pp.write('>' + headers[i] + '\n' + seqdata[i] + '\n')

    return


def write_json(writedict, path, indent=4, sort_keys=False):
    f = open(path, 'w')
    _ = f.write(json.dumps(writedict, indent=indent, sort_keys=sort_keys))
    f.close()


def read_json(path):
    f = open(path, 'r')
    readdict = json.load(f)
    f.close()

    return readdict


def replace_noncanonical(seq, replace_char='X'):
    '''Replace all non-canonical amino acids with a specific character'''

    for char in ['B', 'J', 'O', 'U', 'Z']:
        seq = seq.replace(char, replace_char)
    return seq


def get_amino_composition(seq, normalize=True):
    '''Return the amino acid composition for a protein sequence'''

    aac = np.array([seq.count(amino) for amino in list('ACDEFGHIKLMNPQRSTVWY')])
    if normalize:
        aac = aac / len(seq)

    return aac


# freeze the model parameters
def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


#count trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_safetonsors_model(model, checkpoint_path):
    state_dict = {}
    with safe_open(checkpoint_path, 'pt') as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    model.load_state_dict(state_dict)


def load_model_part(model, checkpoint_path, part):
    state_dict = {}
    with safe_open(checkpoint_path, 'pt') as f:
        for key in f.keys():
            if part == key.split('.')[0]:
                state_dict[key] = f.get_tensor(key)
    print(state_dict.keys())
    model.load_state_dict(state_dict, strict=False)


# init model weight xavier
def init_weights(model):
    for name, param in model.named_parameters():
        if 'weight' in name and len(param.size()) > 1:
            torch.nn.init.xavier_uniform_(param.data)
        elif 'bias' in name:
            torch.nn.init.zeros_(param.data)


def cls_metrics(eval_pred):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='macro')
    recall = recall_score(labels, predictions, average='macro')
    f1 = f1_score(labels, predictions, average='macro')
    return {"acc": acc, "precision": precision, "recall": recall, "f1": f1}


def reg_metrics(eval_pred):
    predictions, targets = eval_pred

    # R² (Coefficient of Determination)
    r2 = r2_score(targets, predictions)
    # Pearson Correlation Coefficient
    pearson_corr = np.corrcoef(predictions, targets)[0, 1]
    # Spearman Correlation Coefficient
    spearman_corr = np.corrcoef(rankdata(predictions), rankdata(targets))[0, 1]
    # MAE
    mae = np.mean(np.abs(predictions - targets))

    return {
        'R²': r2,
        'Pearson Correlation': pearson_corr,
        'Spearman Correlation': spearman_corr,
        'MAE': mae,
    }



def plot_roc(models_pred, n_classes, fig_title='ROC Curve', save_dir=None):
    plt.figure()
    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'blue', 'yellow', 'purple', 'pink'])
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(len(models_pred))]
    colors = cycle(colors)
    for model_name, model_pred in models_pred.items():
        y_true, y_pred = model_pred
        # Binarize the output
        y_true_bin = label_binarize(y_true, classes=[*range(n_classes)])
        y_pred_bin = label_binarize(y_pred, classes=[*range(n_classes)])

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute macro-average ROC curve and ROC area
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.plot(fpr["macro"], tpr["macro"], color=colors.__next__(),
                 label=model_name + ' macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('macro-average ROC curve')
    plt.legend(loc="lower right")

    if save_dir is not None:
        plt.savefig(f'{save_dir}/{fig_title}.png')

    plt.show()


def plot_pr(models_pred, n_classes, fig_title='Precision-Recall Curve', save_dir=None):
    plt.figure()
    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'blue', 'yellow', 'purple', 'pink'])
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(len(models_pred))]
    colors = cycle(colors)
    for model_name, model_pred in models_pred.items():
        y_true, y_pred = model_pred
        # Binarize the output
        y_true_bin = label_binarize(y_true, classes=[*range(n_classes)])
        y_pred_bin = label_binarize(y_pred, classes=[*range(n_classes)])

        # Compute PR curve and PR area for each class
        precision = dict()
        recall = dict()
        pr_auc = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_pred_bin[:, i])
            pr_auc[i] = auc(recall[i], precision[i])

        # Compute macro-average PR curve and PR area
        all_precision = np.unique(np.concatenate([precision[i] for i in range(n_classes)]))
        mean_recall = np.zeros_like(all_precision)
        for i in range(n_classes):
            mean_recall += np.interp(all_precision, precision[i], recall[i])
        mean_recall /= n_classes

        precision["macro"] = all_precision
        recall["macro"] = mean_recall
        pr_auc["macro"] = auc(recall["macro"], precision["macro"])

        # Plot all PR curves
        plt.plot(recall["macro"], precision["macro"], color=colors.__next__(),
                 label=model_name + ' macro-average PR curve (area = {0:0.2f})'
                       ''.format(pr_auc["macro"]))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('macro-average PR curve')
    plt.legend(loc="lower right")

    if save_dir is not None:
        plt.savefig(f'{save_dir}/{fig_title}.png')
    plt.show()


def plot_barh(models_metrics, save_dir=None):
    # models_metrics: dict {'model_name1': [acc, precision, recall, f1], 'model_name2': [acc, precision, recall, f1], ...}

    metrics = ['accuracy', 'precision', 'recall', 'f1-score']
    for idx, metric in enumerate(metrics):
        plt.figure(figsize=(8, 5))
        x = np.array([_ for _ in models_metrics.keys()])
        y = np.array([model_metrics[idx] for model_metrics in models_metrics.values()])
        # set the color of the bar
        colors = plt.cm.plasma(np.linspace(0, 1, len(x)))
        plt.barh(x, y, color=colors)
        plt.title(metric)
        plt.xlim([0.0, 1.05])
        # display the value of the bar
        for i, v in enumerate(y):
            plt.text(v, i, f'{v:.4f}', color='black', va='center')

        if save_dir is not None:
            plt.savefig(f'{save_dir}/{metric}.png', bbox_inches='tight')
        plt.show(bbox_inches='tight')


class WeightVisualizer:
    def __init__(self, tokenizer_path, model, save_dir='./'):
        self.tokenizer = EsmTokenizer.from_pretrained(tokenizer_path)
        self.model = model
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.train()
        self.model.to(self.device)
        self.save_dir = save_dir

    def __call__(self, accession, seq):
        length = len(seq)
        inputs = self.tokenizer([seq], return_tensors="pt", padding='max_length', truncation=True, max_length=1000)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        self.model.zero_grad()

        # Get input embeddings and enable gradient tracking
        input_embeds = self.model.pretrain_model.embeddings.word_embeddings(input_ids)
        input_embeds.requires_grad_(True)

        # Retain gradients for non-leaf tensor
        input_embeds.retain_grad()

        # Forward pass
        extended_attention_mask = self.model.pretrain_model.get_extended_attention_mask(attention_mask,
                                                                                                 input_ids.size())
        outputs = self.model.pretrain_model.encoder(input_embeds, attention_mask=extended_attention_mask)
        embeddings = outputs[0]
        mrlat_logits = self.model.mrlat(embeddings)
        queries = torch.mean(embeddings, dim=1)
        mea_logits = self.model.mea(queries)[0]

        logits = mrlat_logits + mea_logits
        logits = torch.softmax(logits, dim=-1)
        cls_pred = torch.argmax(logits, dim=-1) + 1

        bias = self.model.calibrator(embeddings, logits)
        bias = torch.squeeze(bias)
        pH = cls_pred + bias

        # Ensure pH is scalar or use a vector of ones for backward
        pH = pH.sum()  # Ensures pH is scalar
        pH.backward()

        input_grads = input_embeds.grad
        input_grads = input_grads[:, 1:length + 1, :]
        # Calculate gradient magnitudes
        grad_magnitudes = torch.norm(input_grads, dim=2).squeeze(0)
        grad_magnitudes = grad_magnitudes.cpu().detach().numpy()

        plt.figure(figsize=(15, 5))
        plt.bar(range(len(grad_magnitudes)), grad_magnitudes)
        plt.title("Token Importance")
        plt.xlabel("Token Position")
        plt.ylabel("Gradient Magnitude")
        plt.savefig(f'{self.save_dir}/{accession}_token_importances.png')
        plt.show()

        if length > 1000:
            seq = seq[:999]
        df = pd.DataFrame({'token': list(seq), 'Importance': grad_magnitudes.tolist()})
        df.to_csv(f'{self.save_dir}/{accession}_token_importances.csv', index=False)
        return pH


class ReferenceVisualizer:
    def __init__(self, tokenizer_path, model, save_dir='./', database='database.csv'):
        self.tokenizer = EsmTokenizer.from_pretrained(tokenizer_path)
        self.model = model
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.train()
        self.model.to(self.device)
        self.save_dir = save_dir
        self.database = pd.read_csv(database)

    def __call__(self, accession, seq):
        length = len(seq)
        inputs = self.tokenizer([seq], return_tensors="pt", padding='max_length', truncation=True, max_length=1000)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls_logits = outputs.cls_logits
        retrieval_indices = outputs.retrieval_indices
        retrieval_indices = torch.squeeze(retrieval_indices)
        retrieval_indices = retrieval_indices.cpu().detach().numpy().tolist()
        embeddings = outputs.embeddings
        rea_embeddings = outputs.rea_embeddings
        rea_labels = outputs.rea_labels
        rea_seqs = [self.database.loc[_, 'seq'] for _ in retrieval_indices]
        rea_accessions = [self.database.loc[_, 'accession'] for _ in retrieval_indices]
        embeddings = torch.squeeze(embeddings)
        distance = torch.abs(embeddings - rea_embeddings[0])

        # Visualize using heatmap


        plt.figure()
        sns.heatmap([distance.cpu().detach().numpy()], cmap='coolwarm', cbar=True)
        plt.title('Element-wise Euclidean Distance between Two Tensors')
        plt.show()

        plt.figure()
        tensors = torch.stack([embeddings, rea_embeddings[0]]).cpu().detach().numpy()
        pca = PCA(n_components=2)
        tensors_pca = pca.fit_transform(tensors)

        # Plot the 2D projection
        plt.scatter(tensors_pca[:, 0], tensors_pca[:, 1], c=['red', 'blue'])
        plt.text(tensors_pca[0, 0], tensors_pca[0, 1], 'Tensor 1', fontsize=12, color='red')
        plt.text(tensors_pca[1, 0], tensors_pca[1, 1], 'Tensor 2', fontsize=12, color='blue')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title('2D PCA of Two Tensors')
        plt.show()

        plt.figure()
        cos_sim = torch.nn.functional.cosine_similarity(embeddings, rea_embeddings[0], dim=0).item()

        # Visualize using heatmap
        sns.heatmap([[cos_sim]], annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title(f'Cosine Similarity: {cos_sim:.4f}')
        plt.show()

        return rea_accessions, rea_seqs
