import torch
from dataset_pH import DatasetPh, DatasetPhClassification
# from models.meatransformer import Model, Collator, Config
from models.mrlat import Model, Collator, Config
from torch.utils.data import DataLoader
import yaml


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    config['training']['lr'] = float(config['training']['lr'])
    return config


config = load_config('configs/mrlat.yaml')
model_config = Config(**config['model'])
model = Model(model_config)
model.to('cuda')
collate_fn = Collator(model_config.tokenizer_path)
dataset = DatasetPhClassification(config['dataset']['train'])
dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

for batch in dataloader:
    batch = {k: v.to('cuda') for k, v in batch.items()}
    print(batch)
    with torch.no_grad():
        outputs = model(**batch)
    print(outputs)
    break