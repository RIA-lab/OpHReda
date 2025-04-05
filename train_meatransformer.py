import os
import yaml
import argparse
from transformers import Trainer, TrainingArguments
import numpy as np
import wandb
from models.meatransformer import Config, Model, Collator
from dataset_pH import DatasetPh
from torch.optim import AdamW
from utils import (freeze_model,
                   count_parameters,
                   cls_metrics,)
from sklearn.metrics import classification_report


def parse_args():
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument('--config', type=str, required=False, default='configs/meatransformer.yaml', help='Path to the YAML config file')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    config['training']['lr'] = float(config['training']['lr'])
    return config


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)

    dataset_train = DatasetPh(config['dataset']['train'])
    dataset_val = DatasetPh(config['dataset']['val'])
    dataset_test = DatasetPh(config['dataset']['test'])

    print(len(dataset_train))
    print(len(dataset_val))
    print(len(dataset_test))

    model_config = Config(**config['model'])
    model = Model(model_config)
    collate_fn = Collator(model_config.tokenizer_path)
    freeze_model(model.pretrain_model)
    print(f'trainable parameters: {round(count_parameters(model) / 1000000, 2)}M')
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config['training']['lr'])

    wandb.init(project=config['wandb']['project'])
    wandb.run.name = config['wandb']['run_name']
    args = TrainingArguments(
        output_dir=f'output/{wandb.run.name}',
        logging_dir=f'output/{wandb.run.name}/log',
        logging_strategy='epoch',
        learning_rate=config['training']['lr'],
        per_device_train_batch_size=config['training']['train_batch_size'],
        per_device_eval_batch_size=config['training']['eval_batch_size'],
        num_train_epochs=config['training']['num_epochs'],
        weight_decay=config['training']['weight_decay'],
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        dataloader_num_workers=config['training']['dataloader_num_workers'],
        dataloader_pin_memory=config['training']['dataloader_pin_memory'],
        run_name=wandb.run.name,
        overwrite_output_dir=True,
        save_total_limit=config['training']['save_total_limit'],
        remove_unused_columns=False,
        report_to=["wandb"],
        fp16=config['training']['fp16'],
        metric_for_best_model='f1',
        greater_is_better=True,
        max_grad_norm=config['training']['max_grad_norm'],
    )

    trainer = Trainer(
        model=model,
        optimizers=(optimizer, None),
        args=args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        data_collator=collate_fn,
        compute_metrics=cls_metrics,
    )

    trainer.train(resume_from_checkpoint=False)

    if not os.path.exists(f'results/{wandb.run.name}/'):
        os.makedirs(f'results/{wandb.run.name}/')

    datasets = {'train': dataset_train, 'val': dataset_val, 'test': dataset_test}
    for k, v in datasets.items():
        print(f'------------------------{k}--------------------------')
        predictions, labels, metrics = trainer.predict(v)
        report = classification_report(labels, np.argmax(predictions, axis=-1))
        print(report)

        with open(f'results/{wandb.run.name}/{k}_report.txt', 'w') as f:
            f.write(report)

        if k == 'test':
            np.save(f'results/{wandb.run.name}/predictions.npy', predictions)
            np.save(f'results/{wandb.run.name}/labels.npy', labels)

    wandb.finish()