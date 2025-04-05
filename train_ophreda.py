import os
import yaml
import argparse
from transformers import Trainer, TrainingArguments
import numpy as np
import wandb
from models.ophreda import Config, Model, Collator
from dataset_pH import DatasetPh
from torch.optim import AdamW
from utils import (freeze_model,
                   count_parameters,
                   write_json,
                   reg_metrics,
                   load_safetonsors_model)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Ophreda model")
    parser.add_argument('--config', type=str, required=False, default='configs/ophreda.yaml', help='Path to the YAML config file')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    config['training']['lr'] = float(config['training']['lr'])
    return config


if __name__ == '__main__':
    args = parse_args()
    config_train = load_config(args.config)
    config_model = Config()

    dataset_train = DatasetPh(config_train['dataset']['train'])
    dataset_val = DatasetPh(config_train['dataset']['val'])
    dataset_test = DatasetPh(config_train['dataset']['test'])

    print(len(dataset_train))
    print(len(dataset_val))
    print(len(dataset_test))

    model = Model(config_model)
    load_safetonsors_model(model, 'output/meatransformer/checkpoint-1115/model.safetensors', False)
    load_safetonsors_model(model, 'output/mrlat/checkpoint-8855/model.safetensors', False)
    collate_fn = Collator(config_model.tokenizer_path)
    freeze_model(model.pretrain_model)
    freeze_model(model.mea)
    freeze_model(model.mrlat)
    print(f'trainable parameters: {round(count_parameters(model) / 1000000, 2)}M')
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config_train['training']['lr'])

    wandb.init(project=config_train['wandb']['project'])
    wandb.run.name = config_train['wandb']['run_name']
    args = TrainingArguments(
        output_dir=f'output/{wandb.run.name}',
        logging_dir=f'output/{wandb.run.name}/log',
        logging_strategy='epoch',
        learning_rate=config_train['training']['lr'],
        per_device_train_batch_size=config_train['training']['train_batch_size'],
        per_device_eval_batch_size=config_train['training']['eval_batch_size'],
        num_train_epochs=config_train['training']['num_epochs'],
        weight_decay=config_train['training']['weight_decay'],
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        dataloader_num_workers=config_train['training']['dataloader_num_workers'],
        dataloader_pin_memory=config_train['training']['dataloader_pin_memory'],
        run_name=wandb.run.name,
        overwrite_output_dir=True,
        save_total_limit=config_train['training']['save_total_limit'],
        remove_unused_columns=False,
        report_to=["wandb"],
        fp16=config_train['training']['fp16'],
        max_grad_norm=config_train['training']['max_grad_norm'],
    )

    trainer = Trainer(
        model=model,
        optimizers=(optimizer, None),
        args=args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        data_collator=collate_fn,
        compute_metrics=reg_metrics,
    )

    trainer.train(resume_from_checkpoint=False)

    if not os.path.exists(f'results/{wandb.run.name}/'):
        os.makedirs(f'results/{wandb.run.name}/')

    datasets = {'train': dataset_train, 'val': dataset_val, 'test': dataset_test}
    for k, v in datasets.items():
        print(f'------------------------{k}--------------------------')
        predictions, labels, metrics = trainer.predict(v)

        write_json(metrics, f'results/{wandb.run.name}/{k}_metrics.json')

        if k == 'test':
            np.save(f'results/{wandb.run.name}/predictions.npy', predictions)
            np.save(f'results/{wandb.run.name}/labels.npy', labels)

    wandb.finish()