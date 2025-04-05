import torch
from models.ophreda import Model, Config
from utils import load_safetonsors_model, load_config
from transformers import EsmTokenizer
from Bio import SeqIO

import sys

# Load config and model
config = load_config('configs/ophreda.yaml')
model_config = Config(**config['model'])
model = Model(model_config)
model.eval()
load_safetonsors_model(model, 'OpHReda_weight/model.safetensors')
tokenizer = EsmTokenizer.from_pretrained(model_config.tokenizer_path)

def predict_sequence(seq):
    inputs = tokenizer([seq], return_tensors="pt", padding='max_length', truncation=True, max_length=1000)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits.item()  # Assuming batch size = 1

if __name__ == '__main__':
    fasta_path = sys.argv[1]  # Get fasta file from command line, e.g. `python predict.py input.fasta`

    for record in SeqIO.parse(fasta_path, "fasta"):
        seq_id = record.id
        sequence = str(record.seq)
        prediction = predict_sequence(sequence)
        print(f"{seq_id}\t{prediction:.4f}")


