from transformers import EsmModel, EsmTokenizer
from tqdm import tqdm
import torch
import numpy as np


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


class DatabaseBuilder:
    def __init__(self, pretrain_model_path):
        self.batch_size = 32
        self.tokenizer = EsmTokenizer.from_pretrained(pretrain_model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoder = EsmModel.from_pretrained(pretrain_model_path, ignore_mismatched_sizes=True)
        self.encoder.to(self.device)
        self.encoder.eval()

    @torch.inference_mode
    def build_embedding_database(self, seqs, pHs):
        database = []
        for i in tqdm(range(0, len(seqs), self.batch_size)):
            seq_batch = seqs[i:i + self.batch_size]
            pH_batch = pHs[i:i + self.batch_size]
            seq_inputs = self.tokenizer(seq_batch, return_tensors="pt", padding='max_length', truncation=True, max_length=1000)
            seq_inputs = {k: v.to(self.device) for k, v in seq_inputs.items()}
            embeddings = self.encoder(input_ids=seq_inputs['input_ids'], attention_mask=seq_inputs['attention_mask'])
            embeddings = embeddings.last_hidden_state
            embeddings = torch.mean(embeddings, dim=1) #shape: (batch_size, seq_len, dim) -> (batch_size, dim)
            pH_batch = torch.tensor(pH_batch).to(self.device)
            #concat embedding and pH_batch
            embeddings = torch.concat([embeddings, pH_batch.unsqueeze(1)], dim=1)
            database.append(embeddings)
        database = torch.cat(database, dim=0)
        #save database
        torch.save(database, 'database.pt')
        # np.save('database.npy', database.cpu().numpy())
        return database.shape
            


if __name__ == '__main__':
    builder = DatabaseBuilder('../pretrain_models/esm150')
    pH_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    headers_all = []
    sequences_all = []
    pHs = []
    for pH in pH_values:
        file = f'../pH_dataset/database/class{pH - 1}_all_filtered_results.fasta'
        headers, sequences = read_fasta(file)
        headers_all.extend(headers)
        sequences_all.extend(sequences)
        pHs.extend([pH-1]*len(headers))


    # shape = builder.build_embedding_database(sequences_all, pHs)
    # print(shape)

    accession = [_[1:].split()[0] for _ in headers_all]
    import pandas as pd
    database_csv = pd.DataFrame({'accession': accession, 'seq': sequences_all, 'ph': [_+1 for _ in pHs]})
    database_csv.to_csv('database.csv', index=False)