import torch
from models.ophreda import OpHReda, OphredaConfig
from utils import load_safetonsors_model, load_config
from transformers import EsmTokenizer


config = load_config('configs/opreda.yaml')
model_config = OphredaConfig(**config['model'])
model = OpHReda(model_config)
load_safetonsors_model(model, 'OpHReda_weight/model.safetensors')
tokenizer = EsmTokenizer.from_pretrained(model_config.tokenizer_path)


if __name__ == '__main__':
    accession = 'A2Z7C4'
    enzyme_seq = 'MENEFQDGKTEVIEAWYMDDSEEDQRLPHHREPKEFIHVDKLTELGVISWRLNPDNWENCENLKRIREARGYSYVDICDVCPEKLPNYETKIKSFFEEHLHTDEEIRYCLEGSGYFDVRDQNDQWIRIALKKGGMIVLPAGMYHRFTLDTDNYIKAMRLFVGDPVWTPYNRPHDHLPARKEFLAKLLKSEGENQAVEGF'
    inputs = tokenizer([enzyme_seq], return_tensors="pt", padding='max_length', truncation=True, max_length=1000)
    with torch.no_grad():
        outputs = model(**inputs)
    print(outputs.logits)







