import os
import argparse
import sys
sys.path.append("/storage/yuanfajieLab/yuanfajie/my_project/analysis/clip_score")
from model.claprot.claprot_trimodal_model import CLAProtTrimodalModel
import torch
from tqdm import tqdm
import os
import numpy as np
import json

def load_txt_dir(txt_dir):
    # loading ground truth structure token, sampled structrue token and descriptions 
    gts = {}
    descriptions = {}
    samples = {}
    txt_files = [i for i in os.listdir(txt_dir) if i.endswith("txt")]
    for txt_file in txt_files:
        f = open(os.path.join(txt_dir, txt_file), 'r')
        lines = f.readlines()
        description = lines[0][len("descriptions:"): ].strip()
        gt = lines[1][len(">Ground_truth: "):].strip()
        gts[txt_file] = gt
        descriptions[txt_file] = description

        samples_list =[]
        for line in lines[2:]:
            if line.startswith('>'):
                continue
            seq_list = line.strip().split()
            seq_list = [i for i in seq_list if not i.startswith("<")]
            samples_list.append("".join(seq_list))
        samples[txt_file] = samples_list
    return samples, gts, descriptions

def evaluate_clip_score(model, seq, pred_text, seq_type='prot'):
    if not isinstance(seq, list):
        seq = [seq]
    if not isinstance(pred_text, list):
        pred_text = [pred_text]

    with torch.no_grad():
        if seq_type == 'prot':
            seq_repr = model.get_protein_repr(seq)
        else:
            seq_repr = model.get_structure_repr(seq)
        pred_text_repr = model.get_text_repr(pred_text)

        sim_pred_text = torch.matmul(pred_text_repr, seq_repr.T) / model.temperature

        sim_pred_mask = torch.eye(sim_pred_text.size(0), device=sim_pred_text.device)
        sim_pred_text = sim_pred_text.masked_fill(sim_pred_mask == 0, -1e9)
        sim_pred_text = sim_pred_text.max(dim=1)[0]
        return sim_pred_text



def load_CLIP():
    # loading model
    model_config = {
        "protein_config": "/storage/yuanfajieLab/yuanfajie/my_project/analysis/ProTrekWeight/esm2_t33_650M_UR50D",
        "text_config": "/storage/yuanfajieLab/yuanfajie/my_project/analysis/ProTrekWeight/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "structure_config": "/storage/yuanfajieLab/yuanfajie/my_project/analysis/ProTrekWeight/foldseek_t30_150M",
        "from_checkpoint": "/storage/yuanfajieLab/yuanfajie/my_project/analysis/ProTrekWeight/ProTrek_650M_UniRef50.pt",
        "load_protein_pretrained": False,
        "load_text_pretrained": False,
    }
    model = CLAProtTrimodalModel(**model_config)
    model.to("cuda").eval()
    return model



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('jsonl', type=str)
    args = parser.parse_args()

    clip_model = load_CLIP()
    ###
    res = []
    jsonl_path = args.jsonl
    with open(jsonl_path, 'r') as f:
        for line in f:
            res.append(json.loads(line))
    print("all of designs", len(res))

    ###
    mean_scores = []
    idx2score = {}
    for design_dic in tqdm(res):
        idx = design_dic["idx"]
        seq = design_dic["seq"]
        text = design_dic["text"]
        scores = evaluate_clip_score(clip_model, seq, text, seq_type='prot').cpu().numpy().item()
        mean_scores.append(scores)
        idx2score[idx] = scores

    print(f"Mean score: {np.mean(mean_scores)}")

    results_path = jsonl_path.replace(".jsonl", "protrek_score.json")
    
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            result_dic = json.load(f)
    else:
        result_dic = {}
    result_dic[f"mean sequence text clip score"] = float(np.mean(mean_scores))
    result_dic[f"clip score"] = idx2score
    with open(results_path, 'w') as f:
        json.dump(result_dic, f, indent=4)
    

if __name__ == "__main__":
    main()