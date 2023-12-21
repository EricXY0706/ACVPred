import pandas as pd
import numpy as np
from augmentation import load
from model import Encoder_Classifier
import torch
from gensim.models import Word2Vec
import random
import os
import argparse

def ACVPred(input_file, output_file):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    def set_seed(seed=42):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    def load_data():
        set_seed(seed=42)
        ACVP = load(p=0.1, seq_type='ACVPs')
        nonAVP = load(p=0.2, seq_type='nonAVPs')
        max_prot_length = 100
        ACVP.sort()
        nonAVP.sort()
        ACVP_tags, nonAVP_tags = [1] * len(ACVP), [0] * len(nonAVP)
        seqs = ACVP + nonAVP
        tags = ACVP_tags + nonAVP_tags
        zipped = list(zip(seqs, tags))
        random.shuffle(zipped)
        seqs, tags = zip(*zipped)
        seqs = list(seqs)
        tags = list(tags)
        return seqs, tags, max_prot_length
    def Seq_Embedding(query):
        x, Y, max_prot_length = load_data()
        x.append(query)
        kmer, emb_length = 2, 100
        max_seq_length = max_prot_length - kmer + 1
        # Dictionary
        st_for_seqs, pad_index_for_seqs = [], np.arange(len(x) * max_seq_length ** 2).reshape(len(x), max_seq_length ** 2)
        for seq in x:
            st_for_seq = []
            for i in range(len(seq)):
                st_for_seq.append(seq[i:i + kmer])
                if i + kmer == len(seq):
                    break
            st_for_seq.extend('Z' * (max_seq_length - len(st_for_seq)))
            st_for_seqs.append(st_for_seq)
        # Padding Index
        aa = [one * max_seq_length for one in st_for_seqs]
        for i in range(len(aa)):
            for j in range(max_seq_length ** 2):
                pad_index_for_seqs[i][j] = aa[i][j] == 'Z'
        pad_index_for_seqs = torch.from_numpy(pad_index_for_seqs.reshape(len(x), max_seq_length, max_seq_length))
        # W2V + PE
        w2v_model = Word2Vec(st_for_seqs, vector_size=emb_length, window=25, min_count=0, seed=42, workers=1, sg=1, hs=0, negative=10)
        vec_for_query = np.arange(emb_length)
        for j in range(len(st_for_seqs[-1])):
            pos_table = np.array([j / np.power(10000, 2 * k / emb_length) for k in range(emb_length)])
            pos_table[0::2] = np.sin(pos_table[0::2])
            pos_table[1::2] = np.cos(pos_table[1::2])
            emb_vec = w2v_model.wv.get_vector(st_for_seqs[-1][j])
            vec_for_query = np.vstack((vec_for_query, emb_vec + pos_table))
        vec_for_query, pad_for_query = torch.from_numpy(vec_for_query[1:, :]), pad_index_for_seqs[-1, :, :]
        vec_for_query = (vec_for_query - torch.min(vec_for_query, dim=1)[0].unsqueeze(1)) / (
                torch.max(vec_for_query, dim=1)[0] - torch.min(vec_for_query, dim=1)[0]).unsqueeze(1)
        vec_for_query, pad_for_query = vec_for_query.to(device), pad_for_query.to(device)
        return vec_for_query, pad_for_query
    def Seq_Query(vec_for_query, pad_for_query):
        PATH = r'model.pth'
        model = Encoder_Classifier(emb_length=100, n_heads=20, hidden=50, dropout=0, n_encoder=20).to(device)
        model.load_state_dict(torch.load(PATH))
        model.eval()
        y = model(vec_for_query.to(torch.float32), pad_for_query.to(torch.float32))
        classification = {0: 'an ACVP', 1: 'a non-ACVP'}
        label = torch.argmax(y, -1).item()
        probability = round(y[0].item(), 4)
        return classification[label], probability
    f = open(input_file)
    lines = f.readlines()
    query_seqs, cfs, probs = [], [], []
    for line in lines:
        if line[0] != '>':
            query_seqs.append(line)
    for seq in query_seqs:
        print(f'Now predicting sequence: {seq}')
        vec, pad = Seq_Embedding(seq)
        cf, prob = Seq_Query(vec, pad)
        cfs.append(cf.split(' ')[1])
        probs.append(prob)
        print(f'it is {cf}, and the probability is {prob}.')
    df = pd.DataFrame()
    df['Sequence'] = pd.Series(query_seqs)
    df['Prediction'] = pd.Series(cfs)
    df['Probability'] = pd.Series(probs)
    df.to_csv(rf'{output_file}', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ACVPred')
    parser.add_argument('-i', '--input', help='Input FASTA file', required=True)
    parser.add_argument('-o', '--output', help='Output file in csv format', required=True)
    args = parser.parse_args()
    InputFile = args.input
    OutputFile = args.output
    ACVPred(InputFile, OutputFile)