import pandas as pd
import random
import os

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
def load_data(seq_type):
    seqs = list(set(pd.read_csv(rf'datasets\{seq_type}.csv')['Sequence'].tolist()))
    if seq_type == 'nonAVPs':
        set_seed(42)
        seqs.sort()
        indexes = random.sample(range(len(seqs)), 154)
        seqs_ = []
        for index in indexes:
            seqs_.append(seqs[index])
        seqs_.sort()
        seqs = seqs_
    seqs.sort()
    return seqs
def replacement_dictionary(seqs, p, seq_type):
    if seq_type == 'nonAVPs':
        set_seed(42)
    rep_dict = [['A', 'V'], ['S', 'T'], ['F', 'Y'], ['K', 'R'], ['C', 'M'], ['D', 'E'], ['N', 'Q'], ['V', 'I']]
    probability0 = random.random()
    seqs_plus = []
    if probability0 > p:
        indexes = random.sample(range(0, len(seqs)), round(len(seqs) * (probability0 - p)))
        for index in indexes:
            for rep in rep_dict:
                probability1 = random.random()
                if probability1 > p:
                    new_seq = seqs[index].replace(rep[0], rep[1])
                    seqs_plus.append(new_seq)
                else:
                    pass
    else:
        pass
    return seqs_plus, seqs + seqs_plus
def replacement_alanine(seqs, p, seq_type):
    if seq_type == 'nonAVPs':
        set_seed(42)
    probability0 = random.random()
    seqs_plus = []
    if probability0 > p:
        indexes = random.sample(range(0, len(seqs)), round(len(seqs) * (probability0 - p)))
        for index in indexes:
            amino_index = random.sample(range(0, len(seqs[index])), round(len(seqs[index]) * (1 - probability0)))
            new_seq = list(seqs[index])
            for amino_id in amino_index:
                new_seq[amino_id] = 'A'
            new_seq = ''.join(new_seq)
            seqs_plus.append(new_seq)
    else:
        pass
    return seqs_plus, seqs + seqs_plus
def global_random_shuffling(seqs, p, seq_type):
    if seq_type == 'nonAVPs':
        set_seed(42)
    probability0 = random.random()
    seqs_plus = []
    if probability0 > p:
        indexes = random.sample(range(0, len(seqs)), round(len(seqs) * (probability0 - p)))
        for index in indexes:
            str_list = list(seqs[index])
            random.shuffle(str_list)
            seqs_plus.append(''.join(str_list))
    else:
        pass
    return seqs_plus, seqs + seqs_plus
def Local_seq_shuffling(seqs, p, seq_type):
    if seq_type == 'nonAVPs':
        set_seed(42)
    probability0 = random.random()
    seqs_plus = []
    if probability0 > p:
        indexes = random.sample(range(0, len(seqs)), round(len(seqs) * (probability0 - p)))
        for index in indexes:
            seq1 = seqs[index][:round((probability0 - p) * len(seqs[index]))]
            seq2 = seqs[index][round((probability0 - p) * len(seqs[index])):-round((probability0 - p) * len(seqs[index]))]
            seq3 = seqs[index][-round((probability0 - p) * len(seqs[index])):]
            seq2_str_list = list(seq2)
            random.shuffle(seq2_str_list)
            new_seq = seq1 + ''.join(seq2_str_list) + seq3
            seqs_plus.append(new_seq)
    else:
        pass
    return seqs_plus, seqs + seqs_plus
def sequence_reversion(seqs, p, seq_type):
    if seq_type == 'nonAVPs':
        set_seed(42)
    probability0 = random.random()
    seqs_plus = []
    if probability0 > p:
        indexes = random.sample(range(0, len(seqs)), round(len(seqs) * (probability0 - p)))
        for index in indexes:
            new_seq = seqs[index][::-1]
            seqs_plus.append(new_seq)
    else:
        pass
    return seqs_plus, seqs + seqs_plus
def subsampling(seqs, p, seq_type):
    if seq_type == 'nonAVPs':
        set_seed(42)
    probability0 = random.random()
    seqs_plus = []
    if probability0 > p:
        indexes = random.sample(range(0, len(seqs)), round(len(seqs) * (probability0 - p)))
        for index in indexes:
            new_seq = seqs[index][round(len(seqs[index]) * (probability0 - p)):-round(len(seqs[index]) * (probability0 - p))]
            seqs_plus.append(new_seq)
    else:
        pass
    return seqs_plus, seqs + seqs_plus
def augmentation_combination(seqs, p, seq_type):
    if seq_type == 'nonAVPs':
        set_seed(42)
    plus, seq_1 = replacement_dictionary(seqs, p, seq_type)
    plus, seq_2 = replacement_alanine(seqs, p, seq_type)
    plus, seq_3 = global_random_shuffling(seqs, p, seq_type)
    plus, seq_4 = Local_seq_shuffling(seqs, p, seq_type)
    plus, seq_5 = sequence_reversion(seqs, p, seq_type)
    plus, seq_6 = subsampling(seqs, p, seq_type)
    seqs = seq_1 + seq_2 + seq_3 + seq_4 + seq_5 + seq_6
    seqs = list(set(seqs))
    return seqs
def load(p, seq_type):
    set_seed(42)
    original_seqs = load_data(seq_type)
    augmented_seqs = augmentation_combination(original_seqs, p, seq_type)
    augmented_seqs = list(set(augmented_seqs))
    return augmented_seqs