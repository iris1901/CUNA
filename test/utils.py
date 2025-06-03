from subprocess import PIPE, Popen
import os, shutil, pysam, sys, datetime, re, pickle
import numpy as np
from numba import jit
import torch
from itertools import repeat
import pysam
from .models import *
import torch.nn.utils.prune as prune
from tqdm import tqdm

comp_base_map={'A':'T','T':'A','C':'G','G':'C','[':']', ']':'['}

def revcomp(s):
    return ''.join(comp_base_map[x] for x in s[::-1])
 
def get_model(params):
    try:
        model_config_path, model_path = params['model'].split(',')
    except ValueError:
        print(f"[ERROR] --model debe tener formato: config_path,checkpoint_path", flush=True)
        sys.exit(2)

    with open(model_config_path, 'rb') as handle:
        model_config = pickle.load(handle)

    if model_config['model_type'] == 'bilstm':
        model = BiLSTM(model_dims=model_config['model_dims'],
                       num_layers=model_config['num_layers'],
                       dim_feedforward=model_config['dim_feedforward'],
                       num_fc=model_config['num_fc'],
                       embedding_dim=model_config['embedding_dim'],
                       embedding_type=model_config['embedding_type'],
                       fc_type=model_config['fc_type'])

        checkpoint = torch.load(model_path, map_location=params['dev'])
        model.load_state_dict(checkpoint['model_state_dict'])

        if not params['disable_pruning']:
            module = model.classifier.fc
            prune.l1_unstructured(module, name="weight", amount=0.95)
            prune.remove(module, 'weight')

        return model, model_config

    elif model_config['model_type'] == 'transformer':
        model = TransformerModel(model_dims=model_config['model_dims'],
                                 num_layers=model_config['num_layers'],
                                 dim_feedforward=model_config['dim_feedforward'],
                                 num_fc=model_config['num_fc'],
                                 embedding_dim=model_config['embedding_dim'],
                                 embedding_type=model_config['embedding_type'],
                                 pe_dim=model_config['pe_dim'],
                                 nhead=model_config['nhead'],
                                 pe_type=model_config['pe_type'],
                                 fc_type=model_config['fc_type'])

        checkpoint = torch.load(model_path, map_location=params['dev'])
        model.load_state_dict(checkpoint['model_state_dict'])

        if not params['disable_pruning']:
            module = model.classifier.fc
            prune.l1_unstructured(module, name="weight", amount=0.5)
            prune.remove(module, 'weight')
            for l in model.transformer_encoder.layers:
                for attr in ['linear1', 'linear2', 'self_attn.out_proj']:
                    module = eval(f'l.{attr}')
                    prune.l1_unstructured(module, name="weight", amount=0.25)
                    prune.remove(module, 'weight')

        return model, model_config

    else:
        print(f"[ERROR] Modelo no reconocido: {model_config['model_type']}", flush=True)
        sys.exit(2)

def generate_batches(features, base_seq, window, batch_size=512):
    features = torch.Tensor(features)
    base_seq = torch.Tensor(base_seq).type(torch.LongTensor)

    for local_index in range(0, features.shape[0], batch_size):
        batch_x = features[local_index:(local_index + batch_size)]
        batch_base_seq = base_seq[local_index:(local_index + batch_size)]

        yield batch_x, batch_base_seq

@jit(nopython=True)
def get_aligned_pairs(cigar_tuples, ref_start):
    alen=np.sum(cigar_tuples[:,0])
    pairs=np.zeros((alen,2)).astype(np.int32)

    i=0
    ref_cord=ref_start-1
    read_cord=-1
    pair_cord=0
    for i in range(len(cigar_tuples)):
        len_op, op= cigar_tuples[i,0], cigar_tuples[i,1]
        if op==0:
            for k in range(len_op):            
                ref_cord+=1
                read_cord+=1

                pairs[pair_cord,0]=read_cord
                pairs[pair_cord,1]=ref_cord
                pair_cord+=1

        elif op==2:
            for k in range(len_op):            
                read_cord+=1            
                pairs[pair_cord,0]=read_cord
                pairs[pair_cord,1]=-1
                pair_cord+=1

        elif op==1:
            for k in range(len_op):            
                ref_cord+=1            
                pairs[pair_cord,0]=-1
                pairs[pair_cord,1]=ref_cord
                pair_cord+=1
    return pairs

def motif_check(motif):
    nt_dict = {
        'R': 'GA', 'Y': 'CT', 'K': 'GT', 'M': 'AC',
        'S': 'GC', 'W': 'AT', 'B': 'GTC', 'D': 'GAT',
        'H': 'ACT', 'V': 'GCA', 'N': 'AGCT'}

    valid_alphabet = set(nt_dict.keys()).union({'A', 'C', 'G', 'T'})

    motif_seq, exp_motif_seq, final_motif_ind, valid = None, None, None, False

    if len(motif) < 2:
        print('--motif not specified correctly. You need to specify a motif and at least one index', flush=True)
        return motif_seq, exp_motif_seq, final_motif_ind, valid

    elif len(set(motif[0]) - valid_alphabet) > 0:
        print('--motif not specified correctly. Motif should only consist of the following extended nucleotide letters: {}'.format(','.join(valid_alphabet)), flush=True)
        return motif_seq, exp_motif_seq, final_motif_ind, valid

    elif not all(a.isnumeric() for a in motif[1:]):
        print('--motif not specified correctly. Motif indices should be integers separated by whitespace and shoud come after the motif sequence.', flush=True)
        return motif_seq, exp_motif_seq, final_motif_ind, valid

    else:
        motif_seq = motif[0]
        motif_ind = [int(x) for x in motif[1:]]

        if len(set(motif_seq[x] for x in motif_ind)) != 1 or len(set(motif_seq[x] for x in motif_ind) - set('ACGT')) > 0:
            print('Base of interest should be same for all indices and must be one of A, C, G or T.', flush=True)
            return motif_seq, exp_motif_seq, final_motif_ind, valid

        exp_motif_seq = motif_seq
        for nt in nt_dict:
            if nt in exp_motif_seq:
                exp_motif_seq = exp_motif_seq.replace(nt, '[{}]'.format(nt_dict[nt]))

        return motif_seq, exp_motif_seq, motif_ind, True

def get_output(params, input_list):
    qscore_cutoff = params['qscore_cutoff']
    length_cutoff = params['length_cutoff']

    mod_threshold = params['mod_t']
    unmod_threshold = params['unmod_t']

    total_files = len(input_list)
    print('%s: Reading %d files.' % (str(datetime.datetime.now()), total_files), flush=True)
    pbar = tqdm(total=total_files)

    per_site_pred = {}

    for read_pred_file in input_list:
        with open(read_pred_file, 'r') as read_file:
            read_file.readline()
            for line in read_file:
                fields = line.rstrip('\n').split('\t')
                if len(fields) < 10:
                    continue

                read, chrom, pos, pos_after, read_pos, strand, score, mean_qscore, sequence_length, _ = fields

                if pos == 'NA' or float(mean_qscore) < qscore_cutoff or int(sequence_length) < length_cutoff:
                    continue

                score = float(score)
                if score < mod_threshold and score > unmod_threshold:
                    continue
                else:
                    mod = score >= mod_threshold

                pos = int(pos)

                if (chrom, pos) not in per_site_pred:
                    per_site_pred[(chrom, pos)] = [0, 0]

                per_site_pred[(chrom, pos)][mod] += 1

        pbar.update(1)
    pbar.close()

    print('%s: Writing Per Site Modification Output.' % str(datetime.datetime.now()), flush=True)

    per_site_fields = ['#chromosome', 'position', 'coverage', 'mod_coverage', 'unmod_coverage', 'mod_fraction']
    per_site_header = '\t'.join(per_site_fields) + '\n'
    per_site_file_path = os.path.join(params['output'], '%s.per_site' % params['prefix'])

    with open(per_site_file_path, 'w') as per_site_file:
        per_site_file.write(per_site_header)
        for x in sorted(per_site_pred.keys()):
            chrom, pos = x
            unmod, mod = per_site_pred[x]
            cov = unmod + mod
            frac = mod / cov if cov > 0 else 0
            per_site_file.write('%s\t%d\t%d\t%d\t%d\t%.4f\n' % (chrom, pos, cov, mod, unmod, frac))

    print('%s: Finished Writing Per Site Modification Output.' % str(datetime.datetime.now()), flush=True)
    print('%s: Per Site Prediction file: %s' % (str(datetime.datetime.now()), per_site_file_path), flush=True)

def get_per_site(params, input_list):
    print('%s: Starting Per Site Modification Detection.' % str(datetime.datetime.now()), flush=True)
    get_output(params, input_list)