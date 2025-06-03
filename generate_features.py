from collections import defaultdict, ChainMap
import time, itertools, h5py, pysam
import datetime, os, shutil, argparse, sys, re, array
import os
from itertools import repeat
import multiprocessing as mp
import numpy as np
from pathlib import Path
from numba import jit
import queue, gzip
import pod5 as p5
import utils

# Maps each DNA base to an integer for numerical processing.
base_to_num_map={'A':0, 'C':1, 'G':2, 'T':3,'N':4}

# Maps integers back to their corresponding bases (for reverse encoding).
num_to_base_map={0:'A', 1:'C', 2:'G', 3:'T', 4:'N'}

# Complementary base dictionary used to obtain the reverse strand.
comp_base_map={'A':'T','T':'A','C':'G','G':'C','[':']', ']':'['}

# Returns the reverse complement of a DNA sequence.
def revcomp(s):
    return ''.join(comp_base_map[x] for x in s[::-1])

# Extracts aligned pairs matching candidate motif positions in the reference.
# Adjusts coordinates based on strand orientation.
def get_candidates(read_seq, align_data, aligned_pairs, ref_pos_dict):
    is_mapped, is_forward, ref_name, reference_start, reference_end, read_length = align_data
    ref_motif_pos = ref_pos_dict[ref_name][0] if is_forward else ref_pos_dict[ref_name][1]
    common_pos = ref_motif_pos[(ref_motif_pos >= reference_start) & (ref_motif_pos < reference_end)]

    aligned_pairs_ref_wise = aligned_pairs[aligned_pairs[:, 1] != -1][common_pos - reference_start]
    aligned_pairs_ref_wise = aligned_pairs_ref_wise[aligned_pairs_ref_wise[:, 0] != -1]
    aligned_pairs_read_wise_original = aligned_pairs[aligned_pairs[:, 0] != -1]
    aligned_pairs_read_wise = np.copy(aligned_pairs_read_wise_original)

    if not is_forward:
        aligned_pairs_ref_wise = aligned_pairs_ref_wise[::-1]
        aligned_pairs_read_wise = aligned_pairs_read_wise[::-1]
        aligned_pairs_ref_wise[:, 0] = read_length - aligned_pairs_ref_wise[:, 0] - 1
        aligned_pairs_read_wise[:, 0] = read_length - aligned_pairs_read_wise[:, 0] - 1

    return aligned_pairs_ref_wise, aligned_pairs_read_wise_original

# Generates read-reference aligned pairs from CIGAR tuples.
# Marks insertions or deletions with -1.
@jit(nopython=True)
def get_aligned_pairs(cigar_tuples, ref_start):
    
    alen = np.sum(cigar_tuples[:, 0])
    pairs = np.zeros((alen, 2), dtype=np.int32)

    ref_cord = ref_start - 1
    read_cord = -1
    pair_cord = 0

    for i in range(len(cigar_tuples)):
        len_op, op = cigar_tuples[i, 0], cigar_tuples[i, 1]
        if op == 0:
            for _ in range(len_op):
                ref_cord += 1
                read_cord += 1
                pairs[pair_cord] = [read_cord, ref_cord]
                pair_cord += 1
        elif op == 2:
            for _ in range(len_op):
                read_cord += 1
                pairs[pair_cord] = [read_cord, -1]
                pair_cord += 1
        elif op == 1:
            for _ in range(len_op):
                ref_cord += 1
                pairs[pair_cord] = [-1, ref_cord]
                pair_cord += 1
    return pairs

# Encodes a reference sequence as integers (forward and reverse).
# Used to represent ACGT bases numerically.
@jit(nopython=True)
def get_ref_to_num(x):
    b=np.full((len(x)+1,2),fill_value=0,dtype=np.int8)
    
    for i,l in enumerate(x):
        if l=='A':
            b[i,0]=0
            b[i,1]=3
            
        elif l=='T':
            b[i,0]=3
            b[i,1]=0
            
        elif l=='C':
            b[i,0]=1
            b[i,1]=2
            
        elif l=='G':
            b[i,0]=2
            b[i,1]=1
            
        else:
            b[i,0]=4
            b[i,1]=4
    
    b[-1,0]=4
    b[-1,1]=4
            
    return b

# Fetches and encodes the sequence of a specific chromosome from the reference genome.
def get_ref_info(args):
    params, chrom = args
    ref_fasta = pysam.FastaFile(params['ref'])
    seq = ref_fasta.fetch(chrom).upper()
    seq_array = get_ref_to_num(seq)
    return chrom, seq_array, None, None

# Extracts statistics from the normalized signal between events defined by the move table.
# Computes quartile means, median, MAD, mean, standard deviation, and log-transformed duration.
@jit(nopython=True)
def get_events(signal, move, norm_type):
    stride, start, move_table = move

    if norm_type == 'mad':
        median = np.median(signal)
        mad = np.median(np.abs(signal - median))
        signal = (signal - median) / mad
    else:
        mean = np.mean(signal)
        std = np.std(signal)
        signal = (signal - mean) / std

    signal = np.clip(signal, -5, 5)

    move_len = len(move_table)
    move_index = np.where(move_table)[0]
    rlen = len(move_index)
    data = np.zeros((rlen, 9))

    for i in range(len(move_index) - 1):
        prev = move_index[i] * stride + start
        sig_end = move_index[i + 1] * stride + start
        sig_len = sig_end - prev

        data[i, 8] = np.log10(sig_len)
        segment = signal[prev:sig_end]
        data[i, 4] = np.median(segment)
        data[i, 5] = np.median(np.abs(segment - data[i, 4]))
        data[i, 6] = np.mean(segment)
        data[i, 7] = np.std(segment)

        for j in range(4):
            tmp_cnt = 0
            for t in range(j * sig_len // 4, min(sig_len, (j + 1) * sig_len // 4)):
                data[i, j] += signal[t + prev]
                tmp_cnt += 1
            if tmp_cnt > 0:
                data[i, j] = data[i, j] / tmp_cnt
            else:
                data[i, j] = 0

    return data

# Loads a list of annotated positions with labels (modification probabilities).
# Groups by chromosome and strand.
def get_pos(path):
    labelled_pos_list={}
    strand_map={'+':0, '-':1}
    
    with open(path) as file:
            for line in file:
                line=line.rstrip('\n').split('\t')
                if line[0] not in labelled_pos_list:
                    labelled_pos_list[line[0]]={0:{}, 1:{}}
  
                labelled_pos_list[line[0]][strand_map[line[2]]][int(line[1])]=float(line[3])
    
    return labelled_pos_list

# Saves the extracted data and labels into a `.npz` file for model training.
def write_to_npz(output_file_path, mat, base_qual, base_seq, label, ref_coordinates, read_name, ref_name, window, norm_type):
    np.savez(output_file_path, mat=mat, base_qual=base_qual, base_seq=base_seq, label=label, ref_coordinates=ref_coordinates, read_name=read_name, ref_name=ref_name, window=window, norm_type=norm_type, strides_per_base=1, model_depth=mat.shape[2], full_signal=False)

# Collects processed data from the output queue and saves it into `.npz` files.
# Batches, shuffles, and writes each chunk after a set number of reads.
def get_output(params, output_Q, process_event):
    output = params['output']
    reads_per_chunk = params['reads_per_chunk']

    chunk = 1
    read_count = 0
    output_file_path = os.path.join(output, f"{params['prefix']}.features.{chunk}.npz")

    mat, base_qual, base_seq, label = [], [], [], []
    ref_coordinates, read_name, ref_name = [], [], []

    while True:
        if process_event.is_set() and output_Q.empty():
            break
        else:
            try:
                res = output_Q.get(block=False)

                mat.append(res[0])
                base_qual.append(res[1])
                base_seq.append(res[2])
                ref_coordinates.append(res[3])
                label.append(res[4])
                read_name.append(res[5])
                ref_name.append(res[6])

                read_count += 1

                if read_count % reads_per_chunk == 0 and len(mat) > 0:
                    mat = np.vstack(mat)
                    base_qual = np.vstack(base_qual)
                    base_seq = np.vstack(base_seq).astype(np.int8)
                    label = np.hstack(label).astype(np.float16)
                    ref_coordinates = np.hstack(ref_coordinates)
                    read_name = np.hstack(read_name)
                    ref_name = np.hstack(ref_name)

                    idx = np.random.permutation(np.arange(len(label)))
                    mat = mat[idx]
                    base_qual = base_qual[idx]
                    base_seq = base_seq[idx]
                    label = label[idx]
                    ref_coordinates = ref_coordinates[idx]
                    read_name = read_name[idx]
                    ref_name = ref_name[idx]

                    print(f"{datetime.datetime.now()}: Number of reads processed = {read_count}.", flush=True)

                    write_to_npz(
                        output_file_path, mat, base_qual, base_seq,
                        label, ref_coordinates, read_name, ref_name,
                        window=params['window'], norm_type=params['norm_type']
                    )

                    chunk += 1
                    output_file_path = os.path.join(output, f"{params['prefix']}.features.{chunk}.npz")
                    mat, base_qual, base_seq, label = [], [], [], []
                    ref_coordinates, read_name, ref_name = [], [], []

            except queue.Empty:
                pass

    if read_count > 0 and len(mat) > 0:
        mat = np.vstack(mat)
        base_qual = np.vstack(base_qual)
        base_seq = np.vstack(base_seq).astype(np.int8)
        label = np.hstack(label).astype(np.float16)
        ref_coordinates = np.hstack(ref_coordinates)
        read_name = np.hstack(read_name)
        ref_name = np.hstack(ref_name)

        idx = np.random.permutation(np.arange(len(label)))
        mat = mat[idx]
        base_qual = base_qual[idx]
        base_seq = base_seq[idx]
        label = label[idx]
        ref_coordinates = ref_coordinates[idx]
        read_name = read_name[idx]
        ref_name = ref_name[idx]

        print(f"{datetime.datetime.now()}: Number of reads processed = {read_count}.", flush=True)

        write_to_npz(
            output_file_path, mat, base_qual, base_seq,
            label, ref_coordinates, read_name, ref_name,
            params['window'], params['norm_type']
        )

    return

# Extracts candidate modification sites, generates per-site features, and labels the data.
# Filters valid events, prepares the signal, and pushes processed data to the output queue for training.
def process(params, ref_pos_dict, signal_Q, output_Q, input_event, ref_seq_dict, labelled_pos_list):
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    window = params['window']
    window_range = np.arange(-window, window + 1)
    norm_type = params['norm_type']
    div_threshold = params['div_threshold']

    cigar_map = {'M': 0, '=': 0, 'X': 0, 'D': 1, 'I': 2, 'S': 2, 'H': 2, 'N': 1, 'P': 4, 'B': 4}
    cigar_pattern = r'\d+[A-Za-z]'

    seq_type = params['seq_type']
    ref_available = True if params['ref'] else False

    while True:
        if signal_Q.empty() and input_event.is_set():
            break

        try:
            data = signal_Q.get(block=False)
            signal, move, read_dict, align_data = data

            is_mapped, is_forward, ref_name, reference_start, reference_end, read_length = align_data

            fq = read_dict['seq']
            qual = read_dict['qual']
            sequence_length = len(fq)
            reverse = not is_forward
            fq = revcomp(fq) if reverse else fq
            qual = qual[::-1] if reverse else qual

            if is_mapped:
                cigar_tuples = np.array([(int(x[:-1]), cigar_map[x[-1]]) for x in re.findall(cigar_pattern, read_dict['cigar'])])
                ref_start = int(read_dict['ref_pos']) - 1
                aligned_pairs = get_aligned_pairs(cigar_tuples, ref_start)
            else:
                continue

            init_pos_list_candidates, read_to_ref_pairs = get_candidates(fq, align_data, aligned_pairs, ref_pos_dict)

            init_pos_list_candidates = init_pos_list_candidates[(init_pos_list_candidates[:, 0] > window) &
                                                                (init_pos_list_candidates[:, 0] < sequence_length - window - 1)] if len(init_pos_list_candidates) > 0 else init_pos_list_candidates

            if len(init_pos_list_candidates) == 0:
                continue

            base_seq = np.array([base_map[x] for x in fq])

            pos_list_candidates = init_pos_list_candidates

            if len(pos_list_candidates) == 0:
                continue
            else:
                print(f"{len(pos_list_candidates)} candidate positions extracted from reading {read_dict['name']}")

            if not move[0]:
                try:
                    tags = {x.split(':')[0]: x for x in read_dict.pop('tags')}
                    start = int(tags['ts'].split(':')[-1])
                    mv = tags['mv'].split(',')
                    stride = int(mv[1])
                    move_table = np.fromiter(mv[2:], dtype=np.int8)
                    move = (stride, start, move_table)
                    read_dict['tags'] = [x for x in tags.values() if x[:2] not in ['mv', 'ts', 'ML', 'MM']]
                except KeyError:
                    print('Read:%s no move table or stride or signal start found' % read_dict['name'])
                    continue

            base_qual = 10 ** ((33 - np.array([ord(x) for x in qual])) / 10)
            mean_qscore = -10 * np.log10(np.mean(base_qual))
            base_qual = (1 - base_qual)

            mat = get_events(signal, move, norm_type)

            per_site_features = np.array([mat[candidate[0] - window: candidate[0] + window + 1] for candidate in pos_list_candidates])
            per_site_base_qual = np.array([base_qual[candidate[0] - window: candidate[0] + window + 1] for candidate in pos_list_candidates])
            per_site_base_seq = np.array([base_seq[candidate[0] - window: candidate[0] + window + 1] for candidate in pos_list_candidates])

            per_site_ref_coordinates = pos_list_candidates[:, 1]
            per_site_label = np.array([labelled_pos_list[ref_name][1 - is_forward][coord] for coord in per_site_ref_coordinates])

            read_name_array = np.array([read_dict['name'] for _ in pos_list_candidates])
            ref_name_array = np.array([ref_name for _ in pos_list_candidates])

            read_chunks = [per_site_features, per_site_base_qual, per_site_base_seq,
                           per_site_ref_coordinates, per_site_label, read_name_array, ref_name_array]

            output_Q.put(read_chunks)

        except queue.Empty:
            pass

    return

# Reads signals from POD5 files and matches them with BAM alignments by read name.
# Filters reads by length and chromosome, then pushes them to the input queue for processing.
def get_input(params, signal_Q, output_Q, input_event):

    chrom_list = params['chrom']
    length_cutoff = params['length_cutoff']
    bam = params['bam']
    bam_file = pysam.AlignmentFile(bam, 'rb', check_sq=False)

    print('%s: Building BAM index.' % str(datetime.datetime.now()), flush=True)
    bam_index = pysam.IndexedReads(bam_file)
    bam_index.build()
    print('%s: Finished building BAM index.' % str(datetime.datetime.now()), flush=True)

    input_ = params['input']
    signal_files = [input_] if os.path.isfile(input_) else Path(input_).rglob("*.%s" % params['file_type'])

    counter = 0
    move = (None, None, None)

    for filename in signal_files:
        with p5.Reader(filename) as reader:
            for read in reader.reads():
                counter += 1
                if counter % 10000 == 0:
                    print('%s: Number of reads processed = %d.' % (str(datetime.datetime.now()), counter), flush=True)

                if signal_Q.qsize() > 10000:
                    time.sleep(10)

                read_name = str(read.read_id)
                try:
                    read_iter = bam_index.find(read_name)
                    for bam_read in read_iter:
                        if bam_read.flag & 0x900 == 0 and bam_read.reference_name in chrom_list and bam_read.query_length >= length_cutoff:
                            read_dict = bam_read.to_dict()
                            signal = read.signal
                            align_data = (
                                bam_read.is_mapped if params['ref'] else False,
                                bam_read.is_forward,
                                bam_read.reference_name,
                                bam_read.reference_start,
                                bam_read.reference_end,
                                bam_read.query_length
                            )
                            data = (signal, move, read_dict, align_data)
                            signal_Q.put(data)
                except KeyError:
                    continue

    input_event.set()
    return

# Coordinates the feature extraction pipeline for training.
# Loads labeled positions and launches parallel processes for input, processing, and writing data.
def call_manager(params):        
    
    bam = params['bam']
    bam_file = pysam.AlignmentFile(bam, 'rb', check_sq=False)
    header_dict = bam_file.header.to_dict()

    print('%s: Getting labelled positions from the reference.' % str(datetime.datetime.now()), flush=True)

    ref_seq_dict = {}
    ref_pos_dict = {}
    labelled_pos_list = {}

    if params['pos_list']:
        labelled_pos_list = get_pos(params['pos_list'])
        params['chrom'] = [x for x in params['chrom'] if x in labelled_pos_list.keys()]

    _ = get_ref_to_num('ACGT')

    with mp.Pool(processes=params['threads']) as pool:
        res = pool.map(get_ref_info, zip(repeat(params), params['chrom']))
        for r in res:
            chrom, seq_array, fwd_pos_array, rev_pos_array = r
            ref_seq_dict[chrom] = seq_array

            ref_pos_dict[chrom] = (
                np.array(sorted(list(labelled_pos_list[chrom][0].keys()))).astype(int),
                np.array(sorted(list(labelled_pos_list[chrom][1].keys()))).astype(int)
            )

    print('%s: Finished getting labelled positions from the reference.' % str(datetime.datetime.now()), flush=True)

    pmanager = mp.Manager()
    signal_Q = pmanager.Queue()
    output_Q = pmanager.Queue()
    process_event = pmanager.Event()
    input_event = pmanager.Event()

    handlers = []

    input_process = mp.Process(target=get_input, args=(params, signal_Q, output_Q, input_event))
    input_process.start()
    handlers.append(input_process)

    for hid in range(max(1, params['threads'] - 1)):
        p = mp.Process(target=process, args=(params, ref_pos_dict, signal_Q, output_Q, input_event, ref_seq_dict, labelled_pos_list))
        p.start()
        handlers.append(p)

    output_process = mp.Process(target=get_output, args=(params, output_Q, process_event))
    output_process.start()

    for job in handlers:
        job.join()

    process_event.set()
    output_process.join()

    return

# Main entry point of the script.
# Parses arguments, sets up parameters, and launches the feature generation pipeline.
if __name__ == '__main__':
   
    t = time.time()
    print('%s: Starting feature generation.' % str(datetime.datetime.now()))

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--bam", help='Path to BAM file with move table', type=str, required=True)
    parser.add_argument("--window", help='Number of bases before or after the base of interest to include in the model. Total will be 2xwindow+1.', type=int, default=10)
    parser.add_argument("--prefix", help='Prefix for the output files', type=str, default='output')
    parser.add_argument("--input", help='Path to folder containing POD5 files', type=str, required=True)
    parser.add_argument("--output", help='Path to folder where features will be stored', type=str, required=True)
    parser.add_argument("--threads", help='Number of processors to use', type=int, default=1)
    parser.add_argument("--div_threshold", help='Divergence Threshold.', type=float, default=0.25)
    parser.add_argument("--reads_per_chunk", help='Reads per chunk', type=int, default=100000)
    parser.add_argument("--ref", help='Path to reference FASTA file', type=str, required=True)
    parser.add_argument("--pos_list", help='File with positions and labels (chrom pos strand label)', type=str, required=True)
    parser.add_argument("--file_type", help='Signal file format', choices=['pod5'], type=str, required=True)
    parser.add_argument("--seq_type", help='Specify DNA sequencing only', choices=['dna'], type=str, required=True)
    parser.add_argument("--norm_type", help='Normalization method', choices=['mad', 'standard'], type=str, default='mad')
    parser.add_argument("--chrom", nargs='*', help='List of contigs to include (optional)')
    parser.add_argument("--length_cutoff", help='Minimum read length', type=int, default=0)

    args = parser.parse_args()

    if not args.output:
        args.output = os.getcwd()
    os.makedirs(args.output, exist_ok=True)

    if args.chrom:
        chrom_list = args.chrom
    else:
        chrom_list = pysam.Samfile(args.bam).references

    motif_seq = None
    motif_ind = None
    exp_motif_seq = None
    motif_label = None

    params = dict(
        bam=args.bam,
        seq_type=args.seq_type,
        window=args.window,
        pos_list=args.pos_list,
        ref=args.ref,
        input=args.input,
        norm_type=args.norm_type,
        motif_seq=motif_seq,
        motif_ind=motif_ind,
        exp_motif_seq=exp_motif_seq,
        motif_label=motif_label,
        file_type=args.file_type,
        chrom=chrom_list,
        threads=args.threads,
        length_cutoff=args.length_cutoff,
        output=args.output,
        prefix=args.prefix,
        div_threshold=args.div_threshold,
        reads_per_chunk=args.reads_per_chunk)

    print(args)
    with open(os.path.join(args.output, 'args'), 'w') as file:
        file.write('Command: python %s\n\n\n' % (' '.join(sys.argv)))
        file.write('------Parameters Used For Running CUNA------\n')
        for k in vars(args):
            file.write('{}: {}\n'.format(k, vars(args)[k]))

    call_manager(params)
    print('\n%s: Time elapsed=%.4fs' % (str(datetime.datetime.now()), time.time() - t))