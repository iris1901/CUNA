from collections import defaultdict, ChainMap
import time, itertools, h5py, pysam
import datetime, os, shutil, argparse, sys, re, array
import os
import multiprocessing as mp
import numpy as np
from pathlib import Path
from .utils import *
from numba import jit
import queue
import pod5 as p5
import torch

@jit(nopython=True)
def get_segment_events(signal, move, norm_type):
    stride, start, move_table = move

    if norm_type == 'mad':
        median = np.median(signal)
        mad = np.median(np.abs(signal - median))
        signal = (signal - median) / mad if mad != 0 else np.zeros_like(signal)
    else:
        mean = np.mean(signal)
        std = np.std(signal)
        signal = (signal - mean) / std if std != 0 else np.zeros_like(signal)

    signal = np.clip(signal, -5, 5)

    move_len = len(move_table)
    data = np.zeros((move_len, stride + 2))
    indexes = np.full(move_len, fill_value=0, dtype=np.int32)
    z = 1
    idx = -1

    segments = np.full(np.sum(move_table), fill_value=0, dtype=np.int32)

    for i in range(move_len):
        if move_table[i]:
            z ^= 1
            idx += 1
            segments[idx] = i

        data[i, z] = 1
        indexes[i] = idx

        for k in range(stride):
            if start + i * stride + k < len(signal):
                data[i, 2 + k] = signal[start + i * stride + k]

    return data, indexes, segments

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

    move_index = np.where(move_table)[0]
    rlen = len(move_index)
    data = np.zeros((rlen, 9))

    for i in range(len(move_index) - 1):
        prev = move_index[i] * stride + start
        sig_end = move_index[i + 1] * stride + start
        sig_len = sig_end - prev
        segment = signal[prev:sig_end]

        data[i, 8] = np.log10(sig_len) if sig_len > 0 else 0
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
                data[i, j] /= tmp_cnt
            else:
                data[i, j] = 0

    return data

def get_candidates(read_seq, align_data, aligned_pairs, exp_motif_seq, motif_base, motif_ind):
    
    base_id = {m.start(0): i for i, m in enumerate(re.finditer(r'(?={})'.format(motif_base), read_seq))}
    motif_anchor = np.array([m.start(0) for m in re.finditer(r'(?={})'.format(exp_motif_seq), read_seq)])
    motif_id = np.array(sorted(list(set.union(*[set(motif_anchor + i) for i in motif_ind]))))

    if align_data[0]:
        is_mapped, is_forward, _, _, _, read_length = align_data
        aligned_pairs_read_wise_original = aligned_pairs[aligned_pairs[:, 0] != -1]
        aligned_pairs_read_wise = np.copy(aligned_pairs_read_wise_original)

        if not is_forward:
            aligned_pairs_read_wise = aligned_pairs_read_wise[::-1]
            aligned_pairs_read_wise[:, 0] = read_length - aligned_pairs_read_wise[:, 0] - 1

        if len(motif_id) > 0:
            aligned_pairs_read_wise = aligned_pairs_read_wise[motif_id]
            return base_id, aligned_pairs_read_wise, aligned_pairs_read_wise_original
        else:
            return base_id, np.empty((0, 2), dtype=int), aligned_pairs_read_wise_original

    else:
        motif_id = np.vstack([motif_id, -1 * np.ones(len(motif_id))]).T.astype(int)
        return base_id, motif_id, None

def get_output(params, output_Q, modification_event, header_dict, ref_pos_dict):
    import os, datetime, queue
    header = pysam.AlignmentHeader.from_dict(header_dict)
    bam_threads = params['bam_threads']

    output = params['output']
    bam_output = os.path.join(output, f"{params['prefix']}.bam")
    per_read_file_path = os.path.join(output, f"{params['prefix']}.per_read")

    per_site_file_path = os.path.join(output, f"{params['prefix']}.per_site")
    qscore_cutoff = params['qscore_cutoff']
    length_cutoff = params['length_cutoff']

    mod_threshold = params['mod_t']
    unmod_threshold = params['unmod_t']
    skip_per_site = params['skip_per_site']

    per_site_pred = {}

    counter = 0
    counter_check = 0

    with open(per_read_file_path, 'w') as per_read_file:
        per_read_file.write('read_name\tread_position\tstrand\tmodification_score\tmean_read_qscore\tread_length\n')

        with pysam.AlignmentFile(bam_output, "wb", threads=bam_threads, header=header) as outf:
            while True:
                if modification_event.is_set() and output_Q.empty():
                    break
                try:
                    res = output_Q.get(block=False, timeout=10)
                    if counter // 10000 > counter_check:
                        counter_check = counter // 10000
                        print('%s: Number of reads processed: %d' % (str(datetime.datetime.now()), counter), flush=True)

                    if res[0]:
                        _, total_read_info, total_candidate_list, total_MM_list, read_qual_list, pred_list = res

                        for read_data, candidate_list, MM, ML, pred_vals in zip(*res[1:]):
                            counter += 1
                            read_dict, read_info = read_data
                            read = pysam.AlignedSegment.from_dict(read_dict, header)
                            if MM:
                                read.set_tag('MM', MM, value_type='Z')
                                read.set_tag('ML', ML)
                            outf.write(read)

                            read_name = read_dict['name']
                            is_forward, chrom, read_length, mean_qscore = read_info
                            strand = '+' if is_forward else '-'

                            if float(mean_qscore) < qscore_cutoff or int(read_length) < length_cutoff:
                                continue

                            for i in range(len(pred_vals)):
                                read_pos = candidate_list[i][0] + 1
                                ref_pos = candidate_list[i][1]
                                score = pred_vals[i]

                                if score < mod_threshold and score > unmod_threshold:
                                    continue

                                mod = score >= mod_threshold
                                key = (chrom, ref_pos, strand)
                                if key not in per_site_pred:
                                    per_site_pred[key] = [0, 0]
                                per_site_pred[key][mod] += 1

                                per_read_file.write('%s\t%d\t%s\t%.4f\t%.2f\t%d\n' %
                                                    (read_name, read_pos, strand,
                                                     score, mean_qscore, read_length))

                    else:
                        _, total_read_info = res
                        for read_dict in total_read_info:
                            counter += 1
                            read = pysam.AlignedSegment.from_dict(read_dict, header)
                            outf.write(read)

                except queue.Empty:
                    pass

    print('%s: Number of reads processed: %d' % (str(datetime.datetime.now()), counter), flush=True)
    print('%s: Finished Per-Read Output. Starting Per-Site output.' % str(datetime.datetime.now()), flush=True)
    print('%s: BAM file written to: %s' % (str(datetime.datetime.now()), bam_output), flush=True)
    print('%s: Per-read file written to: %s' % (str(datetime.datetime.now()), per_read_file_path), flush=True)

    if skip_per_site:
        return

    with open(per_site_file_path, 'w') as per_site_file:
        per_site_file.write('#coverage\tmod_coverage\tunmod_coverage\tmod_fraction\n')

        for (chrom, pos, strand), counts in sorted(per_site_pred.items()):
            mod_cov = counts[1]
            unmod_cov = counts[0]
            cov = mod_cov + unmod_cov
            if cov == 0:
                continue
            mod_frac = mod_cov / cov
            per_site_file.write('%d\t%d\t%d\t%.4f\n' %
                                (cov, mod_cov, unmod_cov, mod_frac))

    print('%s: Per-site file written to: %s' % (str(datetime.datetime.now()), per_site_file_path), flush=True)
    print('%s: Finished Writing Per Site Output.' % str(datetime.datetime.now()), flush=True)
    return

def process(params, signal_Q, output_Q, input_event):
    torch.set_grad_enabled(False)

    dev = params['dev']
    motif_seq = params['motif_seq']
    exp_motif_seq = params['exp_motif_seq']
    motif_base = motif_seq[params['motif_ind'][0]]
    motif_ind = params['motif_ind']
    if params['mod_symbol']:
        mod_symbol = params['mod_symbol']
    elif motif_seq == 'T':
        mod_symbol = 'U'
    else:
        mod_symbol = motif_base


    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    cigar_map = {'M': 0, '=': 0, 'X': 0, 'D': 1, 'I': 2, 'S': 2, 'H': 2, 'N': 1, 'P': 4, 'B': 4}
    cigar_pattern = r'\d+[A-Za-z=]'

    model, model_config = get_model(params)
    window = model_config['window']
    full_signal = model_config['full_signal']
    strides_per_base = model_config['strides_per_base']
    norm_type = model_config['norm_type']

    model.eval()
    model.to(dev)
    print(f"Active device: {dev}", flush=True)
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    print(f"MPS available: {torch.backends.mps.is_available()}", flush=True)
    print(f"Model device: {next(model.parameters()).device}", flush=True)

    reads_per_round = 100
    chunk_size = 256 if dev == 'cpu' else params['batch_size']

    total_candidate_list = []
    total_feature_list = []
    total_base_seq_list = []
    total_MM_list = []
    total_read_info = []
    total_c_idx = []
    total_unprocessed_reads = []

    while True:
        if signal_Q.empty() and input_event.is_set():
            break

        try:
            chunk = signal_Q.get(block=False, timeout=10)
            if output_Q.qsize() > 200:
                time.sleep(30)
                if output_Q.qsize() > 500:
                    time.sleep(60)
                print('Pausing output due to queue size limit. Output_qsize=%d   Signal_qsize=%d' %
                      (output_Q.qsize(), signal_Q.qsize()), flush=True)

            for data in chunk:
                signal, move, read_dict, align_data = data
                is_mapped, is_forward, ref_name, reference_start, reference_end, read_length = align_data

                fq = read_dict['seq']
                qual = read_dict['qual']
                fq = revcomp(fq) if not is_forward else fq
                qual = qual[::-1] if not is_forward else qual

                aligned_pairs = None
                if is_mapped:
                    cigar_tuples = np.array([(int(x[:-1]), cigar_map[x[-1]]) for x in re.findall(cigar_pattern, read_dict['cigar'])])
                    ref_start = int(read_dict['ref_pos']) - 1
                    aligned_pairs = get_aligned_pairs(cigar_tuples, ref_start)

                _, pos_list_candidates, _ = get_candidates(fq, align_data, aligned_pairs, exp_motif_seq, motif_base, motif_ind)

                sequence_length = len(fq)
                pos_list_candidates = pos_list_candidates[
                    (pos_list_candidates[:, 0] > window * strides_per_base) &
                    (pos_list_candidates[:, 0] < sequence_length - (window + 1) * strides_per_base)
                ] if len(pos_list_candidates) > 0 else pos_list_candidates

                if len(pos_list_candidates) == 0:
                    total_unprocessed_reads.append(read_dict)
                    continue

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
                        total_unprocessed_reads.append(read_dict)
                        continue

                base_seq = np.array([base_map[x] for x in fq])
                base_qual = 10 ** ((33 - np.array([ord(x) for x in qual])) / 10)
                mean_qscore = -10 * np.log10(np.mean(base_qual))
                base_qual = (1 - base_qual)[:, np.newaxis]

                if full_signal:
                    mat, indexes, segments = get_segment_events(signal, move, norm_type)
                    segments_ends = np.concatenate([segments[1:], np.array([len(move[2])])])
                    segment_ranges = np.vstack([segments, segments_ends]).T

                    base_seq_full = base_seq[indexes]
                    base_qual_full = base_qual[indexes]

                    per_site_features = np.array([
                        mat[segment_ranges[candidate[0]][0] - strides_per_base * window:
                            segment_ranges[candidate[0]][0] + strides_per_base * (window + 1)]
                        for candidate in pos_list_candidates
                    ])
                    per_site_base_qual = np.array([
                        base_qual_full[segment_ranges[candidate[0]][0] - strides_per_base * window:
                                       segment_ranges[candidate[0]][0] + strides_per_base * (window + 1)]
                        for candidate in pos_list_candidates
                    ])
                    per_site_indexes = np.array([
                        (indexes == candidate[0])[segment_ranges[candidate[0]][0] - strides_per_base * window:
                                                  segment_ranges[candidate[0]][0] + strides_per_base * (window + 1)]
                        for candidate in pos_list_candidates
                    ])
                    per_site_features = np.dstack([per_site_features, per_site_indexes[:, :, np.newaxis], per_site_base_qual])
                    per_site_base_seq = np.array([
                        base_seq_full[segment_ranges[candidate[0]][0] - strides_per_base * window:
                                      segment_ranges[candidate[0]][0] + strides_per_base * (window + 1)]
                        for candidate in pos_list_candidates
                    ])
                else:
                    mat = get_events(signal, move, norm_type)
                    mat = np.hstack((mat, base_qual))
                    per_site_features = np.array([
                        mat[candidate[0] - window: candidate[0] + window + 1]
                        for candidate in pos_list_candidates
                    ])
                    per_site_base_seq = np.array([
                        base_seq[candidate[0] - window: candidate[0] + window + 1]
                        for candidate in pos_list_candidates
                    ])

                total_candidate_list.append(pos_list_candidates)
                total_feature_list.append(per_site_features)
                total_base_seq_list.append(per_site_base_seq)
                total_read_info.append((read_dict, [align_data[1], align_data[2], align_data[5], mean_qscore]))

                try:
                    if len(pos_list_candidates) == 0:
                        total_c_idx.append([])
                        total_MM_list.append(None)
                    else:
                        pos_in_read = pos_list_candidates[:, 0].astype(int)
                        pos_in_read_sorted = np.sort(pos_in_read)

                        deltas = pos_in_read_sorted.copy()
                        deltas[1:] = pos_in_read_sorted[1:] - pos_in_read_sorted[:-1] - 1

                        MM = f"{motif_base}+{mod_symbol}?," + ",".join(map(str, deltas)) + ";"

                        c_idx = [True for _ in pos_in_read]
                        total_c_idx.append(c_idx)
                        total_MM_list.append(MM)
                except Exception as e:
                    total_c_idx.append([])
                    total_MM_list.append(None)

            if len(total_read_info) >= reads_per_round:
                read_counts = np.cumsum([len(x) for x in total_feature_list])[:-1]
                features_list = np.vstack(total_feature_list)
                base_seq_list = np.vstack(total_base_seq_list)

                pred_list = [
                    model(batch_x.to(dev), batch_base_seq.to(dev)).cpu().numpy()
                    for batch_x, batch_base_seq in generate_batches(features_list, base_seq_list, window, batch_size=chunk_size)
                ]
                pred_list = np.vstack(pred_list)
                pred_list = np.split(pred_list.ravel(), read_counts)

                read_qual_list = [
                    array.array('B', np.round(255 * read_pred_list[c_idx]).astype(int))
                    for read_pred_list, c_idx in zip(pred_list, total_c_idx)
                ]

                output_Q.put([True, total_read_info, total_candidate_list, total_MM_list, read_qual_list, pred_list])
                total_candidate_list, total_feature_list, total_base_seq_list, total_MM_list, total_read_info, total_c_idx = [], [], [], [], [], []

            if len(total_unprocessed_reads) > 100:
                output_Q.put([False, total_unprocessed_reads])
                total_unprocessed_reads = []

        except queue.Empty:
            pass

    if len(total_read_info) > 0:
        read_counts = np.cumsum([len(x) for x in total_feature_list])[:-1]
        features_list = np.vstack(total_feature_list)
        base_seq_list = np.vstack(total_base_seq_list)

        pred_list = [
            model(batch_x.to(dev), batch_base_seq.to(dev)).cpu().numpy()
            for batch_x, batch_base_seq in generate_batches(features_list, base_seq_list, window, batch_size=chunk_size)
        ]
        pred_list = np.vstack(pred_list)
        pred_list = np.split(pred_list.ravel(), read_counts)

        read_qual_list = [
            array.array('B', np.round(255 * read_pred_list[c_idx]).astype(int))
            for read_pred_list, c_idx in zip(pred_list, total_c_idx)
        ]

        output_Q.put([True, total_read_info, total_candidate_list, total_MM_list, read_qual_list, pred_list])

    if len(total_unprocessed_reads) > 0:
        output_Q.put([False, total_unprocessed_reads])

    return

def get_input(params, signal_Q, output_Q, input_event):   
    length_cutoff = params['length_cutoff']
    skip_unmapped = params['skip_unmapped']
    
    bam = params['bam']
    bam_file = pysam.AlignmentFile(bam, 'rb', check_sq=False)

    print('%s: Building BAM index.' % str(datetime.datetime.now()), flush=True)
    bam_index = pysam.IndexedReads(bam_file)
    bam_index.build()
    print('%s: Finished building BAM index.' % str(datetime.datetime.now()), flush=True)

    input_ = params['input']
    signal_files = [input_] if os.path.isfile(input_) else Path(input_).rglob("*.pod5")

    chunk = []
    non_primary_reads = []
    reads_per_chunk = 100
    max_qsize = 200

    move = (None, None, None)

    for filename in signal_files:
        with p5.Reader(filename) as reader:
            for read in reader.reads():
                if signal_Q.qsize() > max_qsize:
                    time.sleep(20)
                    print('Pausing input due to INPUT queue size limit. Signal_qsize=%d' % signal_Q.qsize(), flush=True)

                read_name = str(read.read_id)
                try:
                    read_iter = bam_index.find(read_name)
                    for bam_read in read_iter:
                        if bam_read.query_length < length_cutoff:
                            continue
                        if not bam_read.is_mapped and skip_unmapped:
                            continue
                        if bam_read.is_supplementary or bam_read.is_secondary:
                            continue

                        read_dict = bam_read.to_dict()
                        signal = read.signal

                        align_data = (
                            bam_read.is_mapped,
                            bam_read.is_forward,
                            bam_read.reference_name,
                            bam_read.reference_start,
                            bam_read.reference_end,
                            bam_read.query_length
                        )

                        data = (signal, move, read_dict, align_data)
                        chunk.append(data)

                        if len(chunk) >= reads_per_chunk:
                            signal_Q.put(chunk)
                            chunk = []

                except KeyError:
                    continue

    if len(chunk) > 0:
        signal_Q.put(chunk)

    input_event.set()
    return
    
def call_manager(params):
    print('%s: Starting Per Read Modification Detection.' % str(datetime.datetime.now()), flush=True)

    if params['dev'] != 'cpu':
        torch.multiprocessing.set_start_method('spawn')
    torch.set_num_threads(1)

    pmanager = mp.Manager()

    bam = params['bam']
    bam_file = pysam.AlignmentFile(bam, 'rb', check_sq=False)
    header_dict = bam_file.header.to_dict()

    signal_Q = pmanager.Queue()
    output_Q = pmanager.Queue()
    modification_event = pmanager.Event()
    input_event = pmanager.Event()

    handlers = []

    # Lanzar proceso que lee el .pod5 y lo empareja con el BAM
    input_process = mp.Process(target=get_input, args=(params, signal_Q, output_Q, input_event))
    input_process.start()

    # Siempre usamos get_output (no get_cpg_output)
    output_process = mp.Process(target=get_output, args=(params, output_Q, modification_event, header_dict, {}))
    output_process.start()

    # Lanzar trabajadores de predicción
    for hid in range(max(1, params['threads'] - 1)):
        p = mp.Process(target=process, args=(params, signal_Q, output_Q, input_event))
        p.start()
        handlers.append(p)

    input_process.join()
    print('%s: Reading inputs complete.' % str(datetime.datetime.now()), flush=True)

    for job in handlers:
        job.join()

    modification_event.set()
    print('%s: Model predictions complete. Wrapping up output.' % str(datetime.datetime.now()), flush=True)

    output_process.join()

    return
