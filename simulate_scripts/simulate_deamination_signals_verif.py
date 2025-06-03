import pod5
import pysam
import numpy as np
from uuid import UUID
from tqdm import tqdm
import random
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import sys
import argparse
import os

# Argument Parser
parser = argparse.ArgumentParser(description="Insert U signals from RNA into C positions in DNA POD5 data.")
parser.add_argument("--dna_pod5", required=True, help="Path to DNA POD5 file.")
parser.add_argument("--rna_pod5", required=True, help="Path to RNA POD5 file.")
parser.add_argument("--dna_bam", required=True, help="Path to BAM file for DNA.")
parser.add_argument("--rna_bam", required=True, help="Path to BAM file for RNA.")
parser.add_argument("--output_dir", required=True, help="Directory where outputs will be saved.")
parser.add_argument("--n", type=int, default=1000, help="Number of U signals to extract and insert.")
parser.add_argument("--log", default="log.txt", help="Log file name.")

args = parser.parse_args()

# Path validation
for path in [args.dna_pod5, args.rna_pod5, args.dna_bam, args.rna_bam]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

os.makedirs(args.output_dir, exist_ok=True)

# Define derived paths
output_pod5_path = os.path.join(args.output_dir, "dna_with_U_inserted.pod5")
final_pod5_path = os.path.join(args.output_dir, "pod5_fully_corrected.pod5")
inserted_sites_path = os.path.join(args.output_dir, "inserted_U_sites.txt")
mixed_list_path = os.path.join(args.output_dir, "mixed_list.txt")

# Log file path
log_file_path = "log.txt"

# Redirect output to console and log file
class Logger:
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # write in real time

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(log_file_path)
sys.stderr = sys.stdout  # redirect errors too

print(f"Everything will also be logged to: {log_file_path}")

# File paths
DNA_POD5_PATH = args.dna_pod5
DNA_BAM_PATH = args.dna_bam
RNA_POD5_PATH = args.rna_pod5
RNA_BAM_PATH = args.rna_bam
log_file_path = args.log

def calculate_mean_signal(pod5_path):
    total_sum = 0
    total_count = 0
    with pod5.Reader(pod5_path) as reader:
        for read in reader.reads():
            signal = read.signal.astype(np.float32)
            total_sum += signal.sum()
            total_count += len(signal)
    return total_sum / total_count if total_count > 0 else 0

# Calculate mean intensity shift
dna_mean = calculate_mean_signal(DNA_POD5_PATH)
rna_mean = calculate_mean_signal(RNA_POD5_PATH)
intensity_shift = dna_mean - rna_mean

print(f"Intensity shift to apply: {intensity_shift}")

def insert_U_into_C_general(dna_pod5_path, dna_bam_path, rna_pod5_path, rna_bam_path, output_pod5_path, n=1000):
    # Extract resampled U signals from RNA with any move pattern
    print("\n[RNA] Searching for U bases with any move pattern...")

    reader_rna = pod5.Reader(rna_pod5_path)
    bam_rna = pysam.AlignmentFile(rna_bam_path, "rb", check_sq=False)
    resampled_signals = []
    inserted_positions = []

    for aln in tqdm(bam_rna.fetch(until_eof=True)):
        if not aln.has_tag("mv") or not aln.has_tag("ts") or aln.query_sequence is None:
            continue
        try:
            read = next(reader_rna.reads([UUID(aln.query_name)]))
        except:
            continue

        seq = aln.query_sequence
        tags = {x.split(":")[0]: x for x in aln.to_dict()["tags"]}
        start = int(tags["ts"].split(":")[-1])
        mv = tags["mv"].split(",")
        stride = int(mv[1])
        move_table = np.array(list(map(int, mv[2:])), dtype=np.int8)
        move_index = np.where(move_table != 0)[0]
        signal = read.signal.astype(np.float32)

        for i in range(len(move_index) - 1):
            if seq[i] != "T":
                continue

            start_idx = move_index[i] * stride + start
            end_idx = move_index[i + 1] * stride + start
            if end_idx > len(signal) or end_idx <= start_idx:
                continue

            segment = signal[start_idx:end_idx]
            if len(segment) < 2:
                continue

            resampled_signals.append(segment)

            if len(resampled_signals) == n:
                break
        if len(resampled_signals) == n:
            break

    print(f"Resampled candidate U signals: {len(resampled_signals)}")

    if len(resampled_signals) < n:
        print("Not enough U signals found.")
        return

    # Insert into C positions in DNA
    print("\n[DNA] Inserting U signals into C positions...")

    reader_dna = pod5.Reader(dna_pod5_path)
    bam_dna = pysam.AlignmentFile(dna_bam_path, "rb", check_sq=False)

    reads_dict = {read.read_id: read for read in reader_dna.reads()}
    replaced = 0

    with pod5.Writer(output_pod5_path) as writer:
        for aln in tqdm(bam_dna.fetch(until_eof=True)):
            if not aln.has_tag("mv") or not aln.has_tag("ts") or aln.query_sequence is None:
                continue
            try:
                read_id = UUID(aln.query_name)
                read = reads_dict[read_id]
            except:
                continue

            signal = read.signal.astype(np.float32).copy()
            seq = aln.query_sequence
            tags = {x.split(":")[0]: x for x in aln.to_dict()["tags"]}
            start = int(tags["ts"].split(":")[-1])
            mv = tags["mv"].split(",")
            stride = int(mv[1])
            move_table = np.array(list(map(int, mv[2:])), dtype=np.int8)
            move_index = np.where(move_table != 0)[0]

            for i in range(len(move_index) - 1):
                if replaced >= n:
                    break
                if seq[i] != "C":
                    continue

                start_idx = move_index[i] * stride + start
                end_idx = move_index[i + 1] * stride + start
                if end_idx > len(signal) or end_idx <= start_idx:
                    continue

                original_len = end_idx - start_idx
                resampled = resampled_signals[replaced]
                f = interp1d(np.linspace(0, 1, len(resampled)), resampled, kind='linear')
                inserted = f(np.linspace(0, 1, original_len)) + intensity_shift
                inserted_int = np.round(inserted).astype(np.int16)
                signal[start_idx:end_idx] = inserted_int
                inserted_positions.append((read_id, i))
                replaced += 1

            # Save the modified read
            new_read = pod5.Read(
                read_id=read.read_id,
                signal=signal.astype(np.int16),
                read_number=read.read_number,
                start_sample=read.start_sample,
                pore=read.pore,
                calibration=read.calibration,
                run_info=read.run_info,
                median_before=read.median_before,
                end_reason=read.end_reason)

            print(f"[CHECK] Signal length -> {read.read_id} : original = {len(read.signal)} | modified = {len(signal)}")

            writer.add_read(new_read)

    print(f"\nTotal Cs bases replaced with U signals: {replaced}")
    print(f"Modified POD5 file saved to: {output_pod5_path}")

    # Check number of reads in the modified POD5
    with pod5.Reader(output_pod5_path) as reader_check:
        total_reads = sum(1 for _ in reader_check.reads())
        print(f"[CHECK] Total reads in modified POD5: {total_reads}")

    generate_modified_position_list(
        dna_pod5_path=dna_pod5_path,
        dna_bam_path=dna_bam_path,
        inserted_positions=inserted_positions,
        output_path= inserted_sites_path)

    return resampled_signals

# Save the genomic coordinates (chr, pos, strand, 1) where U signals were inserted in place of C signals.
def generate_modified_position_list(dna_pod5_path, dna_bam_path, inserted_positions, output_path):
    
    print("\nGenerating file with modified coordinates...")

    bam = pysam.AlignmentFile(dna_bam_path, "rb", check_sq=False)
    output_lines = []

    # Create a dictionary for quick access
    aln_dict = {aln.query_name: aln for aln in bam.fetch(until_eof=True)}

    for read_id, base_idx in inserted_positions:
        aln = aln_dict.get(str(read_id))
        if aln is None:
            continue
        ref_positions = aln.get_reference_positions()
        if base_idx >= len(ref_positions):
            continue

        chrom = aln.reference_name
        pos = ref_positions[base_idx]
        strand = "-" if aln.is_reverse else "+"

        output_lines.append(f"{chrom}\t{pos}\t{strand}\t1\n")

    with open(output_path, "w") as f:
        f.writelines(output_lines)

    print(f"File saved: {output_path}")
    print(f"Total modified positions saved: {len(output_lines)}")
    print("[CHECK] First 5 modified positions:")
    for line in output_lines[:5]:
        print(line.strip())


# Generate mixed_list.txt combining:
# - U (label 1) from inserted_U_sites.txt
# - Original unmodified T (label 0) from the BAM of the modified POD5
def generate_mixed_list(inserted_sites_path, bam_path, output_path):
    
    print("\nGenerating balanced mixed_list.txt file...")

    # Load modified positions
    modified_sites = set()
    with open(inserted_sites_path, "r") as f:
        for line in f:
            chrom, pos, strand, _ = line.strip().split("\t")
            modified_sites.add((chrom, int(pos), strand))

    # Search for unmodified T positions
    bam = pysam.AlignmentFile(bam_path, "rb", check_sq=False)
    unmodified_candidates = set()

    for aln in tqdm(bam.fetch(until_eof=True), desc="Scanning BAM for unmodified T"):
        if aln.query_sequence is None:
            continue

        chrom = aln.reference_name
        strand = "-" if aln.is_reverse else "+"
        ref_positions = aln.get_reference_positions()
        seq = aln.query_sequence

        for i, base in enumerate(seq):
            if base != "T":
                continue
            if i >= len(ref_positions):
                continue
            pos = ref_positions[i]
            key = (chrom, pos, strand)
            if key not in modified_sites:
                unmodified_candidates.add(key)

    # Balance both classes
    modified_count = len(modified_sites)
    unmodified_selected = random.sample(
        list(unmodified_candidates), 
        min(modified_count, len(unmodified_candidates)))

    # Write the mixed_list.txt file
    with open(output_path, "w") as f:
        for chrom, pos, strand in sorted(modified_sites):
            f.write(f"{chrom}\t{pos}\t{strand}\t1\n")
        for chrom, pos, strand in sorted(unmodified_selected):
            f.write(f"{chrom}\t{pos}\t{strand}\t0\n")

    print(f"mixed_list.txt created with {modified_count} modified and {len(unmodified_selected)} unmodified positions.")

# Visualize the raw signal with color coding:
# - Red for U positions (label 1)
# - Blue for unmodified T positions (label 0)
def visualize_signal_with_colors(pod5_path, bam_path, mixed_list_path, num_points_to_plot=5000):
    # Load mixed_list.txt
    modified = set()
    unmodified = set()
    with open(mixed_list_path) as f:
        for line in f:
            chrom, pos, strand, label = line.strip().split("\t")
            key = (chrom, int(pos), strand)
            if label == "1":
                modified.add(key)
            else:
                unmodified.add(key)

    # Open files
    reader = pod5.Reader(pod5_path)
    bam = pysam.AlignmentFile(bam_path, "rb", check_sq=False)

    for aln in bam.fetch(until_eof=True):
        if not aln.has_tag("mv") or not aln.has_tag("ts") or aln.query_sequence is None:
            continue
        try:
            read_id = UUID(aln.query_name)
            read = next(reader.reads([read_id]))
        except:
            continue

        chrom = aln.reference_name
        strand = "-" if aln.is_reverse else "+"
        ref_positions = aln.get_reference_positions()
        seq = aln.query_sequence
        tags = {x.split(":")[0]: x for x in aln.to_dict()["tags"]}
        start = int(tags["ts"].split(":")[-1])
        mv = tags["mv"].split(",")
        stride = int(mv[1])
        move_table = np.array(list(map(int, mv[2:])), dtype=np.int8)
        move_index = np.where(move_table != 0)[0]
        signal = read.signal.astype(np.float32)

        color_map = np.full(len(signal), "lightgray", dtype=object)

        for i in range(len(move_index) - 1):
            if i >= len(seq) or i >= len(ref_positions):
                continue
            base = seq[i]
            pos = ref_positions[i]
            key = (chrom, pos, strand)

            start_idx = move_index[i] * stride + start
            end_idx = move_index[i + 1] * stride + start
            if end_idx > len(signal):
                continue

            if key in modified:
                color_map[start_idx:end_idx] = "red"
            elif key in unmodified:
                color_map[start_idx:end_idx] = "blue"

        # Plot the colored signal
        fig, ax = plt.subplots(figsize=(15, 4))
        last_color = color_map[0]
        seg_start = 0
        for i in range(1, min(num_points_to_plot, len(signal))):
            if color_map[i] != last_color:
                ax.plot(range(seg_start, i), signal[seg_start:i], color=last_color, linewidth=1)
                seg_start = i
                last_color = color_map[i]
        ax.plot(range(seg_start, min(num_points_to_plot, len(signal))),
                signal[seg_start:min(num_points_to_plot, len(signal))], color=last_color, linewidth=1)

        ax.set_title("Raw signal with modified U (red) and original T (blue)")
        ax.set_xlabel("Index")
        ax.set_ylabel("Intensity")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        if len(modified) > 0:
            break  # Visualize only one read

# Compares the inserted signals in the modified POD5 with the resampled U signals from RNA. Verifies they match exactly.
def verify_inserted_T_signals(pod5_path, bam_path, inserted_sites_path, resampled_signals):

    print("\n[CHECK] Verifying match between inserted signals and resampled U signals...")

    # Load inserted sites list
    with open(inserted_sites_path) as f:
        inserted_coords = [tuple(line.strip().split("\t")[:3]) for line in f]

    # Fast lookup set
    inserted_set = set(inserted_coords)

    # Open BAM and POD5
    reader = pod5.Reader(pod5_path)
    bam = pysam.AlignmentFile(bam_path, "rb", check_sq=False)

    idx_check = 0
    errors = 0

    for aln in tqdm(bam.fetch(until_eof=True), desc="Verifying signals in BAM"):
        if not aln.has_tag("mv") or not aln.has_tag("ts") or aln.query_sequence is None:
            continue

        read_id = aln.query_name
        chrom = aln.reference_name
        strand = "-" if aln.is_reverse else "+"
        ref_positions = aln.get_reference_positions()
        seq = aln.query_sequence

        try:
            read = next(reader.reads([UUID(read_id)]))
        except:
            continue

        tags = {x.split(":")[0]: x for x in aln.to_dict()["tags"]}
        start = int(tags["ts"].split(":")[-1])
        mv = tags["mv"].split(",")
        stride = int(mv[1])
        move_table = np.array(list(map(int, mv[2:])), dtype=np.int8)
        move_index = np.where(move_table != 0)[0]
        signal = read.signal.astype(np.float32)

        for i in range(len(move_index) - 1):
            if idx_check >= len(resampled_signals):
                break

            if i >= len(seq) or i >= len(ref_positions):
                continue
            base = seq[i]
            pos = ref_positions[i]
            key = (chrom, str(pos), strand)

            if key not in inserted_set:
                continue

            # Extract inserted raw signal
            start_idx = move_index[i] * stride + start
            end_idx = move_index[i + 1] * stride + start
            if end_idx > len(signal) or end_idx <= start_idx:
                continue

            segment = signal[start_idx:end_idx]
            resampled = resampled_signals[idx_check]

            # Resample expected signal to match segment length
            f = interp1d(np.linspace(0, 1, len(resampled)), resampled, kind='linear')
            expected_resampled = f(np.linspace(0, 1, len(segment))) + intensity_shift

            # Compare
            if not np.allclose(segment, expected_resampled, atol=1):
                print(f"[ERROR] Mismatch at {key} → inserted: {segment} (len={len(segment)}) | expected: {resampled} (len={len(resampled)})")
                errors += 1
            else:
                print(f"[CHECK] Signal #{idx_check+1} matches at {key}")

            idx_check += 1

        if idx_check >= len(resampled_signals):
            break

    print(f"\n[CHECK] Verified signals: {idx_check}, Errors: {errors}")

# Inserts the resampled U signals into all reads that cover the positions listed in inserted_U_sites.txt.
def insert_into_all_reads(pod5_modified_path, bam_path, inserted_sites_path, resampled_signals, output_final_path):

    print("\n[CHECK] Modifying all reads covering positions in inserted_U_sites.txt...")

    # Load modified positions and associate each with its corresponding signal
    pos_to_signal = {}
    with open(inserted_sites_path) as f:
        for idx, line in enumerate(f):
            chrom, pos, strand, _ = line.strip().split("\t")
            key = (chrom, int(pos), strand)
            pos_to_signal[key] = resampled_signals[idx]

    # Open BAM and modified POD5
    bam = pysam.AlignmentFile(bam_path, "rb", check_sq=False)
    reader = pod5.Reader(pod5_modified_path)

    read_dict = {read.read_id: read for read in reader.reads()}

    # Create new POD5 with all reads properly corrected
    with pod5.Writer(output_final_path) as writer:
        for aln in tqdm(bam.fetch(until_eof=True), desc="Fixing reads"):
            if not aln.has_tag("mv") or not aln.has_tag("ts") or aln.query_sequence is None:
                continue

            try:
                read_id = UUID(aln.query_name)
                read = read_dict[read_id]
            except:
                continue

            signal = read.signal.astype(np.float32).copy()
            seq = aln.query_sequence
            tags = {x.split(":")[0]: x for x in aln.to_dict()["tags"]}
            start = int(tags["ts"].split(":")[-1])
            mv = tags["mv"].split(",")
            stride = int(mv[1])
            move_table = np.array(list(map(int, mv[2:])), dtype=np.int8)
            move_index = np.where(move_table != 0)[0]
            ref_positions = aln.get_reference_positions()
            chrom = aln.reference_name
            strand = "-" if aln.is_reverse else "+"

            for i in range(len(move_index) - 1):
                if i >= len(seq) or i >= len(ref_positions):
                    continue

                pos = ref_positions[i]
                key = (chrom, pos, strand)
                if key not in pos_to_signal:
                    continue

                start_idx = move_index[i] * stride + start
                end_idx = move_index[i + 1] * stride + start
                if end_idx > len(signal) or end_idx <= start_idx:
                    continue

                original_len = end_idx - start_idx
                resampled = pos_to_signal[key]
                f_interp = interp1d(np.linspace(0, 1, len(resampled)), resampled, kind='linear')
                inserted = f_interp(np.linspace(0, 1, original_len)) + intensity_shift
                signal[start_idx:end_idx] = inserted

            # Save modified read
            new_read = pod5.Read(
                read_id=read.read_id,
                signal=signal.astype(np.int16),
                read_number=read.read_number,
                start_sample=read.start_sample,
                pore=read.pore,
                calibration=read.calibration,
                run_info=read.run_info,
                median_before=read.median_before,
                end_reason=read.end_reason)
            writer.add_read(new_read)

    print(f"[CHECK] Fixed reads saved in: {output_final_path}")

# Verifies that the number of reads and the signal length of each read is identical
# between the original and the modified POD5 file.
def verify_signal_length_equality(original_pod5_path, modified_pod5_path):

    print("\n[CHECK] Verifying that the modified POD5 has the same signal lengths as the original...")

    original_lengths = {}
    modified_lengths = {}

    # Read lengths from the original file
    with pod5.Reader(original_pod5_path) as reader_orig:
        for read in reader_orig.reads():
            original_lengths[read.read_id] = len(read.signal)

    # Read lengths from the modified file
    with pod5.Reader(modified_pod5_path) as reader_mod:
        for read in reader_mod.reads():
            modified_lengths[read.read_id] = len(read.signal)

    # Compare number of reads
    if len(original_lengths) != len(modified_lengths):
        print(f"[ERROR] Different number of reads: original={len(original_lengths)}, modified={len(modified_lengths)}")
        return

    errors = 0
    for read_id in original_lengths:
        if read_id not in modified_lengths:
            print(f"[ERROR] Read {read_id} is missing in the modified file.")
            errors += 1
            continue

        len_orig = original_lengths[read_id]
        len_mod  = modified_lengths[read_id]
        if len_orig != len_mod:
            print(f"[ERROR] Signal length mismatch in {read_id}: original={len_orig}, modified={len_mod}")
            errors += 1

    if errors == 0:
        print("[CHECK] All reads have the same signal length.")
    else:
        print(f"[ERROR] Found {errors} reads with different lengths.")

# Verifies that in all reads covering a modified position, the raw signal has actually been modified.
# Compares against the inserted resampled signals.
def verify_modifications_in_all_reads(pod5_path, bam_path, inserted_sites_path, resampled_signals):

    print("\n[CHECK] Verifying that all reads have been modified at the desired positions...")

    # Load expected signals per position
    pos_to_signal = {}
    with open(inserted_sites_path) as f:
        for idx, line in enumerate(f):
            chrom, pos, strand, _ = line.strip().split("\t")
            pos_to_signal[(chrom, int(pos), strand)] = resampled_signals[idx]

    reader = pod5.Reader(pod5_path)
    bam = pysam.AlignmentFile(bam_path, "rb", check_sq=False)
    read_dict = {read.read_id: read for read in reader.reads()}

    errors = 0
    total = 0

    for aln in tqdm(bam.fetch(until_eof=True), desc="Verifying reads covering modified positions"):
        if not aln.has_tag("mv") or not aln.has_tag("ts") or aln.query_sequence is None:
            continue

        chrom = aln.reference_name
        strand = "-" if aln.is_reverse else "+"
        ref_positions = aln.get_reference_positions()
        seq = aln.query_sequence

        try:
            read = read_dict[UUID(aln.query_name)]
        except:
            continue

        tags = {x.split(":")[0]: x for x in aln.to_dict()["tags"]}
        start = int(tags["ts"].split(":")[-1])
        mv = tags["mv"].split(",")
        stride = int(mv[1])
        move_table = np.array(list(map(int, mv[2:])), dtype=np.int8)
        move_index = np.where(move_table != 0)[0]
        signal = read.signal.astype(np.float32)

        for i in range(len(move_index) - 1):
            if i >= len(seq) or i >= len(ref_positions):
                continue

            pos = ref_positions[i]
            key = (chrom, pos, strand)

            if key not in pos_to_signal:
                continue

            start_idx = move_index[i] * stride + start
            end_idx = move_index[i + 1] * stride + start
            if end_idx > len(signal) or end_idx <= start_idx:
                continue

            segment = signal[start_idx:end_idx]
            expected = pos_to_signal[key]

            f = interp1d(np.linspace(0, 1, len(expected)), expected, kind='linear')
            expected_resampled = f(np.linspace(0, 1, len(segment))) + intensity_shift

            if not np.allclose(segment, expected_resampled, atol=1):
                print(f"[ERROR] In read {read.read_id}, the signal at {key} does not match.")
                errors += 1
            total += 1

    print(f"\n[CHECK] Verified {total} positions in reads. Errors: {errors}")
    if errors == 0:
        print("[CHECK] All reads were correctly modified.")

# =========================== MAIN EXECUTION ===========================

# Extract U signals from RNA and insert them into C positions in DNA
resampled_signals = insert_U_into_C_general(
    dna_pod5_path=DNA_POD5_PATH,
    dna_bam_path=DNA_BAM_PATH,
    rna_pod5_path=RNA_POD5_PATH,
    rna_bam_path=RNA_BAM_PATH,
    output_pod5_path=output_pod5_path,
    n=args.n)


# Insert the same signals into all reads covering those positions
insert_into_all_reads(
    pod5_modified_path=output_pod5_path,
    bam_path=DNA_BAM_PATH,
    inserted_sites_path=inserted_sites_path,
    resampled_signals=resampled_signals,
    output_final_path=final_pod5_path)

# Verify that all reads were correctly modified at those positions
verify_modifications_in_all_reads(
    pod5_path=final_pod5_path,
    bam_path=DNA_BAM_PATH,
    inserted_sites_path=inserted_sites_path,
    resampled_signals=resampled_signals)

# Check that the inserted signals match the original resampled signals
verify_inserted_T_signals(
    pod5_path=output_pod5_path,
    bam_path=DNA_BAM_PATH,
    inserted_sites_path=inserted_sites_path,
    resampled_signals=resampled_signals)

# Generate mixed_list.txt with 50% modified (label 1) and 50% unmodified (label 0) sites
generate_mixed_list(
    inserted_sites_path=inserted_sites_path,
    bam_path=DNA_BAM_PATH,
    output_path=mixed_list_path)

# Visualize the raw signal with inserted U (red) vs original T (blue)
visualize_signal_with_colors(
    pod5_path=output_pod5_path,
    bam_path=DNA_BAM_PATH,
    mixed_list_path=mixed_list_path,
    num_points_to_plot=5000)

# Ensure that the final corrected file has same number of reads and signal lengths
verify_signal_length_equality(
    original_pod5_path=DNA_POD5_PATH,
    modified_pod5_path=final_pod5_path)
