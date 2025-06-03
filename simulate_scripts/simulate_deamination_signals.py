import pod5
import pysam
import numpy as np
from uuid import UUID
from tqdm import tqdm
import random
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import sys
import os
import argparse

# Argument parser
parser = argparse.ArgumentParser(description="Insert U signals into DNA and generate training data.")
parser.add_argument("--pod5_dna", required=True, help="Path to input DNA POD5 file.")
parser.add_argument("--bam_dna", required=True, help="Path to BAM file for DNA.")
parser.add_argument("--pod5_rna", required=True, help="Path to input RNA POD5 file.")
parser.add_argument("--bam_rna", required=True, help="Path to BAM file for RNA.")
parser.add_argument("--temp_pod5", default="temp.pod5", help="Intermediate POD5 file (will be deleted).")
parser.add_argument("--output_final_pod5", default="final.pod5", help="Final modified POD5 file.")
parser.add_argument("--mixed_list", default="mixed_list.txt", help="Output mixed_list.txt file.")
parser.add_argument("--temp_list", default="temp_list.txt", help="Temporary file with inserted positions (will be deleted).")
parser.add_argument("--log", default="log.txt", help="Log file name.")
args = parser.parse_args()

# Log path
log_file_path = args.log

# Redirect output to console and file
class Logger:
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # flush in real time

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(log_file_path)
sys.stderr = sys.stdout  # also redirect errors

print(f"All output will also be logged to: {log_file_path}")

# Paths
POD5_PATH =  args.pod5_dna
BAM_PATH = args.bam_dna
POD5_RNA_PATH = args.pod5_rna
BAM_RNA_PATH =  args.bam_rna

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
dna_mean = calculate_mean_signal(POD5_PATH)
rna_mean = calculate_mean_signal(POD5_RNA_PATH)
intensity_shift = dna_mean - rna_mean

print(f"Intensity shift to apply: {intensity_shift}")

def insert_U_into_C_general(pod5_dna_path, bam_dna_path, pod5_rna_path, bam_rna_path, output_pod5_path, n=1000):
    # Extract U from RNA with any pattern, resampled
    print("\n[RNA] Searching for U with any pattern...")

    reader_rna = pod5.Reader(pod5_rna_path)
    bam_rna = pysam.AlignmentFile(bam_rna_path, "rb", check_sq=False)
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

            resampled_signals.append(segment.copy())

            if len(resampled_signals) == n:
                break
        if len(resampled_signals) == n:
            break

    print(f"Resampled candidate U signals: {len(resampled_signals)}")

    if len(resampled_signals) < n:
        print("Not enough U signals found.")
        return

    # Insert into C positions of DNA
    print("\n[DNA] Inserting U signals into C positions...")

    reader_dna = pod5.Reader(pod5_dna_path)
    bam_dna = pysam.AlignmentFile(bam_dna_path, "rb", check_sq=False)

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
                inserted = f(np.linspace(0, 1, original_len))
                inserted += intensity_shift
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

            writer.add_read(new_read)

    print(f"\nTotal Cs replaced with U signals: {replaced}")
    print(f"Modified POD5 saved to: {output_pod5_path}")

    with pod5.Reader(output_pod5_path) as reader_check:
        total_reads = sum(1 for _ in reader_check.reads())
        print(f"Total reads in the modified POD5: {total_reads}")

    generate_modified_position_list(
        pod5_dna_path=pod5_dna_path,
        bam_dna_path=bam_dna_path,
        inserted_positions=inserted_positions,
        output_path= args.temp_list)

    return resampled_signals

# Save the genomic coordinates (chr, pos, strand, 1) where U signals were inserted replacing C signals.
def generate_modified_position_list(pod5_dna_path, bam_dna_path, inserted_positions, output_path):
    
    print("\nGenerating file with modified coordinates...")

    bam = pysam.AlignmentFile(bam_dna_path, "rb", check_sq=False)
    output_lines = []

    # Create dictionary for quick access
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
    print(f"Total modified positions recorded: {len(output_lines)}")

# Generate mixed_list.txt by combining:
#   - U (label 1) from temp_list.txt
#   - Unmodified T (label 0) from the BAM file of the modified POD5
def generate_mixed_list(inserted_sites_path, bam_path, output_path):
    
    print("\nGenerating balanced mixed_list.txt...")

    # Read modified positions
    mod_sites = set()
    with open(inserted_sites_path, "r") as f:
        for line in f:
            chrom, pos, strand, _ = line.strip().split("\t")
            mod_sites.add((chrom, int(pos), strand))

    # Search for unmodified T positions
    bam = pysam.AlignmentFile(bam_path, "rb", check_sq=False)
    unmod_candidates = set()

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
            if key not in mod_sites:
                unmod_candidates.add(key)

    # Balance the two classes
    mod_count = len(mod_sites)
    unmod_selected = random.sample(list(unmod_candidates), min(mod_count, len(unmod_candidates)))

    # Write mixed_list.txt
    with open(output_path, "w") as f:
        for chrom, pos, strand in sorted(mod_sites):
            f.write(f"{chrom}\t{pos}\t{strand}\t1\n")
        for chrom, pos, strand in sorted(unmod_selected):
            f.write(f"{chrom}\t{pos}\t{strand}\t0\n")

    print(f"mixed_list.txt created with {mod_count} modified and {len(unmod_selected)} unmodified entries.")

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

        colors = np.full(len(signal), "lightgray", dtype=object)

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
                colors[start_idx:end_idx] = "red"
            elif key in unmodified:
                colors[start_idx:end_idx] = "blue"

        # Plot the colored signal
        fig, ax = plt.subplots(figsize=(15, 4))
        last_color = colors[0]
        seg_start = 0
        for i in range(1, min(num_points_to_plot, len(signal))):
            if colors[i] != last_color:
                ax.plot(range(seg_start, i), signal[seg_start:i], color=last_color, linewidth=1)
                seg_start = i
                last_color = colors[i]
        ax.plot(range(seg_start, min(num_points_to_plot, len(signal))),
                signal[seg_start:min(num_points_to_plot, len(signal))], color=last_color, linewidth=1)

        ax.set_title("Raw signal with modified U (red) and original T (blue)")
        ax.set_xlabel("Index")
        ax.set_ylabel("Intensity")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        if len(modified) > 0:
            break  # Only show one read

# Insert into all reads of the modified POD5 the resampled U signals corresponding to the positions in temp_list.txt.
def insert_into_all_reads(pod5_modified_path, bam_path, inserted_sites_path, resampled_signals, output_final_path):

    print("\nModifying all reads that cover the positions in temp_list.txt...")

    # Load modified positions and associate them to their corresponding signals
    pos_to_signal = {}
    with open(inserted_sites_path) as f:
        for idx, line in enumerate(f):
            chrom, pos, strand, _ = line.strip().split("\t")
            key = (chrom, int(pos), strand)
            pos_to_signal[key] = resampled_signals[idx]

    # Load BAM and modified POD5
    bam = pysam.AlignmentFile(bam_path, "rb", check_sq=False)
    reader = pod5.Reader(pod5_modified_path)

    read_dict = {read.read_id: read for read in reader.reads()}

    # Create new POD5
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
                inserted = f_interp(np.linspace(0, 1, original_len))
                inserted += intensity_shift
                signal[start_idx:end_idx] = inserted

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
            writer.add_read(new_read)

    print(f"Fixed reads saved to: {output_final_path}")

# ======================== MAIN EXECUTION ==============================
resampled_signals = insert_U_into_C_general(
    pod5_dna_path=POD5_PATH,
    bam_dna_path=BAM_PATH,
    pod5_rna_path=POD5_RNA_PATH,
    bam_rna_path=BAM_RNA_PATH,
    output_pod5_path= args.temp_pod5,
    n=1000)

insert_into_all_reads(
    pod5_modified_path= args.temp_pod5,
    bam_path= BAM_PATH,
    inserted_sites_path= args.temp_list,
    resampled_signals=resampled_signals, 
    output_final_path= args.output_final_pod5)

generate_mixed_list(
    inserted_sites_path= args.temp_list,
    bam_path= BAM_PATH,
    output_path= args.mixed_list)

visualize_signal_with_colors(
    pod5_path= args.output_final_pod5,
    bam_path= BAM_PATH,
    mixed_list_path= args.mixed_list,
    num_points_to_plot=5000)

try:
    os.remove(args.temp_pod5)
    os.remove(args.temp_list)
    print("Temporary files removed.")
except FileNotFoundError:
    print("One or more temporary files not found. Skipping deletion.")
