import pod5
import pysam
import argparse
import numpy as np
from uuid import UUID
from tqdm import tqdm
from collections import defaultdict
import csv
import matplotlib.pyplot as plt

def parse_arguments():
    parser = argparse.ArgumentParser(description="Summarize signal statistics from DNA and RNA POD5 and BAM files.")
    parser.add_argument("--pod5_dna", required=True, help="Path to the DNA POD5 file.")
    parser.add_argument("--bam_dna", required=True, help="Path to the DNA BAM file.")
    parser.add_argument("--pod5_rna", required=True, help="Path to the RNA POD5 file.")
    parser.add_argument("--bam_rna", required=True, help="Path to the RNA BAM file.")
    parser.add_argument("--output", default="signal_summary.txt", help="Path to save the summary output.")
    return parser.parse_args()

def analyze_pod5_bam(pod5_path, bam_path):
    print(f"\nProcessing: {pod5_path} + {bam_path}")

    # Open BAM and POD5
    bam = pysam.AlignmentFile(bam_path, "rb", check_sq=False)
    reader = pod5.Reader(pod5_path)

    # Reading dictionary for quick access
    read_dict = {read.read_id: read for read in reader.reads()}

    base_signal_dict = defaultdict(list)
    total_bases = 0

    for aln in tqdm(bam.fetch(until_eof=True), desc="Analyzing aligned reads"):
        if not aln.has_tag("mv") or not aln.has_tag("ts") or aln.query_sequence is None:
            continue

        try:
            read = read_dict[UUID(aln.query_name)]
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
            if i >= len(seq):
                continue

            base = seq[i]
            start_idx = move_index[i] * stride + start
            end_idx = move_index[i + 1] * stride + start
            if end_idx > len(signal) or end_idx <= start_idx:
                continue

            segment = signal[start_idx:end_idx]
            if len(segment) > 0:
                base_signal_dict[base].append(segment)
                total_bases += 1

    return base_signal_dict, total_bases

def compute_signal_statistics(base_signal_dict, total_bases):
    print("\nComputing statistics for each base...")
    stats = {}

    for base, segments in base_signal_dict.items():
        all_values = np.concatenate(segments)
        segment_lengths = [len(s) for s in segments]

        stats[base] = {
            "count": len(segments),
            "total_bases": total_bases,
            "percentage": 100 * len(segments) / total_bases if total_bases else 0,
            "avg_segment_length": np.mean(segment_lengths),
            "mean_signal": np.mean(all_values),
            "min_signal": np.min(all_values),
            "max_signal": np.max(all_values),
            "std_signal": np.std(all_values)}

    return stats

def save_stats_to_csv(stats_dict, output_path):
    print(f"\n[INFO] Saving results to: {output_path}")
    fieldnames = ["base", "count", "total_bases", "percentage", "avg_segment_length",
                  "mean_signal", "min_signal", "max_signal", "std_signal"]

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for base, stats in stats_dict.items():
            row = {"base": base, **stats}
            writer.writerow(row)

# Plot the raw signal intensity distributions for DNA and RNA POD5 files.
def plot_signal_distribution_dna_vs_rna(pod5_dna_path, pod5_rna_path, max_reads=100):
    
    print("Loading signals from DNA and RNA POD5 files...")

    # Load DNA signals
    dna_signals = []
    with pod5.Reader(pod5_dna_path) as reader:
        for i, read in enumerate(reader.reads()):
            dna_signals.append(read.signal.astype(np.float32))
            if i + 1 >= max_reads:
                break

    # Load RNA signals
    rna_signals = []
    with pod5.Reader(pod5_rna_path) as reader:
        for i, read in enumerate(reader.reads()):
            rna_signals.append(read.signal.astype(np.float32))
            if i + 1 >= max_reads:
                break

    # Concatenate and plot histograms
    all_dna = np.concatenate(dna_signals)
    all_rna = np.concatenate(rna_signals)

    plt.figure(figsize=(10, 5))
    plt.hist(all_dna, bins=100, alpha=0.6, label="DNA", color="blue", density=True)
    plt.hist(all_rna, bins=100, alpha=0.6, label="RNA", color="red", density=True)
    plt.title("Signal Intensity Distribution (DNA vs RNA)")
    plt.xlabel("Signal intensity (pA)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    # Parse arguments
    args = parse_arguments()

    # Read base-wise signal information from DNA and RNA
    print("\nProcessing DNA...")
    base_signal_dict_dna, total_bases_dna = analyze_pod5_bam(args.pod5_dna, args.bam_dna)

    print("\nProcessing RNA...")
    base_signal_dict_rna, total_bases_rna = analyze_pod5_bam(args.pod5_rna, args.bam_rna)

    # Compute statistics
    stats_dna = compute_signal_statistics(base_signal_dict_dna, total_bases_dna)
    stats_rna = compute_signal_statistics(base_signal_dict_rna, total_bases_rna)

    # Save to CSV
    save_stats_to_csv(stats_dna, "dna_stats.csv")
    save_stats_to_csv(stats_rna, "rna_stats.csv")

    # Plot signal distributions
    plot_signal_distribution_dna_vs_rna(args.pod5_dna, args.pod5_rna)

    print("\nDone.")

if __name__ == "__main__":
    main()
