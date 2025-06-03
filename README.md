# CUNA
CUNA (Cytosine Uracil Neural Algorithm) is a deep learning-based pipeline for detecting cytosine deamination events (C→U) in ancient DNA sequenced with Oxford Nanopore Technologies. It extends the [DeepMod2](https://github.com/WGLab/DeepMod2) framework by introducing preprocessing steps for signal simulation and modification, and supports training and inference using BiLSTM and Transformer models.

This pipeline is specifically designed for studying ancient DNA samples, where cytosines often spontaneously deaminate to uracils due to age-related chemical damage.

<p align="center">
  <img src="images/framework.png" alt="CUNA Framework" width="600"/>
</p>

## Project Structure
```
CUNA/
├── simulate_scripts/
│ ├── dna.pod5
│ ├── rna.pod5
│ ├── bam_files/
│     └── rna.bam
│     └── dna.bam
│     └── dna_sorted.bam
│     └── dna_sorted.bam.bai
│ ├── simulate_deamination_signals.py
│ ├── simulate_deamination_signals_verif.py
│ ├── statistics.py
│ ├── output/
│     └── mixed_list
│     └── deamination.pod5
│     └── log.txt
│     └── dna_stats.csv
│     └── rna_stats.csv
│     └── deamination_pod5_figure
│     └── signal_distribution
├── train_models/
│ ├── generate_features.py
│ ├── train_models.py
│ ├── utils.py
│ ├── reference_genome/
|     └── GRCh38.fa
│     └── GRCh38.fa.fai
│ ├── features_output/
│     └── args
│     └── output.features.X.npz
│ ├── train_output/
│     └── bilstm/
│     └── transformer/
│     └── args
│     └── model.log
├── test/
│ ├── test_data/
│     └── test.pod5
│     └── test.bam
│ ├── CUNA.py
│ ├── utils.py
│ ├── detect.py
│ ├── models.py
│ ├── test_output/
│     └── output.bam
│     └── output.per_read
│     └── output.per_site
│     └── args
```

---

## Environment Setup

We recommend using [Micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) for environment isolation.

```bash
micromamba create -n CUNA -f requirements.yml
micromamba activate CUNA
```

```bash
# Install CUNA
git clone https://github.com/iris1901/CUNA.git ${INPUT_DIR}/CUNA
```

##   Download Software Packges

In this project, we use [Dorado](https://github.com/nanoporetech/dorado.git), the official basecaller from Oxford Nanopore Technologies, to perform signal-to-sequence conversion for both DNA and RNA datasets.
We will download the Dorado basecaller (v0.5.3) along with the appropriate pre-trained basecalling models for each data type and flow cells.
Installation instructions for both Linux and macOS (Apple Silicon) are provided below.

```bash
# For Linux:
wget -qO- https://cdn.oxfordnanoportal.com/software/analysis/dorado-0.5.3-linux-x64.tar.gz | tar xzf - -C ${INPUT_DIR}
${INPUT_DIR}/dorado-0.5.3-linux-x64/bin/dorado download --model  dna_r10.4.1_e8.2_400bps_hac@v4.3.0 --directory ${INPUT_DIR}/dorado-0.5.3-linux-x64/models/
${INPUT_DIR}/dorado-0.5.3-linux-x64/bin/dorado download --model  rna004_130bps_hac@v5.2.0 --directory ${INPUT_DIR}/dorado-0.5.3-linux-x64/models/

# For macOS (Apple Silicon):
wget -qO- https://cdn.oxfordnanoportal.com/software/analysis/dorado-0.5.3-osx-arm64.tar.gz | tar xzf - -C ${INPUT_DIR}
${INPUT_DIR}/dorado-0.5.3-osx-arm64/bin/dorado download --model dna_r10.4.1_e8.2_400bps_hac@v4.3.0 --directory ${INPUT_DIR}/dorado-0.5.3-osx-arm64/models/
${INPUT_DIR}/dorado-0.5.3-osx-arm64/bin/dorado download --model rna004_130bps_hac@v5.2.0 --directory ${INPUT_DIR}/dorado-0.5.3-osx-arm64/models/
```

## Step 0: Basecalling and Alignment (DNA + RNA)

To begin, we perform basecalling on two raw signal datasets:

- A DNA POD5 file generated from [DeepMod2](https://github.com/WGLab/DeepMod2/files/14368872/sample.pod5.tar.gz) of modern DNA using an R10.4.1 flow cell.
- An RNA BLOW5 file publicly available from [nanoCEM](https://github.com/lrslab/nanoCEM/raw/3f7ab5f001448e4f15ef5d17dad04ca6507394bb/example/data/wt/file.blow5), which we converted to POD5 for compatibility.

The data can be downloaded from the official repositories or go to the simulate_scripts folder where they are already downloaded.
We then use Dorado for basecalling, with --emit-moves to create move tables in addition to basecalls.

If you want to download through the terminal: 

```bash
# DNA POD5
wget -qO- https://github.com/WGLab/DeepMod2/files/14368872/sample.pod5.tar.gz| tar xzf - -C ${INPUT_DIR}/CUNA/simulate_scripts

# RNA BLOW5
wget https://github.com/lrslab/nanoCEM/raw/3f7ab5f001448e4f15ef5d17dad04ca6507394bb/example/data/wt/file.blow5 \
  -O ${INPUT_DIR}/CUNA/simulate_scripts/rna.blow5

# Convert BLOW5 to POD5
pip install blue-crab
blue-crab s2p rna.blow5 -o rna.pod5

# Genome Reference (for DNA only)
# Before DNA basecalling, we must download a reference genome for anchored alignment:
wget ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.39_GRCh38.p13/GRCh38_major_release_seqs_for_alignment_pipelines/GCA_000001405.15_GRCh38_no_alt_plus_hs38d1_analysis_set.fna.gz -O -| gunzip -c > ${INPUT_DIR}/CUNA/train_models/reference_genome/GRCh38.fa
wget ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.39_GRCh38.p13/GRCh38_major_release_seqs_for_alignment_pipelines/GCA_000001405.15_GRCh38_no_alt_plus_hs38d1_analysis_set.fna.fai -O ${INPUT_DIR}/CUNA/train_models/reference_genome/GRCh38.fa.fai
```

RNA reads are basecalled without alignment, using the Dorado RNA model:
```bash
${INPUT_DIR}/dorado-0.5.3-linux-x64/bin/dorado basecaller \
  --model rna004_130bps_hac@v5.2.0 \
  --emit-moves \
  --recursive \
  ${INPUT_DIR}/CUNA/simulate_scripts/rna.pod5 > ${INPUT_DIR}/CUNA/simulate_scripts/bam_files/rna.bam
```
For DNA, we perform reference-anchored basecalling and alignment:
```bash
${INPUT_DIR}/dorado-0.5.3-linux-x64/bin/dorado basecaller \
  --model dna_r10.4.1_e8.2_400bps_hac@v4.3.0 \
  --emit-moves \
  --recursive \
  --reference ${INPUT_DIR}/CUNA/train_models/reference_genome/GRCh38.fa \
  ${INPUT_DIR}/CUNA/simulate_scripts/dna.pod5 > ${INPUT_DIR}/CUNA/simulate_scripts/bam_files/dna.bam
```
The resulting BAM files provide a precise alignment between the raw signal and the basecalled sequence. This correlation enables the accurate localization of each base’s signal segment using the move table.

## Step 1: Simulate Ancient DNA from POD5

(Optional) Before simulating uracil insertions, you can explore the characteristics of the raw signals in the DNA and RNA POD5 files. This helps validate signal consistency and understand nucleotide-specific patterns prior to resampling or insertion. We use a helper script (statistics.py) that extracts base-level statistics from both DNA and RNA sources, and generates a CSV summary.

```bash
python ${INPUT_DIR}/CUNA/simulate_scripts/statistics.py \
  --pod5_dna ${INPUT_DIR}/CUNA/simulate_scripts/dna.pod5 \
  --bam_dna ${INPUT_DIR}/CUNA/simulate_scripts/bam_files/dna.bam \
  --pod5_rna ${INPUT_DIR}/CUNA/simulate_scripts/rna.pod5 \
  --bam_rna ${INPUT_DIR}/CUNA/simulate_scripts/bam_files/rna.bam \
  --output ${INPUT_DIR}/CUNA/simulate_scripts/output/
```

Since no real ancient DNA POD5 is available, we simulate uracil-induced damage by extracting U signals from RNA POD5, inserting those signals in place of cytosines in DNA POD5 at the beginning and end of DNA fragments more frequently, respecting empirical estimates of deamination rates (~3–9% C→U), maintaining total signal length to preserve BAM compatibility and generating a corresponding `mixed_list.txt` with modified and unmodified positions.

```bash
python ${INPUT_DIR}/CUNA/simulate_scripts/simulate_deamination_signals.py \
  --pod5_dna ${INPUT_DIR}/CUNA/simulate_scripts/DNA_can.pod5 \
  --bam_dna ${INPUT_DIR}/CUNA/simulate_scripts/bam_files/DNA.bam \
  --pod5_rna ${INPUT_DIR}/CUNA/simulate_scripts/RNA.pod5 \
  --bam_rna ${INPUT_DIR}/CUNA/simulate_scripts/bam_files/RNA.bam \
```
(Optional) To ensure that uracil signals were inserted correctly and signal lengths were preserved, you can use an alternative version of the simulation script that includes automatic verification steps

```bash
python ${INPUT_DIR}/CUNA/simulate_scripts/simulate_deamination_signals_verif.py \
  --dna_pod5 ${INPUT_DIR}/CUNA/simulate_scripts/dna.pod5 \
  --rna_pod5 ${INPUT_DIR}/CUNA/simulate_scripts/rna.pod5 \
  --dna_bam ${INPUT_DIR}/CUNA/simulate_scripts/bam_files/dna.bam \
  --rna_bam ${INPUT_DIR}/CUNA/simulate_scripts/bam_files/rna.bam \
  --output_dir ${INPUT_DIR}/CUNA/simulate_scripts/output/ \
  --log  ${INPUT_DIR}/CUNA/simulate_scripts/output/log.txt
```

We will generate:

- A new POD5 file that simulates cytosine deamination events (C→U) typically found in ancient DNA, by replacing the signal at selected cytosine sites with uracil signals extracted from RNA.
- A `mixed_list.txt` file used in training contains one genomic position per line, with the following format:
    - `contig`: Chromosome or scaffold name from the reference genome
    - `position`: 0-based genomic coordinate of the base of interest
    - `strand`: Either `+` or `-`
    - `label`: 1 for uracil (deaminated cytosine), 0 for canonical thymine

## Step 2: Generate Training Features
Once we have simulated cytosine deaminations (C→U) and produced the corresponding `mixed_list.txt`, we extract features from the modified signals using the script `generate_features.py`. We will generate features for the sample by providing signal POD5 file as --input, BAM file as --bam and a list of positions with modified/unmodified labels as --pos_list and use --threads NUM_THREADS to speed up feature generation. In this case, we will use a window size of 10, which means how many bases before and after each base position of interest (from pos_list) to include in feature generation. These features will later be used to train a deep learning model capable of distinguishing deaminated C→U sites from natural T bases.

This structure allows the pipeline to train a binary classifier that distinguishes between truly unmodified thymines and uracils resulting from cytosine damage.

```bash
python ${INPUT_DIR}/CUNA/train_models/generate_features.py \
  --bam ${INPUT_DIR}/CUNA/simulate_scripts/bam_files/dna.bam \
  --input ${INPUT_DIR}/CUNA/simulate_scripts/output/deamination.pod5 \
  --ref ${INPUT_DIR}/CUNA/train_models/reference_genome/GRCh38.fa \
  --file_type pod5 \
  --threads 4 \
  --output ${INPUT_DIR}/CUNA/train_models/features_output/ \
  --pos_list ${INPUT_DIR}/CUNA/simulate_scripts/output/mixed_list \
  --window 10 \
  --seq_type dna
```
The output folder will contain .npz files with the extracted features and labels. These files are ready to be used in model training.

## Step 3: Model Training (BiLSTM and Transformer)

In this step, we train neural networks to detect cytosine deamination (C→U) events using the features generated in Step 2. The training is performed using the `train_models.py` script, which supports two architectures:

- BiLSTM (Bidirectional Long Short-Term Memory) – a recurrent model suitable for capturing temporal dependencies in the signal
- Transformer – an attention-based model better suited for learning long-range interactions in both signal and sequence context

Both models take as input the one-hot encoded sequence context (±10 bases), the resampled raw signal window and the binary label (1 = uracil, 0 = thymine).

They are trained to output a modification probability for each sample. You can find a full list of options using python ${INPUT_DIR}/CUNA/train_models/train_models.py --help command. In this demo we will train:

```bash
python ${INPUT_DIR}/CUNA/train_models/train_models.py \
  --mixed_training_dataset ${INPUT_DIR}/CUNA/train_models/features_output/ \
  --validation_type split \
  --validation_fraction 0.2 \
  --model_save_path ${INPUT_DIR}/CUNA/train_models/train_output/bilstm \
  --model_type bilstm \
  --embedding_type one_hot \
  --num_layers 2 \
  --num_fc 128 \
  --fc_type middle \
  --dim_feedforward 256 \
  --embedding_dim 16 \
  --epochs 40 \
  --batch_size 512 \
  --lr 0.0005 \
  --l2_coef 0.0001 \
  --weights auto \
  --seed 0
```

```bash
python ${INPUT_DIR}/CUNA/train_models/train_models.py \
  --mixed_training_dataset ${INPUT_DIR}/CUNA/train_models/features_output/ \
  --validation_type split \
  --validation_fraction 0.1 \
  --model_save_path ${INPUT_DIR}/CUNA/train_models/train_output/transformer \
  --model_type transformer \
  --embedding_type one_hot \
  --num_layers 2 \
  --dim_feedforward 256 \
  --num_fc 128 \
  --fc_type middle \
  --embedding_dim 16 \
  --pe_dim 16 \
  --pe_type fixed \
  --nhead 4 \
  --epochs 35 \
  --batch_size 512 \
  --lr 0.0002 \
  --l2_coef 0.001 \
  --weights auto \
  --seed 0
```
Each model training run produces a long file model.log, epoch-by-epoch training and validation metrics, X saved model checkpoints model.epochX.pt, a model configuration file model.cfg, a file args.txt record of all parameters used for reproducibility and metrics such as accuracy, F1, loss, recall, precision, AUROC, AUPRC and MCC. When we want to use this model, we have to provide a saved checkpoint and the model configuration file to CUNA.

## Step 4: Modification Detection on Test Data

In this final step, we use the trained model to detect uracil modifications in a new DNA dataset.

To prepare the BAM file used for detection, basecall a new modified POD5 file without the --reference option:

```bash
${INPUT_DIR}/dorado-0.5.3-linux-x64/bin/dorado basecaller \
  --emit-moves \
  --model dna_r10.4.1_e8.2_400bps_hac@v4.3.0 \
  ${INPUT_DIR}/CUNA/test/test_data/test.pod5 > ${INPUT_DIR}/CUNA/test/test_data/test.bam
```
We will now use the model model.epoch40.pt and model.cfg on the test dataset using CUNA's detect module. We will provide the model as --model PATH_TO_MODEL_CONFIGURATION_FILE,PATH_TO_MODEL_CHECKPOINT where we provide the paths to model configuration file and model checkpoint separated by a comma to --model parameter.

Each prediction corresponds to a T base in the BAM file. The model will estimate whether that T originated from a true thymine or a deaminated cytosine (uracil), based on the signal pattern and context.
The --motif T 0 argument tells the model to evaluate every T at position 0 of the window. The --mod_symbol U indicates that the predicted modification corresponds to uracil.

The following command runs the detector on the test POD5 file using the trained model:

```bash
python ${INPUT_DIR}/CUNA/CUNA.py detect \
  --model ${INPUT_DIR}/CUNA/train_models/train_output/bilstm/model.cfg,${INPUT_DIR}/CUNA/train_models/train_output/bilstm/model.epoch40.pt \
  --input ${INPUT_DIR}/CUNA/test/test_data/test.pod5 \
  --bam ${INPUT_DIR}/CUNA/test/test_data/test.bam \
  --output ${INPUT_DIR}/CUNA/test/test_output/ \
  --motif T 0 \
  --mod_symbol U \
  --threads 4
```

The detection script generates:
- output.per_read: per-read predictions of modification probability
- output.per_site: per-site predictions of modification probability
- output.bam: BAM file annotated with uracil modification tags
- args.txt: record of the command and options used.

NOTE: All outputs presented in this repository —including performance metrics, model checkpoints, visualizations of signal distributions, and detection results— were generated using the exact configurations and commands provided throughout this README. This ensures complete reproducibility of the experiments.

## Acknowledgements

This project is based on [DeepMod2](https://github.com/WGLab/DeepMod2), developed by Wang Genomics Lab.

Parts of the code and training pipeline were adapted from the original DeepMod2 repository to support the simulation and detection of cytosine deamination events in ancient DNA.
