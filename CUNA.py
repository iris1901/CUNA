#!/usr/bin/env python

import time, itertools, torch
import datetime, os, shutil, argparse, sys, pysam
from src import utils
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--print_models", help='Print details of models available', default=False, action='store_true')
    main_subparsers = parser.add_subparsers(title="Options", dest="option")
    parent_parser = argparse.ArgumentParser(add_help=False)

    parent_parser.add_argument("--prefix", help='Prefix for the output files', type=str, default='output')
    parent_parser.add_argument("--output", help='Path to folder where intermediate and final files will be stored, default is current working directory', type=str)
    parent_parser.add_argument("--qscore_cutoff", help='Minimum cutoff for mean quality score of a read', type=float, default=0)
    parent_parser.add_argument("--length_cutoff", help='Minimum cutoff for read length', type=int, default=0)
    parent_parser.add_argument("--mod_t", help='Probability threshold for a per-read prediction to be considered modified.', default=0.5, type=float)
    parent_parser.add_argument("--unmod_t", help='Probability threshold for a per-read prediction to be considered unmodified.', default=0.5, type=float)

    detect_parser = main_subparsers.add_parser("detect", parents=[parent_parser], add_help=True, help="Call modification from Dorado basecalled POD5 files using move tables for signal alignment.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    detect_required = detect_parser.add_argument_group("Required Arguments")

    detect_parser.add_argument("--motif", required=True, nargs= '+', help='Motif to detect. Format: "<MOTIF> <INDEX>". Example: "T 0" to detect modifications on T.')
    detect_parser.add_argument("--mod_symbol", help='Symbol to use for modified base in BAM tag MM (e.g. "u" for uracil).', type=str)
    detect_parser.add_argument("--threads", help='Number of threads to use for processing signal and running model inference. Recommended: at least 4.', type=int, default=4)

    
    detect_required.add_argument("--model", help='Name of the model to use. For custom models, provide "config.cfg,model.pt".', type=str, required=True)
    detect_required.add_argument("--bam", help='Path to aligned BAM file from Dorado basecalling.', type=str, required=True)
    detect_required.add_argument("--input", help='Path to POD5 file or folder containing POD5 files.', type=str, required=True)
   
    detect_parser.add_argument("--skip_per_site", help='Skip per-site output generation.', default=False, action='store_true')
    detect_parser.add_argument("--device", help='Device to use for model inference: "cpu", "cuda", "cuda:0", "mps", etc.', type=str)
    detect_parser.add_argument("--disable_pruning", help='Disable model pruning (may slow down CPU inference).', default=False, action='store_true')
    detect_parser.add_argument("--batch_size", help='Batch size to use for GPU inference.', type=int, default=1024)
    detect_parser.add_argument("--bam_threads", help='Number of threads for BAM output compression.', type=int, default=4)
    detect_parser.add_argument("--skip_unmapped", help='Skip unmapped reads from modification calling.', default=False, action='store_true')

    merge_parser = main_subparsers.add_parser("merge", parents=[parent_parser], add_help=True, help="Merge per-read modification calls into per-site calls (e.g., to identify desaminations C→U)", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    merge_parser.add_argument("--input", nargs='*', help='List of paths of per-read modification calls to merge. File paths should be separated by space/whitespace. Use either --input or --list argument, but not both.')
    merge_parser.add_argument("--list", help='A file containing paths to per-read modification calls to merge (one per line). Use either --input or --list argument, but not both.', type=str)

    if len(sys.argv)==1:
        parser.print_help()
        parser.exit()
        
    
    elif len(sys.argv)==2:
        if sys.argv[1]=='merge':
            merge_parser.print_help()
            merge_parser.exit()
        
        elif sys.argv[1]=='detect':
            detect_parser.print_help()
            detect_parser.exit()

    args = parser.parse_args()
    
    
    if args.print_models:
        utils.get_model_help()
        parser.exit()
        
    t=time.time()

    print('%s: Starting DeepMod2.' %str(datetime.datetime.now()), flush=True)
            
    if not args.output:
        args.output=os.getcwd()
    
    os.makedirs(args.output, exist_ok=True)

    if args.option == 'merge':
        if args.input:
            input_list = args.input
        elif args.list:
            with open(args.list, 'r') as file_list:
                input_list = [x.rstrip('\n') for x in file_list.readlines()]

        params = {
            'output': args.output,
            'prefix': args.prefix,
            'qscore_cutoff': args.qscore_cutoff,
            'length_cutoff': args.length_cutoff,
            'mod_t': args.mod_t,
            'unmod_t': args.unmod_t
        }

        site_pred_file = utils.get_per_site(params, input_list)

        
    else:
        dev = (
            args.device if args.device else
            "mps" if torch.backends.mps.is_available() else
            "cuda" if torch.cuda.is_available() else
            "cpu"
        )
        print(f"Selected inference device: {dev}", flush=True)

        motif_seq, exp_motif_seq, motif_ind, valid_motif = utils.motif_check(args.motif)
        if not valid_motif:
            sys.exit(3)

        params = {
            'input': args.input,
            'output': args.output,
            'threads': args.threads,
            'prefix': args.prefix,
            'model': args.model,
            'qscore_cutoff': args.qscore_cutoff,
            'length_cutoff': args.length_cutoff,
            'bam': args.bam,
            'mod_t': args.mod_t,
            'unmod_t': args.unmod_t,
            'skip_per_site': args.skip_per_site,
            'dev': dev,
            'disable_pruning': args.disable_pruning,
            'batch_size': args.batch_size,
            'bam_threads': args.bam_threads,
            'skip_unmapped': args.skip_unmapped,
            'motif_seq': motif_seq,
            'motif_ind': motif_ind,
            'exp_motif_seq': exp_motif_seq,
            'mod_symbol': args.mod_symbol,
            'file_type': 'pod5'
        }

        print('\n%s: \nCommand: python %s\n' % (str(datetime.datetime.now()), ' '.join(sys.argv)), flush=True)

        with open(os.path.join(args.output, 'args'), 'w') as file:
            file.write('Command: python %s\n\n\n' % (' '.join(sys.argv)))
            file.write('------Parameters Used For Running DeepMod2------\n')
            for k in vars(args):
                file.write('{}: {}\n'.format(k, vars(args)[k]))

        from src import detect
        detect.call_manager(params)

    print('\n%s: Time elapsed=%.4fs' %(str(datetime.datetime.now()),time.time()-t), flush=True)