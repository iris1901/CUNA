import math, time, argparse, re,  os, sys
import functools, itertools, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import optim, Tensor
import numpy as np
import multiprocessing as mp
import multiprocessing as mp
import queue
import pickle
from utils import *
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef

# One-hot encoding for base or reference sequences.
class OneHotEncode(nn.Module):
    def _init_(self, num_classes: int):
        super()._init_()
        self.num_classes=num_classes
    def forward(self, x: Tensor) -> Tensor:
        return F.one_hot(x, self.num_classes)

# Applies embedding only to the read sequence (one-hot or learnable).
class ReadEmbed(nn.Module):
    def _init_(self, embedding_dim, embedding_type):
        super()._init_()
        
        self.embedding_depth=0
        
        if embedding_type=='one_hot':
            self.read_emb=OneHotEncode(4)
            self.embedding_depth+=4
        
        elif embedding_type=='learnable':
            self.read_emb=nn.Embedding(4, embedding_dim)
            self.embedding_depth+=embedding_dim
            
    def forward(self, batch_base_seq):
        batch_base_seq_emb=self.read_emb(batch_base_seq)
        
        return batch_base_seq_emb

# Fixed positional encoding using sine/cosine functions.
class PositionalEncoding(nn.Module):
    def _init_(self, pe_dim: int, max_len: int):
        super()._init_()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, pe_dim, 2) * (-math.log(pe_dim) / (pe_dim)))
        pe = torch.zeros(1, max_len, pe_dim)
        pe[0,:, 0::2] = torch.sin(position * div_term)
        pe[0,:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x: Tensor) -> Tensor:
        x_pos=torch.Tensor.repeat(self.pe,(x.size(0),1,1)) 
        x = torch.cat((x, x_pos),2)
        return x

# Learnable positional encoding using embedding layers.
class PositionalEmbedding(nn.Module):
    def _init_(self, pe_dim: int, max_len: int):
        super()._init_()
        pos=torch.arange(max_len)
        self.register_buffer('pos', pos)
        self.pe=nn.Embedding(max_len, pe_dim)
    
    def forward(self, x: Tensor) -> Tensor:
        x_pos=self.pe(self.pos)
        x_pos=torch.Tensor.repeat(x_pos,(x.size(0),1,1)) 
        x = torch.cat((x, x_pos),2)
        return x

# Positional encoding as directly learnable parameters.
class PositionalParameter(nn.Module):
    def _init_(self, pe_dim: int, max_len: int):
        super()._init_()
        
        self.pe=torch.nn.Parameter(torch.randn(max_len, pe_dim)) 

    def forward(self, x: Tensor) -> Tensor:
        x_pos=torch.Tensor.repeat(self.pe,(x.size(0),1,1)) 
        x = torch.cat((x, x_pos),2)
        return x

# Positional encoding as directly learnable parameters.
class ClassifierMiddle(nn.Module):
    def _init_(self, in_dim: int, num_fc: int, model_len: int):
        super()._init_()
        self.mid = model_len//2
        self.fc = nn.Linear(in_dim, num_fc)
        self.out = nn.Linear(num_fc,1)
        
    def forward(self, x):
        x = F.relu(self.fc(x[:,self.mid, :]))
        x=self.out(x)
        return x
    
# Classifier that flattens the entire sequence and uses all the information.
class ClassifierAll(nn.Module):
    def _init_(self, in_dim: int, num_fc: int):
        super()._init_()
        self.fc = nn.Linear(in_dim, num_fc)
        self.out = nn.Linear(num_fc,1)
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc(x))
        x=self.out(x)
        return x  

# BiLSTM-based model to learn sequences with optional embedding and classification layer.
class BiLSTM(nn.Module):
    def _init_(self, model_dims, num_layers, dim_feedforward, num_fc, embedding_dim, embedding_type, fc_type):
        super(BiLSTM, self)._init_()
        
        self.emb=ReadEmbed(embedding_dim, embedding_type)
        self.model_len=model_dims[0]
        self.model_depth=model_dims[1]+self.emb.embedding_depth
        
        self.bilstm = nn.LSTM(input_size=self.model_depth, hidden_size=dim_feedforward, num_layers=num_layers, bidirectional=True, batch_first = True)
        
        if fc_type=='middle':
            self.classifier=ClassifierMiddle(in_dim=dim_feedforward*2, num_fc=num_fc, model_len=self.model_len)
        
        else:
            self.classifier=ClassifierAll(in_dim=self.model_len*dim_feedforward*2, num_fc=num_fc)

    def forward(self, batch_x, batch_base_seq):
        seq_emb=self.emb(batch_base_seq)
        x=torch.cat((batch_x, seq_emb), 2)
        x, _=self.bilstm(x)
        x = self.classifier(x)
        return x

# Transformer-based model with positional encoding and embedding, followed by a classification layer.
class TransformerModel(nn.Module):
    def _init_(self, model_dims, num_layers, dim_feedforward, num_fc, embedding_dim, embedding_type, include_ref, pe_dim, nhead, pe_type, fc_type):
        super(TransformerModel, self)._init_()
        
        self.emb=ReadEmbed(embedding_dim, embedding_type)
        self.model_len=model_dims[0]
        
        if pe_type=='fixed':
            self.pe_block=PositionalEncoding(pe_dim=pe_dim, max_len=self.model_len)
        
        elif pe_type=='embedding':
            self.pe_block=PositionalEmbedding(pe_dim=pe_dim, max_len=self.model_len)
            
        elif pe_type=='parameter':                
            self.pe_block=PositionalParameter(pe_dim=pe_dim, max_len=self.model_len)

        self.model_depth=model_dims[1]+self.emb.embedding_depth+pe_dim
        self.pad_length=math.ceil(self.model_depth/nhead)*nhead-self.model_depth        
        pad=torch.zeros(1,self.model_len, self.pad_length)
        self.register_buffer('pad', pad)
        self.model_depth+=self.pad_length
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.model_depth, nhead=nhead, dim_feedforward=dim_feedforward, dropout=0, batch_first = True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        if fc_type=='middle':
            self.classifier=ClassifierMiddle(in_dim=self.model_depth, num_fc=num_fc, model_len=self.model_len)
        
        else:
            self.classifier=ClassifierAll(in_dim=self.model_len*self.model_depth, num_fc=num_fc)

    def forward(self, batch_x, batch_base_seq):
        seq_emb=self.emb(batch_base_seq)
        x=torch.cat((batch_x, seq_emb), 2)
        x=self.pe_block(x)        
        x_pad=torch.Tensor.repeat(self.pad,(x.size(0),1,1)) 
        x = torch.cat((x, x_pad),2)
        
        x=self.transformer_encoder(x)
        x = self.classifier(x)
        return x

# Trains a model (BiLSTM or Transformer) for modification detection.
# Performs epoch-based training, computes metrics, saves checkpoints, and generates performance plots.
def train(training_dataset, validation_dataset, validation_type, validation_fraction, model_config, epochs, prefix, retrain, batch_size, args_str, seed, args):

    print('Starting training.', flush=True)
    torch.manual_seed(seed)
    model_type = model_config['model_type']
    model_save_path = model_config.pop('model_save_path')

    if torch.backends.mps.is_available():
        dev = "mps"
    elif torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    print(f'Using device: {dev}', flush=True)

    weight_counts = np.array([
        np.sum(np.eye(2)[np.load(f)['label'].astype(int)], axis=0)
        for f in itertools.chain.from_iterable(training_dataset)])
    
    weight_counts = np.sum(weight_counts, axis=0)

    if model_config['weights'] == 'equal':
        pos_weight = torch.Tensor([1.0])
    elif model_config['weights'] == 'auto':
        pos_weight = torch.Tensor([weight_counts[0] / weight_counts[1]])
    else:
        pos_weight = torch.Tensor([float(model_config['weights'])])

    pos_weight = pos_weight.to(dev)

    print(f'Number of Modified Instances={weight_counts[1]}\nNumber of Un-Modified Instances={weight_counts[0]}\nPositive Label Weight={pos_weight}\n', flush=True)

    if model_type == 'bilstm':
        net = BiLSTM(
            model_dims=model_config['model_dims'],
            num_layers=model_config['num_layers'],
            dim_feedforward=model_config['dim_feedforward'],
            num_fc=model_config['num_fc'],
            embedding_dim=model_config['embedding_dim'],
            embedding_type=model_config['embedding_type'],
            fc_type=model_config['fc_type'])
        
    elif model_type == 'transformer':
        net = TransformerModel(
            model_dims=model_config['model_dims'],
            num_layers=model_config['num_layers'],
            dim_feedforward=model_config['dim_feedforward'],
            num_fc=model_config['num_fc'],
            embedding_dim=model_config['embedding_dim'],
            embedding_type=model_config['embedding_type'],
            include_ref= False,
            pe_dim=model_config['pe_dim'],
            nhead=model_config['nhead'],
            pe_type=model_config['pe_type'],
            fc_type=model_config['fc_type'])

    net.to(dev)
    optimizer = optim.AdamW(net.parameters(), lr=model_config['lr'], weight_decay=model_config['l2_coef'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    if retrain:
        checkpoint = torch.load(retrain)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model_details = str(net.to(torch.device(dev)))
    print(model_details, flush=True)
    num_params = sum(p.numel() for p in net.parameters())
    print('# Parameters=', num_params, flush=True)

    config_path = os.path.join(model_save_path, f'{prefix}.cfg')
    with open(config_path, 'wb') as handle:
        pickle.dump(model_config, handle, protocol=pickle.HIGHEST_PROTOCOL)

    log_file_path = os.path.join(model_save_path, f'{prefix}.log')

    best_val_loss = float('inf')

    metrics_history = {
    'train_loss': [], 'train_accuracy': [], 'train_precision': [], 'train_recall': [], 'train_f1': [],
    'test_loss': [], 'test_accuracy': [], 'test_precision': [], 'test_recall': [], 'test_f1': [],
    'train_auroc': [], 'train_auprc': [], 'train_mcc': [],
    'test_auroc': [], 'test_auprc': [], 'test_mcc': []}

    with open(log_file_path, 'w') as log_file:
        log_file.write(args_str)
        log_file.write(f'\n# Parameters={num_params}\n')
        log_file.write(model_details)

        best_val_f1 = 0
        epochs_without_improvement = 0

        for j in range(epochs):
            net.train()
            metrics_train = {'Normal': {'TP': 0, 'FP': 0, 'FN': 0, 'loss': 0, 'len': 0, 'true': 0}}
            metrics_test = {'Normal': {'TP': 0, 'FP': 0, 'FN': 0, 'loss': 0, 'len': 0, 'true': 0}}
            t = time.time()

            train_gen = generate_batches_mixed(training_dataset, validation_type, validation_fraction, data_type="train", batch_size=batch_size)
            for batch in train_gen:
                batch_x, batch_base_seq, batch_y = batch
                batch_x, batch_base_seq, batch_y = batch_x.to(dev), batch_base_seq.to(dev), batch_y.to(dev)
                optimizer.zero_grad()
                score = net(batch_x, batch_base_seq)
                loss = F.binary_cross_entropy_with_logits(score, batch_y, pos_weight=pos_weight)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()
                get_metrics(metrics_train, 'Normal', batch_y, score, loss)

            net.eval()
            with torch.no_grad():
                if validation_type == 'split':
                    test_gen = generate_batches(list(itertools.chain.from_iterable(training_dataset)), validation_type, validation_fraction, data_type="test", batch_size=batch_size)
                else:
                    test_gen = generate_batches(validation_dataset, validation_type, validation_fraction, data_type="test", batch_size=batch_size)

                for batch in test_gen:
                    batch_x, batch_base_seq, batch_y = batch
                    batch_x, batch_base_seq, batch_y = batch_x.to(dev), batch_base_seq.to(dev), batch_y.to(dev)
                    score = net(batch_x, batch_base_seq)
                    loss = F.binary_cross_entropy_with_logits(score, batch_y, pos_weight=pos_weight)
                    get_metrics(metrics_test, 'Normal', batch_y, score, loss)

            train_str, train_vals_dict = get_stats(metrics_train, 'Training', return_dict=True)
            test_str, test_vals_dict = get_stats(metrics_test, 'Testing', return_dict=True)
            
            if args.early_stopping > 0:
                current_val_f1 = test_vals_dict['f1']
                if current_val_f1 > best_val_f1:
                    best_val_f1 = current_val_f1
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= args.early_stopping:
                    break

            val_loss = test_vals_dict['loss']

            def get_logits_and_labels(net, generator, dev):
                y_true = []
                y_score = []
                with torch.no_grad():
                    for batch in generator:
                        batch_x, batch_base_seq, batch_y = batch
                        batch_x, batch_base_seq = batch_x.to(dev), batch_base_seq.to(dev)
                        logits = net(batch_x, batch_base_seq).squeeze()
                        probs = torch.sigmoid(logits)
                        y_true.append(batch_y.cpu().numpy().flatten())
                        y_score.append(probs.cpu().numpy())
                return np.concatenate(y_true), np.concatenate(y_score)

            train_gen_eval = generate_batches(list(itertools.chain.from_iterable(training_dataset)), validation_type, validation_fraction, data_type="train", batch_size=batch_size)
            test_gen_eval = generate_batches(validation_dataset, validation_type, validation_fraction, data_type="test", batch_size=batch_size) if validation_type == "dataset" else \
                            generate_batches(list(itertools.chain.from_iterable(training_dataset)), validation_type, validation_fraction, data_type="test", batch_size=batch_size)

            y_true_train, y_score_train = get_logits_and_labels(net, train_gen_eval, dev)
            y_true_test, y_score_test = get_logits_and_labels(net, test_gen_eval, dev)

            try:
                metrics_history['train_auroc'].append(roc_auc_score(y_true_train, y_score_train))
                metrics_history['train_auprc'].append(average_precision_score(y_true_train, y_score_train))
                metrics_history['train_mcc'].append(matthews_corrcoef(y_true_train, y_score_train > 0.5))

                metrics_history['test_auroc'].append(roc_auc_score(y_true_test, y_score_test))
                metrics_history['test_auprc'].append(average_precision_score(y_true_test, y_score_test))
                metrics_history['test_mcc'].append(matthews_corrcoef(y_true_test, y_score_test > 0.5))
            except:
                metrics_history['train_auroc'].append(float('nan'))
                metrics_history['train_auprc'].append(float('nan'))
                metrics_history['train_mcc'].append(float('nan'))
                metrics_history['test_auroc'].append(float('nan'))
                metrics_history['test_auprc'].append(float('nan'))
                metrics_history['test_mcc'].append(float('nan'))


            epoch_log = f'\n\nEpoch {j+1}: #Train={sum(x["len"] for x in metrics_train.values())}  #Test={sum(x["len"] for x in metrics_test.values())}  Time={time.time() - t:.4f}\n{train_str}\n\n{test_str}'
            print(epoch_log, flush=True)
            log_file.write(epoch_log)
            log_file.flush()
            os.fsync(log_file.fileno())

            for metric in ['loss', 'accuracy', 'precision', 'recall', 'f1']:
                metrics_history[f'train_{metric}'].append(train_vals_dict[metric])
                metrics_history[f'test_{metric}'].append(test_vals_dict[metric])

            epoch_model_path = os.path.join(model_save_path, f'model.epoch{j+1}.pt')
            torch.save({
                'epoch': j + 1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, epoch_model_path)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_path = os.path.join(model_save_path, f'model.best.pt')
                torch.save({
                    'epoch': j + 1,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, model_path)

    def smooth(y, box_pts=3):
        pad = box_pts // 2
        y_padded = np.pad(y, (pad, pad), mode='edge') 
        box = np.ones(box_pts)/box_pts
        return np.convolve(y_padded, box, mode='valid') 

    epochs_range = range(1, len(metrics_history['train_f1']) + 1)

    for metric in ['loss', 'accuracy', 'precision', 'recall', 'f1']:
        plt.figure()
        plt.plot(epochs_range, smooth(metrics_history[f'train_{metric}']), label=f'Train {metric.title()}')
        plt.plot(epochs_range, smooth(metrics_history[f'test_{metric}']), label=f'Test {metric.title()}')
        plt.xlabel('Epoch')
        plt.ylabel(metric.title())
        plt.title(f'{metric.title()} per Epoch')
        plt.legend()
        plt.savefig(os.path.join(model_save_path, f'{prefix}_{metric}.png'))
        plt.close()

    plt.figure()
    for metric in ['loss', 'accuracy', 'precision', 'recall', 'f1']:
        plt.plot(epochs_range, metrics_history[f'test_{metric}'], label=f'Test {metric.title()}')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Test Metrics per Epoch')
    plt.legend()
    plt.savefig(os.path.join(model_save_path, f'{prefix}_all_metrics.png'))
    plt.close()

    for metric in ['auroc', 'auprc', 'mcc']:
        plt.figure()
        plt.plot(epochs_range, metrics_history[f'train_{metric}'], label=f'Train {metric.title()}')
        plt.plot(epochs_range, metrics_history[f'test_{metric}'], label=f'Test {metric.title()}')
        plt.xlabel('Epoch')
        plt.ylabel(metric.title())
        plt.title(f'{metric.title()} per Epoch')
        plt.legend()
        plt.savefig(os.path.join(model_save_path, f'{prefix}_{metric}.png'))
        plt.close()

    return net

# Main entry point of the script.
# Parses training arguments, configures the model, and runs the train function.
# Also saves the used parameters and ensures reproducibility using a random seed.
if _name=='main_':
    start_time = time.time()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--mixed_training_dataset", nargs='*', required=True,
                        help='Training dataset with mixed labels. A whitespace separated list of folders containing .npz files or paths to individual .npz files.')

    parser.add_argument("--validation_type", choices=['split', 'dataset'], default="split",
                        help='Validation strategy: "split" uses a fraction of training data, "dataset" uses separate validation data.')
    parser.add_argument("--validation_fraction", type=float, default=0.2,
                        help='Fraction of training dataset to use for validation when validation_type is "split".')
    parser.add_argument("--validation_dataset", nargs='*',
                        help='Validation dataset for "dataset" mode. A list of folders with .npz files or paths to .npz files.')

    parser.add_argument("--prefix", default='model',
                        help='Prefix name for the model checkpoints.')
    parser.add_argument("--weights", default='equal',
                        help='Weight for positive (modified) label in loss. Options: "equal", "auto", or a numeric value.')
    parser.add_argument("--model_save_path", required=True,
                        help='Path to save trained model checkpoints.')
    parser.add_argument("--epochs", type=int, default=100,
                        help='Number of training epochs.')
    parser.add_argument("--batch_size", type=int, default=256,
                        help='Batch size for training.')
    parser.add_argument("--retrain", default=None,
                        help='Path to a saved model checkpoint for retraining.')

    parser.add_argument("--fc_type", choices=['middle', 'all'], default='all',
                        help='Type of fully connected layers used in classifier.')
    parser.add_argument("--model_type", choices=['bilstm', 'transformer'], required=True,
                        help='Model architecture type.')
    parser.add_argument("--num_layers", type=int, default=3,
                        help='Number of BiLSTM or Transformer encoder layers.')
    parser.add_argument("--dim_feedforward", type=int, default=100,
                        help='Dimension of BiLSTM hidden units or Transformer feedforward layers.')
    parser.add_argument("--num_fc", type=int, default=16,
                        help='Size of fully connected layer between encoder and classifier.')
    parser.add_argument("--embedding_dim", type=int, default=4,
                        help='Size of base embedding dimension.')
    parser.add_argument("--embedding_type", choices=['learnable', 'one_hot'], default='one_hot',
                        help='Embedding type for bases.')
    parser.add_argument("--pe_dim", type=int, default=16,
                        help='Dimension for positional encoding in Transformer.')
    parser.add_argument("--pe_type", choices=['fixed', 'embedding', 'parameter'], default='fixed',
                        help='Type of positional encoding.')
    parser.add_argument("--nhead", type=int, default=4,
                        help='Number of attention heads in Transformer.')
    parser.add_argument("--lr", type=float, default=1e-4,
                        help='Learning rate.')
    parser.add_argument("--l2_coef", type=float, default=1e-5,
                        help='L2 regularization coefficient.')
    parser.add_argument("--early_stopping", type=int,default=0, help='Number of epochs to wait without improvement in validation F1 before stopping. 0 to disable.')
    parser.add_argument("--seed", default=None,
                        help='Random seed for reproducibility.')

    args = parser.parse_args()

    os.makedirs(args.model_save_path, exist_ok=True)

    mixed_training_dataset = get_files(args.mixed_training_dataset)
    validation_dataset = get_files(args.validation_dataset)
    validation_type = args.validation_type
    validation_fraction = args.validation_fraction

    valid_data, window, norm_type, strides_per_base, model_depth, full_signal = check_training_files(
        mixed_training_dataset, validation_dataset)

    if not valid_data:
        sys.exit(3)

    model_len = strides_per_base * (2 * window + 1)

    model_config = dict(
        model_dims=(model_len, model_depth + 1),
        window=window,
        model_type=args.model_type,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        num_fc=args.num_fc,
        embedding_dim=args.embedding_dim,
        embedding_type=args.embedding_type,
        pe_dim=args.pe_dim,
        nhead=args.nhead,
        pe_type=args.pe_type,
        l2_coef=args.l2_coef,
        lr=args.lr,
        model_save_path=args.model_save_path,
        fc_type=args.fc_type,
        train_w_wo_ref=False,
        weights=args.weights,
        norm_type=norm_type,
        full_signal=full_signal,
        strides_per_base=strides_per_base)

    args_dict = vars(args)
    args_str = ''.join('%s: %s\n' % (k, str(v)) for k, v in args_dict.items())
    print(args_str, flush=True)

    seed = random.randint(0, 0xffff_ffff_ffff_ffff) if args.seed is None else int(args.seed)

    training_dataset = [mixed_training_dataset, [], []]

    with open(os.path.join(args.model_save_path, 'args'), 'w') as file:
        file.write('Command: python %s\n\n\n' % (' '.join(sys.argv)))
        file.write('------Parameters Used For Running CUNA------\n')
        for k in vars(args):
            file.write('{}: {}\n'.format(k, vars(args)[k]))

    res = train(
        training_dataset,
        validation_dataset,
        validation_type,
        validation_fraction,
        model_config,
        epochs=args.epochs,
        prefix=args.prefix,
        retrain=args.retrain,
        batch_size=args.batch_size,
        args_str=args_str,
        seed=seed,
        args=args)

    print('Time taken=%.4f' % (time.time() - start_time), flush=True)
