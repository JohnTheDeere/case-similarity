import torch
import argparse
import yaml
import os

import torch.nn as nn

from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from ..preprocess.preprocess import *
from ..train.model import Model
from dvclive import Live

live, device = Live(), torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def train(config):
    
    # Extract parameters/dataset according to DVC
    learning_rate    = config['train']['learning_rate']
    epochs           = config['train']['epochs']
    temperature      = config['train']['temperature']
    pretrained_model = config['train']['pretrained_model']
    dataset          = torch.load('data/processed/dataset.pt')

    # Initialize model, loss function and optimizer
    model        = Model(pretrained_model).to(device).train()
    criterion    = nn.BCEWithLogitsLoss()
    optimizer    = Adam(model.parameters(), lr=learning_rate)
    train_losses = AverageMeter()

    # Loop over epochs
    for epoch in range(epochs):

        # Loop over batches
        for idx, batch in enumerate(dataset.train_loader):

            optimizer.zero_grad()

            # Move to device to enable running with GPU. Element -1 consists of identifiers (see data_handler) - these cannot be moved to the device
            q_input_ids, q_attention_mask, a_input_ids, a_attention_mask = [element.to(device) for element in batch[:-1]]

            # Forward on questions & answers
            q_embeddings = model(**{'input_ids':q_input_ids, 'attention_mask':q_attention_mask})
            a_embeddings = model(**{'input_ids':a_input_ids, 'attention_mask':a_attention_mask})

            # Build cosine similarity matrix of embeddings
            sim_norm = sim_matrix(q_embeddings, a_embeddings)

            # Get targets e.g. correct answers are on the diagonal
            targets = torch.arange(q_input_ids.shape[0],device=device)
            loss    = criterion(sim_norm/temperature, targets)
            train_losses.update(loss.data.cpu().numpy(), targets.size(0))

            # Backward etc.
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm = 1.0)
            optimizer.step()

            # Run validation and print results
            if idx % 50 == 0:
                    
                # Validate model
                test_losses = validateModel(model, dataset.test_loader, criterion)

                # Log results with dvc
                live.log_metric('Train loss', train_losses.avg)
                live.log_metric('Dev loss', test_losses.avg)
                live.next_step()

                # Print results in console
                print('Epoch: {:3} | Train Loss: {train_loss.avg:.4f} | Test loss: {test_loss.avg:.4f} |'.format(epoch+1, train_loss=train_losses, test_loss=test_losses))

    # Dump model from final epoch
    if not os.path.isdir('models'): os.mkdir('models')
    torch.save(model, 'models/model.pt')



def validateModel(model, devloader, criterion):

    # Initialize new counter for testlosses
    test_losses = AverageMeter()
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(devloader):
            X, Y, masks = [element.to(device, non_blocking=True) for element in data]
            outs = model(X, masks)
            loss = criterion(outs, Y)
            test_losses.update(loss.data.cpu().numpy(), Y.size(0))

    # Set model back to train
    model.train()
    return test_losses



def sim_matrix(a, b, eps=1e-8):
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    config = yaml.safe_load(open(args.config))
    
    # Fine-tune neural network using custom data
    train(config)
