import torch
import yaml
import argparse

import pandas as pd

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split



def loadData(config):

    # Extract parameters from dvc yaml file
    test_size            = config['preprocess']['test_size']
    pretrained_tokenizer = config['preprocess']['pretrained_tokenizer'] 
    random_state         = config['preprocess']['random_state']

    # Load data
    df = pd.read_excel('data/raw/data.xlsx')

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer)
    
    # Tokenize questions and answers separately 
    encoded_questions, encoded_answers = [tokenizer(df[column].to_list(),
                                                            padding=True,
                                                            truncation=True,
                                                            return_tensors='pt') 
                                                            for column in ['question', 'answer']]

    # This builds train, test
    return [ToDataset({
                'questions':
                            {
                            'input_ids':encoded_questions['input_ids'][indices,:],
                            'attention_mask':encoded_questions['attention_mask'][indices,:]
                            }, 
                'answers':
                            {
                            'input_ids':encoded_answers['input_ids'][indices,:],
                            'attention_mask':encoded_answers['attention_mask'][indices,:]
                            },
                'identifiers':df.identifier[indices].to_list()
            }) for indices in train_test_split(df.index, test_size=test_size, random_state=random_state)]



class BuildDataset():

    def __init__(self, config):
        
        # Create dataloaders
        self.train_loader, self.test_loader = [DataLoader(dataset=dataset, 
                                                          shuffle=True, 
                                                          batch_size=config['preprocess']['batch_size']) 
                                                          for dataset in loadData(config)]



class ToDataset(Dataset):

    def __init__(self, inputs):
        
        # Build items in class instance
        self.questions   = inputs['questions']
        self.answers     = inputs['answers']
        self.identifiers = inputs['identifiers']

    def __len__(self):
        return self.questions['input_ids'].shape[0]

    def __getitem__(self,index):
        return (self.questions['input_ids'][index],
                self.questions['attention_mask'][index],
                self.answers['input_ids'][index], 
                self.answers['attention_mask'][index], 
                self.identifiers[index])



if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    config = yaml.safe_load(open(args.config))
    
    # Build dataset and store it
    dataset = BuildDataset(config)
    torch.save(dataset,'data/processed/dataset.pt')
