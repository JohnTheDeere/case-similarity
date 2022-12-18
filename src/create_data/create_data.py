import lorem
import uuid
import argparse
import yaml

import pandas as pd


def create(config):

    # Extract parameters from dvc yaml file
    N = config['create_data']['N']

    # Create and store dummy data
    pd.DataFrame([[uuid.uuid4().hex, 
        lorem.paragraph(),
        lorem.paragraph()] for i in range(N)], 
        columns=['identifier', 'question', 'answer']).to_excel('data/raw/data.xlsx', index=False)



if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    config = yaml.safe_load(open(args.config))
    
    # Create dummy data
    create(config)