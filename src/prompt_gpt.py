import argparse, os
import logging
from exp_helper import *

DEFAULT_DIR = os.path.abspath(os.getcwd())
DATASET_DIR = os.path.join(DEFAULT_DIR, "data/") 
DEFAULT_RESULTS_DIR = os.path.join(DEFAULT_DIR, "results/")
FEATURE_LIST_FNAME = 'GPT_3_feature_df - Sheet1.csv'
OPENAI_API_KEY_TIM = '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'
OPENAI_API_KEY_SID = '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'



def main():
    parser = argparse.ArgumentParser(description="""""")
    parser.add_argument('--dataset_name', type=str, 
                        help="""Name of the dataset""")
    parser.add_argument('--exp_name',
                        type=str, help="""specify the experiment you want to run""")
    parser.add_argument('--feature_list_fname',
                    type=str, help=""" Name of the feature listing file""")
    parser.add_argument('--model',
                    type=str, help=""" Name of the feature listing file""")
    args = parser.parse_args()
    logging.basicConfig(filename="logs/{}_{}.log".format(args.exp_name, args.dataset_name), encoding='utf-8', level=logging.DEBUG, 
                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    logging.warning('is when this event was logged.')
    logging.info('Running experiments with the following parameters')
    for arg, value in sorted(vars(args).items()):
        logging.info("Argument %s: %r", arg, value)

    run_exp(exp_name = args.exp_name, 
            dataset_name = args.dataset_name, 
            dataset_dir = DATASET_DIR , 
            feature_list_fname = args.feature_list_fname, 
            model = args.model, 
            openai_api_key = OPENAI_API_KEY_TIM, 
            results_dir = DEFAULT_RESULTS_DIR)

if __name__=="__main__":
    main()
