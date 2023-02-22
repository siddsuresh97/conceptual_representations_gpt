import argparse, os
import logging
from exp_helper import *

DEFAULT_DIR = os.path.abspath(os.getcwd())
DATASET_DIR = os.path.join(DEFAULT_DIR, "data/")
DEFAULT_RESULTS_DIR = os.path.join(DEFAULT_DIR, "results/")
FEATURE_LIST_FNAME = 'GPT_3_feature_df - Sheet1.csv'



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
    parser.add_argument('--temperature',help = """Tradeoff between deterministic and creative responses of gpt""")
    parser.add_argument('--dataset_dir',help = """dataset directory""", default = DATASET_DIR)
    parser.add_argument('--results_dir',help = """results directory""", default = DEFAULT_RESULTS_DIR)
    parser.add_argument('--local_rank',help = """device local rank""", default =0)
    parser.add_argument('--sample', action='store_true', help = """sample or not""")
    args = parser.parse_args()
    logging.basicConfig(filename="logs/{}_{}_{}.log".format(args.exp_name, args.dataset_name, args.model), level=logging.DEBUG, # encoding='utf-8',
                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    logging.warning('is when this event was logged.')
    logging.info('Running experiments with the following parameters')

    # local_rank  = int(os.environ['LOCAL_RANK']) or args.local_rank
    # print('local_rank', local_rank)
    # device = "cuda:{}".format(local_rank)
    for arg, value in sorted(vars(args).items()):
        logging.info("Argument %s: %r", arg, value)

    run_exp(exp_name = args.exp_name,
            dataset_name = args.dataset_name,
            dataset_dir = args.dataset_dir ,
            feature_list_fname = args.feature_list_fname,
            model = args.model,
            openai_api_key = None, #os.environ['OPENAI_API_KEY_SID']
            results_dir = args.results_dir,
            temperature = float(args.temperature), 
            sample = args.sample
            # device = device
            )

if __name__=="__main__":
    main()
