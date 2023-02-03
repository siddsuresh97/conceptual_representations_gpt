import argparse, os
import logging
from unicodedata import category

from click import prompt
from exp_helper import *

DEFAULT_DIR = os.path.abspath(os.getcwd())
DATASET_DIR = os.path.join(DEFAULT_DIR, "data/") 
DEFAULT_RESULTS_DIR = os.path.join(DEFAULT_DIR, "results/")
TOOLS = ['Spanner',  'Oil can', 'Paint brush', 'Saw', 'Vacuum',
         'Screwdriver','Axe', 'Shovel', 'Lawn Mower','Grinding disk',
          'Nail','Chisel','Hammer','Knife',  'Anvil']

REPTILES = ['Salamander',
 'Blindworm',
 'Crocodile',
 'Lizard',
 'Toad',
 'Tortoise',
 'Caiman',
 'Chameleon',
 'Gecko',
 'Turtle',
 'Cobra',
 'Snake',
 'Boa python',
 'Dinosaur',
 'Alligator']

def save_feature_listing_results_in_csv(results_dir, dataset_name, model, exp_name, temperature):
    file = open(os.path.join(results_dir, dataset_name, model +'_'+ exp_name + '_full_temperature_' + str(temperature)),'rb')
    answer_dict = pickle.load(file)
    actual_total_tokens = 0
    estimated_total_tokens = 0
    concept_list = []
    feature_list = []
    full_answer_list = []
    answer_list = []
    prompt_list = []
    category_list = []
    for k, v in answer_dict.items():
        # actual_total_tokens +=  v[0]['usage']['total_tokens']
        # estimated_total_tokens += v[1] 
        concept_list.append(k[0])
        feature_list.append(k[1])
        prompt_list.append(v[2])
        if model != 'flan':
            answer = v[0]['choices'][0]['text'] 
        else:
            answer = v[0]
        full_answer_list.append(answer)
        # print(k[0], k[1], v[0]['choices'][0]['text'])
        if 'es' in answer:
            answer_list.append(1)
        elif 'o' in answer:
            answer_list.append(0)
        else:
            answer_list.append('SOMETHING WENT WRONG')
            logging.error('Invalid answer')
        if k[0] in REPTILES:
            category_list.append('reptile')
        elif k[0] in TOOLS:
            category_list.append('tool')
        else:
            logging.error('Invalid category')
    result_df = pd.DataFrame({'Concept':concept_list, 'Feature':feature_list, 'Yes/No':answer_list, 'Category':category_list, 'prompt':prompt_list, 'gpt_response':full_answer_list})
    result_df.to_csv(os.path.join(results_dir, dataset_name, results_dir, dataset_name, model +'_'+ exp_name + '_feature_list_temperature_0.csv'))
    file.close()
    logging.info('Estimated cost : {}'.format((estimated_total_tokens/1000)*0.06))
    logging.info('Actual cost : {}'.format((actual_total_tokens/1000)*0.06))

def save_feature_triplet_results_in_csv(results_dir, dataset_name, model, exp_name):
    if dataset_name == 'reptile_tool':
        file = open(os.path.join(results_dir, dataset_name, model +'_'+ exp_name + '_full_temperature_0.0'),'rb')
    elif dataset_name == 'social_categories':
        file = open(os.path.join(results_dir, dataset_name, model +'_'+ exp_name + '_full'),'rb')
    answer_dict = pickle.load(file)
    file.close()
    actual_total_tokens = []
    estimated_total_tokens = []
    concept1_list = []
    concept2_list = []
    anchor_list = []
    full_answer_list = []
    answer_list = []
    prompt_list = []
    category_list = []
    for k, v in answer_dict.items():
        actual_total_tokens.append(v[0]['usage']['total_tokens'])
        estimated_total_tokens.append(v[1]+ESTIMATED_RESPONSE_TOKENS)
        prompt_list.append(v[2])
        anchor_list.append(k[0])
        concept1_list.append(k[1])
        concept2_list.append(k[2])
        answer = v[0]['choices'][0]['text'] 
        full_answer_list.append(answer)
        try:
            if k[1] and k[0] in answer:
                answer_list.append('Fill manually') 
            elif k[1] in answer:
                answer_list.append(k[1])
            elif k[2] in answer:
                answer_list.append(k[2])
            elif k[0] in answer:
                answer_list.append('fill manually')
                logging.info('{}\n{}\n{}\n{}'.format(k[0], k[1], k[2], v[2], answer))
                print(k[0], k[1], k[2], v[2], answer)
            else:
                logging.info('Unexpected answer') 
                logging.info('{}\n{}\n{}\n{}'.format(k[0], k[1], k[2], v[2], answer))
                answer_list.append('fill manually')
                print('Unexpected answer')
                print(k[0], k[1], k[2], v[2], answer) 
        except:
            answer_list.append('fill manually')
        if dataset_name == 'reptile_tool':   
            if k[0] in REPTILES:
                category_list.append('reptile')
            elif k[0] in TOOLS:
                category_list.append('tool')
            else:
                logging.error('Invalid category')
        else:
            category_list.append('')
    # print(len(anchor_list), 
    #      len(concept1_list), 
    #      len(concept2_list), 
    #      len(category_list), 
    #      len(estimated_total_tokens), 
    #      len(actual_total_tokens), 
    #      len(prompt_list), 
    #      len(full_answer_list))
    result_df = pd.DataFrame({'Anchor':anchor_list, 'Concept1':concept1_list, 'Concept2':concept2_list, 'Category':category_list, 'estimated_tokens':estimated_total_tokens, 'real_tokens': actual_total_tokens, 'prompt':prompt_list, 'gpt_response':full_answer_list, 'gpt_choice':answer_list})
    result_df.to_csv(os.path.join(results_dir, dataset_name, results_dir, dataset_name, model +'_'+ exp_name + '_feature_list.csv'))


def extract_results(exp_name, dataset_name, model, results_dir, temperature):
    if exp_name == 'feature_listing':
        save_feature_listing_results_in_csv(results_dir, dataset_name, model, exp_name, temperature)
    elif exp_name == 'triplet':
        save_feature_triplet_results_in_csv(results_dir, dataset_name, model, exp_name) 
    else:
        logging.error('Cant save result. Only feature listing and triplet task implemented')
        
    return

def main():
    parser = argparse.ArgumentParser(description="""""")
    parser.add_argument('--dataset_name', type=str, 
                        help="""Name of the dataset""")
    parser.add_argument('--exp_name',
                        type=str, help="""specify the experiment you want to run""")
    parser.add_argument('--model',
                    type=str, help=""" Name of the feature listing file""")
    parser.add_argument('--temperature',help = """Tradeoff between deterministic and creative responses of gpt""")
    args = parser.parse_args()
    logging.basicConfig(filename="logs/extract_results_{}_{}.log".format(args.exp_name, args.dataset_name), level=logging.DEBUG, #encoding='utf-8', 
                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    logging.warning('is when this event was logged.')
    logging.info('Running experiments with the following parameters')
    logging.info("########    EXTRACTING RESULTS     ##############")
    for arg, value in sorted(vars(args).items()):
        logging.info("Argument %s: %r", arg, value)

    extract_results(exp_name = args.exp_name, 
            dataset_name = args.dataset_name,  
            model = args.model, 
            results_dir = DEFAULT_RESULTS_DIR, 
            temperature = float(args.temperature))

if __name__=="__main__":
    main()
