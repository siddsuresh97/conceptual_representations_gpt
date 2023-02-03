import os
import pandas as pd
from gpt_interaction import *

def run_exp(exp_name, dataset_name, dataset_dir, feature_list_fname, model, openai_api_key, results_dir, temperature):
    if exp_name == 'feature_listing':
        df = pd.read_csv(os.path.join(dataset_dir, dataset_name ,feature_list_fname))
        concepts_set, features_set, concept_feature_matrix = create_and_fill_concept_feature_matrix(df)
        estimated_cost(concepts_set, features_set, concept_feature_matrix, exp_name, dataset_name)
        if model != 'flan':
            batches = make_gpt_prompt_batches_feat_listing(concepts_set, features_set, concept_feature_matrix, exp_name)
        else:
            batches = make_flan_prompt_batches_feat_listing(concepts_set, features_set, concept_feature_matrix, exp_name)
    elif exp_name == 'triplet':
        file = open(os.path.join(dataset_dir, dataset_name ,feature_list_fname),'rb')
        triplets = pickle.load(file)
        file.close()
        batches = make_gpt_prompt_batches_triplet(triplets)
    else:
        logging.error('Undefined task. Only feature listing and triplet implemented')
    print('ESTIMATED TIME in minutes is', len(batches)*4)
    print('Running experiment {} on dataset {} using {} model. Please wait for it to finish'.format(exp_name, dataset_name, model))
    answer_dict = get_gpt_responses(batches, model, openai_api_key, exp_name, results_dir, dataset_name, temperature)   
    save_responses(answer_dict, results_dir, dataset_name, exp_name, model, 'full', temperature)
    return 