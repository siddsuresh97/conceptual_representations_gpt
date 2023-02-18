import os
import pandas as pd
from gpt_interaction import *
from preprocess import *


def run_exp(exp_name, dataset_name, dataset_dir, feature_list_fname, model, openai_api_key, results_dir, temperature, sample):
    if exp_name == 'feature_listing':
        df = pd.read_csv(os.path.join(dataset_dir, dataset_name ,feature_list_fname))
        concepts_set, features_set, concept_feature_matrix = create_and_fill_concept_feature_matrix(df)
        estimated_cost(concepts_set, features_set, concept_feature_matrix, exp_name, dataset_name)
        batches = make_gpt_prompt_batches_feat_listing(concepts_set, features_set, concept_feature_matrix, exp_name)
    elif exp_name == 'triplet':
        file = open(os.path.join(dataset_dir, dataset_name ,feature_list_fname),'rb')
        triplets = pickle.load(file)
        file.close()
        batches = make_gpt_prompt_batches_triplet(triplets)
    elif exp_name == 'leuven_prompts_answers':
        animal_leuven_norms, artifacts_leuven_norms = load_leuven_norms(dataset_dir)
        batches = []
        # batches_animals = []
        # batches_artifacts = []
        # for concept, feature in itertools.product(list(animal_leuven_norms.index), list(animal_leuven_norms.columns)):
        #     batches_animals.append([concept, feature])
        # for concept, feature in itertools.product(list(artifacts_leuven_norms.index), list(artifacts_leuven_norms.columns)):
        #     batches_artifacts.append([concept, feature])
        features = list(set(list(animal_leuven_norms.columns) + list(artifacts_leuven_norms.columns)))
        concepts = list(set(list(animal_leuven_norms.index) + list(artifacts_leuven_norms.index)))
        for concept, feature in itertools.product(concepts[:10], features[:10]):
            batches.append([concept, feature])
        # batches_animals = make_leuven_prompts(batches_animals)
        # batches_artifacts = make_leuven_prompts(batches_artifacts)
        # batches = batches_animals + batches_artifacts
        batches = make_leuven_prompts(batches)
        print('total prompts', len(batches))
    else:
        logging.error('Undefined task. Only feature listing and triplet implemented')
    # print('ESTIMATED TIME in minutes is', len(batches)*4)
    print('Running experiment {} on dataset {} using {} model. Please wait for it to finish'.format(exp_name, dataset_name, model))
    if model != 'flan':
        answer_dict = get_gpt_responses(batches, model, openai_api_key, exp_name, results_dir, dataset_name, temperature)
    else:
        answer_dict = get_transformer_responses(batches, model, exp_name, temperature, sample)
    save_responses(answer_dict, results_dir, dataset_name, exp_name, model, 'full', temperature, sample)
    return