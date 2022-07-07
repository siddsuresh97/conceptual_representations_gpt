import numpy as np
import itertools
import logging
import openai
import time
import pickle
import os
from pattern.en import pluralize
from joblib import Parallel, delayed

import inflect
p = inflect.engine()
ESTIMATED_RESPONSE_TOKENS = 8
PROMPT_TOKEN_INFLATION = 1.25

def generate_prompt_feature_listing(concept, feature):
    feature_words = feature.split('_')
    verb = feature_words[0] 
    if verb in ['are', 'can']:
        if concept == 'Caiman':
            feature_words.insert(1, 'caimans')
        else:    
            feature_words.insert(1, pluralize(concept).lower())
        feature_words[0] = feature_words[0].capitalize()
        feature_words.append('?')

    elif verb in ['eats', 'lay']:
        feature_words.insert(0, 'Do')
        if concept == 'Caiman':
            feature_words.insert(1, 'caimans')
        else:    
            feature_words.insert(1, pluralize(concept).lower())
        if verb == 'eats':
            feature_words.remove('eats')
            feature_words.insert(2, 'eat')
        feature_words.append('?')
    
    elif verb in ['have', 'live', 'lives']:
        feature_words.insert(0, 'Do')
        if concept == 'Caiman':
            feature_words.insert(1, 'caimans')
        else:    
            feature_words.insert(1, pluralize(concept).lower())
        # checking if the noun is singular or not. REturns false if singular
        if verb == 'lives':
            feature_words.remove('lives') 
            feature_words.insert(2, 'live')    
        if verb == 'have' and not(p.singular_noun(feature_words[-1])):
            feature_words.insert(3, 'a')
        feature_words.append('?')
    
    elif verb == 'made':
        feature_words.insert(0, 'Are')
        if concept == 'Caiman':
            feature_words.insert(1, 'caimans')
        else:    
            feature_words.insert(1, pluralize(concept).lower())
        feature_words.append('?')
    else:
        print(feature)
        print('Verb structure not implemented')
    feature_words.insert(0, 'In one word, Yes/No:')
    return ' '.join(feature_words), 5 + len(feature_words)


def get_unique_concepts_and_features(concepts, features):
    concepts_set = list(set(concepts))
    features_set = list(set(list(features)))
    concepts_set.sort()
    features_set.sort()
    return concepts_set, features_set


def create_and_fill_concept_feature_matrix(df):
    concepts = list(df['Concept']) 
    features = list(df['Feature'])
    concepts_set, features_set = get_unique_concepts_and_features(concepts, features) 
    n_concepts = len(concepts_set)
    n_features = len(features_set)
    concept_feature_matrix = np.zeros((n_concepts, n_features))
    for concept, feature in zip(concepts, features):
        concept_idx = concepts_set.index(concept)
        feature_idx = features_set.index(feature)
        concept_feature_matrix[concept_idx, feature_idx] = 1
    return concepts_set, features_set, concept_feature_matrix 


def estimated_cost(concepts_set, features_set, concept_feature_matrix, exp_name, dataset_name):
    tokens = 0
    queries = 0
    for concept, feature in itertools.product(concepts_set, features_set):
        concept_idx = concepts_set.index(concept)
        feature_idx = features_set.index(feature)
        if concept_feature_matrix[concept_idx, feature_idx] == 0:
            _, words = generate_prompt_feature_listing(concept, feature)
            tokens += np.ceil((words + 1)/0.75)*PROMPT_TOKEN_INFLATION + ESTIMATED_RESPONSE_TOKENS
            queries += 1
    logging.info('Estimated cost of running {} on {} experiment is {}'.format(exp_name, dataset_name, tokens/1000*0.06))    
    logging.info('Total queries to be made are {}'.format(queries))


def prompt_gpt(concept, feature, prompt, tokens, answer_dict, each_prompt_api_time, model, openai_api_key):
    openai.api_key = openai_api_key
    prompt_start_time = time.time()
    if model == 'ada':
        response = openai.Completion.create(
                        model="text-ada-001",
                        prompt=prompt,
                        temperature=0.7,
                        max_tokens=256,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0
                        )
    elif model == 'davinci':
        response = openai.Completion.create(
                        model="text-davinci-002",
                        prompt=prompt,
                        temperature=0.7,
                        max_tokens=256,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0
                        )
    else:
        logging.error('Only ada and davinci implemented')
    each_prompt_api_time.append(time.time() - prompt_start_time)
    answer_dict.update({(concept, feature):(response, tokens, prompt)})


def make_gpt_prompt_batches(concepts_set, features_set, concept_feature_matrix, exp_name):
    logging.info('Making batches')
    batches = []
    batch = []
    total_tokens = 0
    for concept, feature in itertools.product(concepts_set, features_set):
        if total_tokens < 150000:
            concept_idx = concepts_set.index(concept)
            feature_idx = features_set.index(feature)
            if concept_feature_matrix[concept_idx, feature_idx] == 0:
                if exp_name == 'feature_listing':
                    prompt, words = generate_prompt_feature_listing(concept, feature)
                    tokens = np.ceil((words + 1)/0.75)*PROMPT_TOKEN_INFLATION
                elif exp_name == 'triplet_matching':
                    logging.error('Yet to implement triplet matching')
                else:
                    logging.error('Only feature listing and triplet matching implemented')
                total_tokens += tokens + (ESTIMATED_RESPONSE_TOKENS)
                batch.append([concept, feature, prompt, tokens])
        else:
            batches.append(batch)
            concept_idx = concepts_set.index(concept)
            feature_idx = features_set.index(feature)
            if concept_feature_matrix[concept_idx, feature_idx] == 0:
                if exp_name == 'feature_listing':
                    prompt, words = generate_prompt_feature_listing(concept, feature)
                    tokens = np.ceil((words + 1)/0.75)*PROMPT_TOKEN_INFLATION
                elif exp_name == 'triplet_matching':
                    logging.error('Yet to implement triplet matching')
                else:
                    logging.error('Only feature listing and triplet matching implemented')
                batch = [[concept, feature, prompt, tokens]]
                total_tokens = tokens + (ESTIMATED_RESPONSE_TOKENS)
    batches.append(batch)
    logging.info('Total batches of 245000 tokesn are {}'.format(len(batches)))
    return batches
                

def get_gpt_responses(batches, model, openai_api_key):
    answer_dict = {}
    each_prompt_api_time = []
    start_time = time.time()
    for batch in batches:
        Parallel(n_jobs=10, require='sharedmem')(delayed(prompt_gpt)(concept, feature, prompt, tokens, answer_dict, each_prompt_api_time, model, openai_api_key) for concept, feature, prompt, tokens in batch)
        time.sleep(60*2)
    exp_run_time = time.time()- start_time
    logging.info('It took {}s to run the experiment'.format(exp_run_time))
    logging.info('Each api request took {}s'.format(np.mean(each_prompt_api_time)))
    logging.info('Total time in running api concurrently is {}s'.format(np.sum(each_prompt_api_time)))
    logging.info('Speedup by parallilsation was {} x'.format((np.sum(each_prompt_api_time)- exp_run_time)/exp_run_time))
    return answer_dict


def save_responses(answer_dict, results_dir, dataset_name, exp_name, model):
    if not os.path.exists(os.path.join(results_dir, dataset_name)):
        os.mkdir(os.path.join(results_dir, dataset_name))
    with open(os.path.join(results_dir, dataset_name, model +'_'+ exp_name), 'wb') as handle:
        pickle.dump(answer_dict,handle ,  protocol=pickle.HIGHEST_PROTOCOL) 
