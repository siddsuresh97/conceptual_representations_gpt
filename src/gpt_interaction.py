import numpy as np
import pandas as pd
import torch
import itertools
import warnings
import logging
import openai
import time
from tqdm import tqdm
import pickle
from datasets import Dataset
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm
import os
# from pattern.en import pluralize
from joblib import Parallel, delayed
from accelerate import Accelerator
from transformers import T5Tokenizer, T5ForConditionalGeneration

import inflect
p = inflect.engine()
ERROR = 3
ESTIMATED_RESPONSE_TOKENS = 8 + ERROR
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
    prompt = ' '.join(feature_words)
    characters = len(prompt)
    return prompt, characters




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


## TODO Figure out a way to make sleeping time optimal
def send_gpt_prompt(prompt, model, temperature):
    prompt_start_time = time.time()
    if model == 'ada':
        try:
            response = openai.Completion.create(
                            model="text-ada-001",
                            prompt=prompt,
                            temperature=temperature,
                            max_tokens=256,
                            top_p=1,
                            frequency_penalty=0,
                            presence_penalty=0
                            )
        except:
            logging.info('Sleeping for 30 in ada')
            time.sleep(30)
            response = openai.Completion.create(
                            model="text-ada-001",
                            prompt=prompt,
                            temperature=temperature,
                            max_tokens=256,
                            top_p=1,
                            frequency_penalty=0,
                            presence_penalty=0
                            )
    elif model == 'babbage':
        try:
            response = openai.Completion.create(
                            model="text-babbage-001",
                            prompt=prompt,
                            temperature=temperature,
                            max_tokens=256,
                            top_p=1,
                            frequency_penalty=0,
                            presence_penalty=0
                            )
        except:
            logging.info('Sleeping for 30 in babbage')
            time.sleep(30)
            response = openai.Completion.create(
                            model="text-babbage-001",
                            prompt=prompt,
                            temperature=temperature,
                            max_tokens=256,
                            top_p=1,
                            frequency_penalty=0,
                            presence_penalty=0
                            )
    elif model == 'curie':
        try:
            response = openai.Completion.create(
                            model="text-curie-001",
                            prompt=prompt,
                            temperature=temperature,
                            max_tokens=256,
                            top_p=1,
                            frequency_penalty=0,
                            presence_penalty=0
                            )
        except:
            logging.info('Sleeping for 30 in babbage')
            time.sleep(30)
            response = openai.Completion.create(
                            model="text-curie-001",
                            prompt=prompt,
                            temperature=temperature,
                            max_tokens=256,
                            top_p=1,
                            frequency_penalty=0,
                            presence_penalty=0
                            )
    elif model == 'davinci':
        try:
            response = openai.Completion.create(
                            model="text-davinci-002",
                            prompt=prompt,
                            temperature=temperature,
                            max_tokens=256,
                            top_p=1,
                            frequency_penalty=0,
                            presence_penalty=0
                            )
        except:
            logging.info('Sleeping for 30 in davinci')
            time.sleep(30)
            response = openai.Completion.create(
                            model="text-ada-001",
                            prompt=prompt,
                            temperature=temperature,
                            max_tokens=256,
                            top_p=1,
                            frequency_penalty=0,
                            presence_penalty=0
                            )
    else:
        logging.error('Only ada and davinci implemented')
    each_prompt_time = time.time() - prompt_start_time
    return response, each_prompt_time

def prompt_gpt_feature_listing(concept, feature, prompt, tokens, answer_dict, each_prompt_api_time, model, openai_api_key, temperature):
    openai.api_key = openai_api_key
    response, each_prompt_time = send_gpt_prompt(prompt, model, temperature)
    each_prompt_api_time.append(each_prompt_time)
    answer_dict.update({(concept, feature):(response, tokens, prompt)})

def prompt_gpt_triplet(anchor, concept1, concept2, prompt, tokens, model, each_prompt_api_time, answer_dict,openai_api_key, temperature):
    openai.api_key = openai_api_key
    response, each_prompt_time = send_gpt_prompt(prompt, model, temperature)
    each_prompt_api_time.append(each_prompt_time)
    # print("Estimated total tokens", tokens+ESTIMATED_RESPONSE_TOKENS, '....Real tokens used', response['usage']['total_tokens'])
    answer_dict.update({(anchor, concept1, concept2,):(response, tokens, prompt)})
    return answer_dict



def make_gpt_prompt_batches_feat_listing(concepts_set, features_set, concept_feature_matrix, exp_name):
    logging.info('Making batches')
    batches = []
    batch = []
    total_tokens = 0
    for concept, feature in itertools.product(concepts_set, features_set):
        # was 150000, changed to 100000
        if total_tokens < 100000:
            concept_idx = concepts_set.index(concept)
            feature_idx = features_set.index(feature)
            if concept_feature_matrix[concept_idx, feature_idx] == 0:
                prompt, characters = generate_prompt_feature_listing(concept, feature)
                tokens = np.ceil((characters + 1)/4)
                batch.append([concept, feature, prompt, tokens])
                total_tokens += tokens + (ESTIMATED_RESPONSE_TOKENS)
        else:
            batches.append(batch)
            batch = [[concept, feature, prompt, tokens]]
            total_tokens = tokens + (ESTIMATED_RESPONSE_TOKENS)
            concept_idx = concepts_set.index(concept)
            feature_idx = features_set.index(feature)
            if concept_feature_matrix[concept_idx, feature_idx] == 0:
                prompt, characters = generate_prompt_feature_listing(concept, feature)
                tokens = np.ceil((characters + 1)/4)
                batch.append([concept, feature, prompt, tokens])
                total_tokens += tokens + (ESTIMATED_RESPONSE_TOKENS)
    if len(batch) != 0:
        batches.append(batch)
    logging.info('Total batches of 150000 tokesn are {}'.format(len(batches)))
    return batches

def make_gpt_prompt_batches_grammar_iclr(concepts_set, features_set):
    logging.info('Making batches')
    batch = []
    batches = []
    total_tokens = 0
    for concept, feature in itertools.product(concepts_set, features_set):
        if len(batch)<3000:
            prompt, characters = generate_prompt_to_form_questions_iclr(concept, feature)
            tokens = np.ceil((characters + 1)/4)
            batch.append([concept, feature, prompt, tokens])
        else:
            batches.append(batch)
            batch = [[concept, feature, prompt, tokens]]
    if len(batch) != 0:
        batches.append(batch)
    logging.info('Total batches of 3000 request batches are {}'.format(len(batches)))
    return batches

def generate_prompt_to_form_questions_iclr(concept, feature):
    # prompt = 'Help me form generated a question from [concept],[feature].\n[sweater], [has_two_eyes]\nQ: Do sweaters have two eyes?\n[table],[used_to_make_pancakes]\nQ:Can tables be used to make pancakes?\n[{}],[{}]\nQ:'.format(concept, feature)
    prompt = 'Turn into a question that starts with Do, Are, Can\n {} {}'.format(concept, ' '.join(feature.split('_')))
    characters = len(prompt)
    return prompt, characters

def generate_prompt_triplet(anchor, concept1, concept2):
    # prompt = 'Keywords "{}", "{}"\nQ)Which is more similar to "{}"?\na){}\nb){}'.format(concept1, concept2, anchor, concept1, concept2)
    # prompt = 'Answer using one word "{}" or "{}". Which is more similar in meaning to "{}"?'.format(concept1, concept2, anchor)
    prompt = 'Answer using only only word - "{}" or "{}" and not "{}".Which is more similar in meaning to "{}"?'.format(concept1, concept2, anchor, anchor)
    characters = len(prompt)
    return prompt, characters

## TODO Figure out optimal condition for total_tokens
def make_gpt_prompt_batches_triplet(triplets):
    total_tokens = 0
    batches = []
    batch = []
    for triplet in triplets:
        anchor, concept1, concept2 = triplet
        prompt, characters = generate_prompt_triplet(anchor, concept1, concept2)
        tokens = np.ceil((characters + 1)/4)
        if total_tokens < 100000:
            batch.append([anchor, concept1, concept2, prompt, tokens])
            total_tokens = tokens + (ESTIMATED_RESPONSE_TOKENS)
        else:
            batches.append(batch)
            batch = [[anchor, concept1, concept2, prompt, tokens]]
            total_tokens = tokens + (ESTIMATED_RESPONSE_TOKENS)
    if len(batch) != 0:
        batches.append(batch)
    logging.info('Total batches of 150000 tokesn are {}'.format(len(batches)))
    return batches




## TODO Figure out optimal sleeping time and n_jobs
def get_gpt_responses(batches, model, openai_api_key, exp_name, results_dir, dataset_name, temperature):
    answer_dict = {}
    each_prompt_api_time = []
    start_time = time.time()
    # import ipdb;ipdb.set_trace()
    for i, batch in enumerate(batches):
        if os.path.exists(os.path.join(results_dir, dataset_name, model +'_'+ exp_name + '_{}_{}'.format(i, temperature))):
            print(os.path.join(results_dir, dataset_name, model +'_'+ exp_name + '_{}_{}'.format(i, temperature)), 'EXISTS')
            continue
        if exp_name == 'feature_listing':
            Parallel(n_jobs=10, require='sharedmem')(delayed(prompt_gpt_feature_listing)(concept, feature, prompt, tokens, answer_dict, each_prompt_api_time, model, openai_api_key, temperature) for concept, feature, prompt, tokens in batch)
        elif exp_name == 'triplet':
            Parallel(n_jobs=10, require='sharedmem')(delayed(prompt_gpt_triplet)(anchor, concept1, concept2, prompt, tokens, model, each_prompt_api_time, answer_dict,openai_api_key, temperature) for anchor, concept1, concept2, prompt, tokens in batch)
        save_responses(answer_dict, results_dir, dataset_name, exp_name, model, i, temperature, sample=False)
        if len(batches) > 1:
            time.sleep(60*2)
    exp_run_time = time.time()- start_time
    logging.info('It took {}s to run the experiment'.format(exp_run_time))
    logging.info('Each api request took {}s'.format(np.mean(each_prompt_api_time)))
    logging.info('Total time in running api concurrently is {}s'.format(np.sum(each_prompt_api_time)))
    logging.info('Speedup by parallilsation was {} x'.format((np.sum(each_prompt_api_time)- exp_run_time)/exp_run_time))
    return answer_dict

def get_gpt_responses_grammar(batches, model, openai_api_key, exp_name, results_dir, dataset_name, temperature):
    answer_dict = {}
    each_prompt_api_time = []
    start_time = time.time()
    for i, batch in enumerate(batches):
        if os.path.exists(os.path.join(results_dir, dataset_name, model +'_'+ exp_name + '_{}_{}'.format(i, temperature))):
            print(os.path.join(results_dir, dataset_name, model +'_'+ exp_name + '_{}_{}'.format(i, temperature)), 'EXISTS')
            continue
        Parallel(n_jobs=10, require='sharedmem')(delayed(prompt_gpt_feature_listing)(concept, feature, prompt, tokens, answer_dict, each_prompt_api_time, model, openai_api_key, temperature) for concept, feature, prompt, tokens in batch)
        save_responses(answer_dict, results_dir, dataset_name, exp_name, model, i, temperature, sample=False)
        if len(batches) > 1:
            time.sleep(60)
    

def get_transformer_responses(batches, model, exp_name, temperature, sample):
    answer_dict = {}
    each_prompt_api_time = []
    start_time = time.time()
    batches = np.array(list(itertools.chain(*batches)))
    # batches = batches[:10]
    if model == 'flan':
        responses = []
        batch_size = 256
        if exp_name == 'feature_listing':
            concepts = batches[:,0]
            features = batches[:,1]
            prompts = batches[:,2]
            tokens = batches[:,3]
            start_time = time.time()
            tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
            prompt_dict = {'prompt':prompts.tolist()}
            ds = Dataset.from_dict(prompt_dict)
            ds = ds.map(lambda examples: T5Tokenizer.from_pretrained("google/flan-t5-xxl")(examples['prompt'], max_length=40, truncation=True, padding='max_length'), batched=True)
            ds.set_format(type='torch', columns=['input_ids', 'attention_mask'])
            dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size)
            # import ipdb;ipdb.set_trace()
            flan_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", device_map="auto",  torch_dtype=torch.bfloat16)
            preds = []
            for batch in dataloader:
                input_ids = batch['input_ids'].to('cuda')
                attention_mask = batch['attention_mask'].to('cuda')
                outputs = flan_model.generate(input_ids, attention_mask=attention_mask, temperature = temperature)
                preds.extend(outputs)
            print('Time taken to generate responses is {}s'.format(time.time()-start_time))
            responses = tokenizer.batch_decode(preds, skip_special_tokens=True)
            del model
            for concept, feature, response , prompt in zip(concepts, features, responses, prompts):
                answer_dict.update({(concept, feature):(response, tokens, prompt)})
        elif exp_name == 'triplet':
            anchor = batches[:,0]
            concept1 = batches[:,1]
            concept2 = batches[:,2]
            prompts = batches[:,3]
            tokens = batches[:,4]
            start_time = time.time()
            tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
            prompt_dict = {'prompt':prompts.tolist()}
            ds = Dataset.from_dict(prompt_dict)
            ds = ds.map(lambda examples: T5Tokenizer.from_pretrained("google/flan-t5-xxl")(examples['prompt'], max_length=40, truncation=True, padding='max_length'), batched=True)
            ds.set_format(type='torch', columns=['input_ids', 'attention_mask'])
            dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size)
            flan_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", device_map="auto",  torch_dtype=torch.bfloat16,  cache_dir="/data")
            preds = []
            for batch in dataloader:
                input_ids = batch['input_ids'].to('cuda')
                attention_mask = batch['attention_mask'].to('cuda')
                outputs = flan_model.generate(input_ids, attention_mask=attention_mask, temperature = temperature)
                preds.extend(outputs)
            print('Time taken to generate responses is {}s'.format(time.time()-start_time))
            responses = tokenizer.batch_decode(preds, skip_special_tokens=True)
            del flan_model
            # import ipdb;ipdb.set_trace()
            for anchor, concept1, concept2, response , prompt in zip(anchor, concept1, concept2, responses, prompts):
                answer_dict.update({(anchor, concept1, concept2,):(response, tokens, prompt)})
        elif exp_name == 'leuven_prompts_answers':
            accelerator = Accelerator()
            device = accelerator.device
            batch_size = 200
            concepts = batches[:,0]
            features = batches[:,1]
            prompts = batches[:,2]
            tokens = batches[:,3]
            start_time = time.time()
            tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
            tokenizer = accelerator.prepare(
                T5Tokenizer.from_pretrained("google/flan-t5-xxl")
            )
            prompt_dict = {'prompt':prompts.tolist()}
            ds = Dataset.from_dict(prompt_dict)
            # ds = ds.map(lambda examples: T5Tokenizer.from_pretrained("google/flan-t5-xxl")(examples['prompt'],truncation=True, padding='max_length'), batched=True)
            ds = ds.map(lambda examples: tokenizer(examples['prompt'],truncation=True, padding='max_length'), batched=True)
            ds.set_format(type='torch', columns=['input_ids', 'attention_mask'])
            dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=10, drop_last=False, pin_memory=True)
            flan_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", return_dict=True,  torch_dtype=torch.bfloat16)
            flan_model = flan_model.to(device)
            # flan_model, dataloader = accelerator.prepare(T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", device_map="balanced_low_0", torch_dtype=torch.bfloat16,  cache_dir="/data"), dataloader)
            preds = []
            flan_model.eval()
            with torch.no_grad():
                for batch in tqdm(dataloader):
                    input_ids = batch['input_ids'].to(device) #.to(device=accelerator.device)
                    attention_mask = batch['attention_mask'].to(device) #.to(device=accelerator.device)
                    if sample == "False":
                        outputs = flan_model.generate(input_ids, attention_mask=attention_mask, temperature = temperature)
                        preds.extend(outputs)
                    else:
                        outputs = model.generate(
                            input_ids,
                            do_sample=True, 
                            max_length=50, 
                            top_k=10, 
                            top_p=0.95, 
                            num_return_sequences=20
                        )
            print('Time taken to generate responses is {}s'.format(time.time()-start_time))
            decode_start_time = time.time()
            del flan_model
            if sample == "False":
                responses = tokenizer.batch_decode(preds, skip_special_tokens=True)
                print('decoding done', time.time()-decode_start_time)
                answer_dict = {'concept':concepts, 'feature':features, 'prompt':prompts, 'response':responses}
            else:
                responses = [tokenizer.decode(sample_output, skip_special_tokens=True) for sample_output in responses]
            print('decoding done', time.time()-decode_start_time)
            answer_dict = {'concept':concepts, 'feature':features, 'prompt':prompts, 'response':responses, 'tokens':tokens}

        # torch distributed
        # elif exp_name == 'leuven_prompts_answers':
        #     batch_size = 32
        #     concepts = batches[:,0]
        #     features = batches[:,1]
        #     prompts = batches[:,2]
        #     tokens = batches[:,3]
        #     start_time = time.time()
        #     tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
        #     prompt_dict = {'prompt':prompts.tolist()}
        #     ds = Dataset.from_dict(prompt_dict)
        #     ds = ds.map(lambda examples: T5Tokenizer.from_pretrained("google/flan-t5-xxl")(examples['prompt'],truncation=True, padding='max_length'), batched=True)
        #     ds.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        #     dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size,  pin_memory=True, num_workers=10, drop_last=False)
        #     flan_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", device_map="auto",  torch_dtype=torch.bfloat16,  cache_dir="/data")
        #     flan_model = flan_model.to(device)
        #     preds = []
        #     flan_model.eval()
        #     with torch.no_grad():
        #         for batch in tqdm(dataloader):
        #             input_ids = batch['input_ids'].to(device)
        #             attention_mask = batch['attention_mask'].to(device)
        #             outputs = flan_model.generate(input_ids, attention_mask=attention_mask, temperature = temperature)
        #             preds.extend(outputs)
        #     print('Time taken to generate responses is {}s'.format(time.time()-start_time))
        #     decode_start_time = time.time()
        #     responses = tokenizer.batch_decode(preds, skip_special_tokens=True)
        #     print('decoding done', time.time()-decode_start_time)
        #     del flan_model
        #     answer_dict = {'concept':concepts, 'feature':features, 'prompt':prompts, 'response':responses}
        # elif exp_name == 'leuven_prompts_answers':
        #     batch_size = 32
        #     concepts = batches[:,0]
        #     features = batches[:,1]
        #     prompts = batches[:,2]
        #     tokens = batches[:,3]
        #     start_time = time.time()
        #     tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
        #     prompt_dict = {'prompt':prompts.tolist()}
        #     ds = Dataset.from_dict(prompt_dict)
        #     ds = ds.map(lambda examples: T5Tokenizer.from_pretrained("google/flan-t5-xxl")(examples['prompt'],truncation=True, padding='max_length'), batched=True)
        #     ds.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        #     dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size,  pin_memory=True, num_workers=10, drop_last=False)
        #     flan_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", device_map="auto",  torch_dtype=torch.bfloat16)
        #     flan_model = flan_model.to('cuda')
        #     preds = []
        #     flan_model.eval()
        #     with torch.no_grad():
        #         for batch in tqdm(dataloader):
        #             input_ids = batch['input_ids'].to('cuda')
        #             attention_mask = batch['attention_mask'].to('cuda')
        #             outputs = flan_model.generate(input_ids, attention_mask=attention_mask, temperature = temperature)
        #             preds.extend(outputs)
        #     print('Time taken to generate responses is {}s'.format(time.time()-start_time))
        #     decode_start_time = time.time()
        #     responses = tokenizer.batch_decode(preds, skip_special_tokens=True)
        #     print('decoding done', time.time()-decode_start_time)
        #     del flan_model
        #     answer_dict = {'concept':concepts, 'feature':features, 'prompt':prompts, 'response':responses}
    return answer_dict

def save_responses(answer_dict, results_dir, dataset_name, exp_name, model, part, temperature, sample):
    if exp_name == 'leuven_prompts_answers':
        #make a df from the answer dict
        if sample:
            answers = [x.replace("True", "Yes").replace("False", "No") for x in answer_dict['response']]
            answers = [answers[i:i+20] for i in range(0, len(answers), 20)]
            answers = [[[x, answers[i].count(x)] for x in set(answers[i])]for i in range(len(answers))]
            answers = [[sorted(x, key=lambda x: x[1], reverse=True)] for x in answers]
            winner = [x[0][0][0] for x in answers]
            confidence = [x[0][0][1]/20 for x in answers]
            answer_dict['response'] = winner
            answer_dict['confidence'] = confidence
        df = pd.DataFrame.from_dict(answer_dict)
        # TODO - change save dir path
        df.to_csv(os.path.join(results_dir, model +'_'+ exp_name + '.csv'))
    else:
        if not os.path.exists(os.path.join(results_dir, dataset_name)):
            os.mkdir(os.path.join(results_dir, dataset_name))
        else:
            with open(os.path.join(results_dir, dataset_name, model +'_'+ exp_name + '_{}_temperature_{}'.format(part, temperature)), 'wb') as handle:
                pickle.dump(answer_dict,handle ,  protocol=pickle.HIGHEST_PROTOCOL)
