import pandas as pd
import os
from joblib import Parallel, delayed

def preprocess_leuven_norms(leuven_dir, save_dir):
    '''This function reads the Leuven Norms, selecsts relevant concepts data and saves the dataframes as csv files.'''
    animal_leuven_norms = pd.read_csv(os.path.join(leuven_dir,'ANIMALSexemplarfeaturesbig.txt'), sep = "\t")
    artiacts_leuven_norms = pd.read_csv(os.path.join(leuven_dir,'ARTIFACTSexemplarfeaturesbig.txt'), sep = "\t", encoding='latin-1')
     # in animal norms, the first column is the word, the second column is the frequency, multiply all cells by the freqeuncy 
    animal_leuven_norms.iloc[:,2:] = animal_leuven_norms.iloc[:,2:].multiply(animal_leuven_norms.iloc[:,1], axis=0)
    animal_leuven_norms.iloc[:,2:] = animal_leuven_norms.iloc[:,2:].div(animal_leuven_norms.iloc[:,1].sum(), axis=0)

    artiacts_leuven_norms.iloc[:,2:] = artiacts_leuven_norms.iloc[:,2:].multiply(artiacts_leuven_norms.iloc[:,1], axis=0)
    artiacts_leuven_norms.iloc[:,2:] = artiacts_leuven_norms.iloc[:,2:].div(artiacts_leuven_norms.iloc[:,1].sum(), axis=0)

    # from the third column onwards, replace the values grater than 0 with 1 and 0 otherwise
    animal_leuven_norms.iloc[:,2:] = animal_leuven_norms.iloc[:,2:].applymap(lambda x: 1 if x > 0 else 0)
    artiacts_leuven_norms.iloc[:,2:] = artiacts_leuven_norms.iloc[:,2:].applymap(lambda x: 1 if x > 0 else 0)

    # select the first column and select columns that contain the animal names in the list animals and also the first column
    animal_leuven_norms = animal_leuven_norms[['feature/_exemplar_ENGLISH'] + [col for col in animal_leuven_norms.columns]]
    artiacts_leuven_norms = artiacts_leuven_norms[['Item'] + [col for col in artiacts_leuven_norms.columns]]

    # rename the first column to 'features'
    animal_leuven_norms.rename(columns={'feature/_exemplar_ENGLISH': 'features'}, inplace=True)
    artiacts_leuven_norms.rename(columns={'Item': 'features'}, inplace=True)

    # transpose the dataframe so that the features are in the rows and the animals are in the columns
    animal_leuven_norms = animal_leuven_norms.T
    artiacts_leuven_norms = artiacts_leuven_norms.T


    # let the first row be the column names and drop the first row
    animal_leuven_norms.columns = animal_leuven_norms.iloc[0]
    animal_leuven_norms = animal_leuven_norms.drop(animal_leuven_norms.index[0])
    animal_leuven_norms = animal_leuven_norms.drop(animal_leuven_norms.index[0])


    artiacts_leuven_norms.columns = artiacts_leuven_norms.iloc[0]
    artiacts_leuven_norms = artiacts_leuven_norms.drop(artiacts_leuven_norms.index[0])
    artiacts_leuven_norms = artiacts_leuven_norms.drop(artiacts_leuven_norms.index[0])

    # save the dataframes as csv files
    animal_leuven_norms.to_csv(os.path.join(save_dir,'animal_leuven_norms.csv'))
    artiacts_leuven_norms.to_csv(os.path.join(save_dir,'artifacts_leuven_norms.csv'))
    return 


def load_leuven_norms(save_dir):
    '''This function loads the preprocessed Leuven Norms dataframes.'''
    animal_leuven_norms = pd.read_csv(os.path.join(save_dir,'animal_leuven_norms.csv'), index_col=0)
    artiacts_leuven_norms = pd.read_csv(os.path.join(save_dir,'artifacts_leuven_norms.csv'), index_col=0)
    return animal_leuven_norms, artiacts_leuven_norms

def add_leuven_prompt(concept, feature, batches_with_prompts):
    '''This function adds the prompts to the batches.'''
    # prompt = "Help me write a prompt as a question from a concept and an attribute. \nConcept: {}\nAttribute: {}.\nPrompt: In one word Yes/No <mask> ?".format(concept, feature)
    prompt = "Q: Is the property [is_female] true for the concept [book]?\nA: False\nQ: Is the property [can_be_digital] true for the concept [book]?\nA: True\nIn one word True/False answer the following question:\nQ: : Is the property [{}] true for the concept [{}]?\nA: <mask>".format(concept, feature)
    batches_with_prompts.append([[concept, feature, prompt, 0]])
    return

def make_leuven_prompts(batches):
    '''This function creates the prompts for the Leuven Norms experiment.'''
    batches_with_prompts = []
    Parallel(n_jobs=10, require='sharedmem')(delayed(add_leuven_prompt)(batch[0], batch[1], batches_with_prompts) for batch in batches)
    return batches_with_prompts