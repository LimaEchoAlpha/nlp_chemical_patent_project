
import os
import io
import re
import sys

import numpy as np
from time import time

import tensorflow as tf
from transformers import BertTokenizer, TFBertModel


def generate_entity_start_mask(snippetTokens, max_length, start1, start2):
    """
    Helper function that generates a mask 
    that picks out the start marker for each entity 
    given a list of snippet tokens
    """
    
    e1_mask = np.zeros(shape=(max_length,), dtype=bool)
    e1_mask[np.argwhere(np.array(snippetTokens) == start1)] = True

    e2_mask = np.zeros(shape=(max_length,), dtype=bool)
    e2_mask[np.argwhere(np.array(snippetTokens) == start2)] = True

    return e1_mask, e2_mask


def generate_entity_mention_mask(snippetTokens, max_length, start1, start2):
    """
    Helper function that generates a mask
    that picks out the tokens for each entity
    between (but not including) the entity markers
    """
    
    em_markers = [start1, '[/E1]', start2, '[/E2]']
    
    e1_mask = np.zeros(shape=(max_length,), dtype=bool)
    e2_mask = np.zeros(shape=(max_length,), dtype=bool)
    in_e1 = False
    in_e2 = False
    
    for (i, t) in enumerate(snippetTokens):
        if t in em_markers:
            if t in [start1, '[/E1]']:
                in_e1 = not in_e1
            elif t in [start2, '[/E2]']:
                in_e2 = not in_e2
        else:
            if in_e1 is True:
                e1_mask[i] = True
            elif in_e2 is True:
                e2_mask[i] = True
                
    return e1_mask, e2_mask


def generate_ner_mention_mask(snippetTokens, max_length, start1, start2):
    """
    Helper function that generates a mask
    that picks out the tokens for each entity
    between the entity markers, including the ner marker
    """
    
    em_markers = [start1, '[/E1]', start2, '[/E2]']
    
    e1_mask = np.zeros(shape=(max_length,), dtype=bool)
    e2_mask = np.zeros(shape=(max_length,), dtype=bool)
    in_e1 = False
    in_e2 = False
    
    for (i, t) in enumerate(snippetTokens):
        if t in em_markers:
            if t in [start1, '[/E1]']:
                in_e1 = not in_e1
            elif t in [start2, '[/E2]']:
                in_e2 = not in_e2
        else:
            if in_e1 is True:
                e1_mask[i] = True
            elif in_e2 is True:
                e2_mask[i] = True
    
    x1 = snippetTokens.index(start1)
    e1_mask[x1] = True
    
    x2 = snippetTokens.index(start2)
    e2_mask[x2] = True
                
    return e1_mask, e2_mask



def generate_entity_inputs(full_path, tokenizer, marker_type, head_type, max_length=500):
    """
    Reads preprocessed chemical patent data for relation extraction line by line and
    constructs arrays for input into BERT models. Also keeps track of snippet lengths
    for EDA and IDs of any discarded entries.
    
    Each snippet is capped at max_length: snippets that are shorter are padded, and
    snippets that are longer are truncated. All snippets end with a [SEP] token, padded or not.
    Truncated snippets that only contain one entry are discarded.
    
    Inputs:
    full_path = full path of chemical patent data file
    tokenizer = loaded tokenizer to be used
    marker_type = denotes whether the file uses entity markers ('em')
                  or ner markers ('ner')
    head_type = denotes the type of fixed length representation
                the inputs are intended for: 
                'cls' = no entity masks 
                'start' = entity masks to pick out start tokens for each entity
                'pool' = entity masks to pick out entity tokens between markers
                'ner' = entity masks to pick out entity tokens plus ner marker
    max_length = max length for capping snippets
    
    Outputs:
    all_lists = [bert_inputs, bert_labels, extras]
    bert_inputs: array of arrays for input into BERT model
                 includes tokenIDs, bert masks, sequence IDs, and entity masks
    bert_labels: array of labels (need to be one hot encoded before use in model)
    extras: list of lists with extra information 
            includes original labels, snippet lengths, and discarded entries    
    """
    
    # lists for BERT input
    bertTokenIDs = []
    bertMasks = []
    bertSeqIDs = []
    
    # list for labels
    origLabels = []
    codedLabels = []

    # lists for entity masks
    entity1Masks = []
    entity2Masks = []
    
    # lists for processing
    snippetLengthList = []
    discardedEntries = []
    
    # dictionary for converting labels to code
    code = {'ARG1': 0, 'ARGM': 1}

    # determine which marker list to use
    if marker_type == 'em':
        markers = ['[E1]', '[/E1]', '[E2]', '[/E2]']
    elif marker_type == 'ner':
        markers = ['Α', 'Β', 'Π', 'Σ', 'Ο', 'Τ', 'Θ', 'Ψ', 'Υ', 'Χ', 'Λ', 'Δ', '[/E1]', '[/E2]']
        
    
    # open file and read lines of text
    # each line is an entry
    with io.open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.readlines()

    for line in text:

        parsed_line = line.strip().split('\t')

        snippet_id = parsed_line[0]
        label = parsed_line[1]
        snippet = parsed_line[2].split()

        # tokenize snippet, except for entity markers
        # identify start markers for each entity
        snippetTokens = ['[CLS]']
        start1 = ''
        start2 = ''
        i = 1

        for word in snippet:
            if word not in markers:
                tokens = tokenizer.tokenize(word)
                snippetTokens.extend(tokens)
            else:
                snippetTokens.append(word)
                if i == 1:
                    start1 = word
                if i == 3:
                    start2 = word
                i += 1

        # check that both entities will make it within max_length
        # by finding the index for [/E2] and comparing it to (max_length - 1)
        check = snippetTokens.index('[/E2]')

        # discard if only one entity will make it
        if check >= (max_length - 1):
            discardedEntries.append(snippet_id)
            continue

        # create space for at least a final [SEP] token
        if len(snippetTokens) >= max_length:
            snippetTokens = snippetTokens[:(max_length - 1)]
        
        # figure out snippet length for padding or truncating
        snippetLength = len(snippetTokens) + 1
        snippetLengthList.append(snippetLength - 2)

        # add [SEP] token and padding
        snippetTokens += ['[SEP]'] + ['[PAD]'] * (max_length - snippetLength)

        # generate BERT input lists
        bertTokenIDs.append(tokenizer.convert_tokens_to_ids(snippetTokens))
        bertMasks.append(([1] * snippetLength) + ([0] * (max_length - snippetLength)))
        bertSeqIDs.append([0] * (max_length))

        # generate label lists
        origLabels.append(label)
        codedLabels.append(code[label])
        
        # generate entity masks
        if head_type == 'start':
            e1_mask, e2_mask = generate_entity_start_mask(snippetTokens, max_length, start1, start2)
            entity1Masks.append(e1_mask)
            entity2Masks.append(e2_mask)
        
        elif head_type == 'pool':
            e1_mask, e2_mask = generate_entity_mention_mask(snippetTokens, max_length, start1, start2)
            entity1Masks.append(e1_mask)
            entity2Masks.append(e2_mask)
        
        elif head_type == 'ner':
            e1_mask, e2_mask = generate_ner_mention_mask(snippetTokens, max_length, start1, start2)
            entity1Masks.append(e1_mask)
            entity2Masks.append(e2_mask)

    # convert bert inputs to np arrays for modeling
    bert_inputs = [np.array(bertTokenIDs), np.array(bertMasks), np.array(bertSeqIDs), 
                   np.array(entity1Masks), np.array(entity2Masks)]
    
    # convert labels to one hot encoded for modeling
    codedLabels_array = np.array(codedLabels)
    #bert_labels = tf.one_hot(codedLabels_array, depth=2)
    
    # collect everything
    extras = [origLabels, snippetLengthList, discardedEntries]
    all_lists = [bert_inputs, codedLabels_array, extras]
    
    return all_lists



def generate_standard_inputs(full_path, tokenizer, max_length=500):
    """
    Reads preprocessed chemical patent data for relation extraction line by line and
    constructs arrays for standard input (i.e., no entity markers) into BERT models. 
    Also keeps track of snippet lengths for EDA and IDs of any discarded entries.
    
    Each snippet is capped at max_length: snippets that are shorter are padded, and
    snippets that are longer are truncated. All snippets end with a [SEP] token, padded or not.
    Truncated snippets that only contain one entry are discarded.
    
    Inputs:
    full_path = full path of chemical patent data file
    tokenizer = loaded tokenizer to be used
    max_length = max length for capping snippets
    
    Outputs:
    all_lists = [bert_inputs, bert_labels, extras]
    bert_inputs: list of numpy arrays for inputs into BERT model
                 includes tokenIDs, bert masks, sequence IDs, and entity masks
    bert_labels: array of labels (need to be one hot encoded before use in model)
    extras: list of lists with extra information 
            includes original labels, snippet lengths, and discarded entries    
    """
    
    # lists for BERT input
    bertTokenIDs = []
    bertMasks = []
    bertSeqIDs = []
    
    # list for labels
    origLabels = []
    codedLabels = []

    # lists for entity masks
    entity1Masks = []
    entity2Masks = []
    
    # lists for processing
    snippetLengthList = []
    discardedEntries = []
    
    # dictionary for converting labels to code
    code = {'ARG1': 0, 'ARGM': 1}

    # determine which marker list to use
    markers = ['[E1]', '[/E1]', '[E2]', '[/E2]']
        
    
    # open file and read lines of text
    # each line is an entry
    with io.open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.readlines()

    for line in text:

        parsed_line = line.strip().split('\t')

        snippet_id = parsed_line[0]
        label = parsed_line[1]
        snippet = parsed_line[2].split()

        # tokenize snippet, remove entity markers
        # collect masking information for each entity
        snippetTokens = ['[CLS]']
        entity1 = []
        e1 = 0
        entity2 = []
        e2 = 0
        i = 1        

        for word in snippet:
            if word not in markers:
                tokens = tokenizer.tokenize(word)
                snippetTokens.extend(tokens)
                entity1.extend([e1]*len(tokens))
                entity2.extend([e2]*len(tokens))
                i += len(tokens)
            else:
                if word == '[E1]':
                    e1 = 1
                elif word == '[/E1]':
                    e1 = 0
                elif word == '[E2]':
                    e2 = 1
                elif word == '[/E2]':
                    e2 = 0
                    # check for whether both entities
                    # will make it within max length
                    check = i - 1

        # discard if only one entity will make it
        if check >= (max_length - 1):
            discardedEntries.append(snippet_id)
            continue

        # create space for at least a final [SEP] token
        if len(snippetTokens) >= max_length:
            snippetTokens = snippetTokens[:(max_length - 1)]
        
        # figure out snippet length for padding or truncating
        snippetLength = len(snippetTokens) + 1
        snippetLengthList.append(snippetLength - 2)

        # add [SEP] token and padding
        snippetTokens += ['[SEP]'] + ['[PAD]'] * (max_length - snippetLength)

        # generate BERT input lists
        bertTokenIDs.append(tokenizer.convert_tokens_to_ids(snippetTokens))
        bertMasks.append(([1] * snippetLength) + ([0] * (max_length - snippetLength)))
        bertSeqIDs.append([0] * (max_length))

        # generate label lists
        origLabels.append(label)
        codedLabels.append(code[label])
        
        # generate entity masks
        e1_mask = np.zeros(shape=(max_length,), dtype=bool)
        e1_mask[np.argwhere(np.array(entity1) == 1)] = True

        e2_mask = np.zeros(shape=(max_length,), dtype=bool)
        e2_mask[np.argwhere(np.array(entity2) == 1)] = True
        
        entity1Masks.append(e1_mask)
        entity2Masks.append(e2_mask)
        
        
    # convert bert inputs to np arrays for modeling
    bert_inputs = [np.array(bertTokenIDs), np.array(bertMasks), np.array(bertSeqIDs), 
                   np.array(entity1Masks), np.array(entity2Masks)]
    
    # convert labels to one hot encoded for modeling
    codedLabels_array = np.array(codedLabels)
    #bert_labels = tf.one_hot(codedLabels_array, depth=2)
    
    # collect everything
    extras = [origLabels, snippetLengthList, discardedEntries]
    all_lists = [bert_inputs, codedLabels_array, extras]
    
    return all_lists
