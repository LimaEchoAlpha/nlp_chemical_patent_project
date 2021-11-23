
import numpy as np

def train_test_split(all_lists):
    """
    Quick and dirty code for doing a train/test split on sample data
    adopted from BERT_T5_NER_2_3_030521 notebook from office hours
    """
    
    numSentences = len(all_lists[0][0])
    np.random.seed(424)
    training_examples = np.random.binomial(1, 0.7, numSentences)
    
    trainSentence_ids = []
    trainMasks = []
    trainSequence_ids = []
    trainE1Masks = []
    trainE2Masks = []

    testSentence_ids = []
    testMasks = []
    testSequence_ids = []
    testE1Masks = []
    testE2Masks = []

    labels_train =[]
    labels_test = []

    for example in range(numSentences):
        if training_examples[example] == 1:
            trainSentence_ids.append(all_lists[0][0][example])
            trainMasks.append(all_lists[0][1][example])
            trainSequence_ids.append(all_lists[0][2][example])
            labels_train.append(all_lists[1][example])
            if len(all_lists[0][3]) > 0:
                trainE1Masks.append(all_lists[0][3][example])
                trainE2Masks.append(all_lists[0][4][example])
        else:
            testSentence_ids.append(all_lists[0][0][example])
            testMasks.append(all_lists[0][1][example])
            testSequence_ids.append(all_lists[0][2][example])
            labels_test.append(all_lists[1][example])
            if len(all_lists[0][3]) > 0:
                testE1Masks.append(all_lists[0][3][example])
                testE2Masks.append(all_lists[0][4][example])

    X_train = [np.array(trainSentence_ids), np.array(trainMasks), np.array(trainSequence_ids), 
               np.array(trainE1Masks), np.array(trainE2Masks)]
    X_test = [np.array(testSentence_ids), np.array(testMasks), np.array(testSequence_ids), 
              np.array(testE1Masks), np.array(testE2Masks)]

    reLabels_train = np.array(labels_train)
    reLabels_test = np.array(labels_test)
    
    train_all = [X_train, reLabels_train]
    test_all = [X_test, reLabels_test]
    
    return train_all, test_all
