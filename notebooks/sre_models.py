
import numpy as np
import tensorflow as tf
import bert
from transformers import BertTokenizer, TFBertModel

def sre_cls_model(bert_model, max_length, train_layers=0):
    """
    Implementation of a Single Relation Extraction (SRE) Model 
    with [CLS] fixed length relation representation
    Reference: https://arxiv.org/pdf/1906.03158.pdf
    
    Variables:
    bert_model = pre-trained BERT model to be used as bert_layer
    max_length = number of tokens per snippet entry
    train_layers = number of layers to be retrained (optional)
    
    Returns: 
    Keras model
    """
    
    # input placeholders
    in_id = tf.keras.layers.Input(shape=(max_length,), dtype='int32', name='input_ids')
    in_mask = tf.keras.layers.Input(shape=(max_length,), dtype='int32', name='input_masks')
    in_segment = tf.keras.layers.Input(shape=(max_length,), dtype='int32', name='segment_ids')
    
    inputs = [in_id, in_mask, in_segment]
    bert_inputs = [inputs[0], inputs[2]]
    bert_layer = bert_model
    
    # optional: freeze layers, i.e. only train number of layers specified, starting from the top
    if not train_layers == -1:
        retrain_layers = []
        for retrain_layer_number in range(train_layers):
            layer_code = '_' + str(11 - retrain_layer_number)
            retrain_layers.append(layer_code)
        for w in bert_layer.weights:
            if not any([x in w.name for x in retrain_layers]):
                w._trainable = False
    # end of freezing section
    
    # pick out representation for [CLS] token
    bert_output = bert_layer(bert_inputs)
    cls = bert_output[:, 0, :]
    
    # post transformer layers (3): 
    # dense with linear activation, drop out, and prediction    
    dense = tf.keras.layers.Dense(256, activation='relu', name='dense')(cls)
    dense = tf.keras.layers.Dropout(rate=0.1)(dense)
    predictions = tf.keras.layers.Dense(2, activation='softmax', name='sre')(dense)
    
    # build model
    model = tf.keras.Model(inputs=inputs, outputs=predictions, name='sre_cls')
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    accuracy = tf.keras.metrics.Accuracy()
    binary_accuracy = tf.keras.metrics.BinaryAccuracy()
    recall = tf.keras.metrics.Recall()
    precision = tf.keras.metrics.Precision()
    
    model.compile(loss="binary_crossentropy", optimizer=optimizer,
                  metrics=[binary_accuracy, recall, precision])
    
    print()
    print("=== SRE [CLS] Model ===")
    print('BERT layer output:', bert_output)
    print('Prediction:', predictions)
    print()

    model.summary()
    
    return model



def sre_start_model(bert_model, max_length, train_layers=0):
    """
    Implementation of a Single Relation Extraction (SRE) Model 
    with Entity Start State: obtains a fixed length relation representation
    by concatenating the final hidden states corresponding to the 
    start tokens of each entity
    Reference: https://arxiv.org/pdf/1906.03158.pdf
    
    Variables:
    bert_model = pre-trained BERT model to be used as bert_layer
    max_length = number of tokens per snippet entry
    train_layers = number of layers to be retrained (optional)
    
    Returns: 
    Keras model
    """
    
    # input placeholders
    in_id = tf.keras.layers.Input(shape=(max_length,), dtype='int32', name='input_ids')
    in_mask = tf.keras.layers.Input(shape=(max_length,), dtype='int32', name='input_masks')
    in_segment = tf.keras.layers.Input(shape=(max_length,), dtype='int32', name='segment_ids')
    e1_mask = tf.keras.layers.Input(shape=(max_length,), dtype=tf.bool, name='e1_mask')
    e2_mask = tf.keras.layers.Input(shape=(max_length,), dtype=tf.bool, name='e2_mask')
    
    inputs = [in_id, in_mask, in_segment, e1_mask, e2_mask]
    bert_inputs = [inputs[0], inputs[2]]
    bert_layer = bert_model
    
    # optional: freeze layers, i.e. only train number of layers specified, starting from the top
    if not train_layers == -1:
        retrain_layers = []
        for retrain_layer_number in range(train_layers):
            layer_code = '_' + str(11 - retrain_layer_number)
            retrain_layers.append(layer_code)
        for w in bert_layer.weights:
            if not any([x in w.name for x in retrain_layers]):
                w._trainable = False
    # end of freezing section
    
    bert_output = bert_layer(bert_inputs)
    
    # apply masks to pick out start entity tokens
    e1_start = tf.ragged.boolean_mask(bert_output, e1_mask, name='e1_mention')
    e1_start = tf.squeeze(e1_start, axis=1)
    e2_start = tf.ragged.boolean_mask(bert_output, e2_mask, name='e2_mention')
    e2_start = tf.squeeze(e2_start, axis=1)
    
    # concatenate start entity representations
    dense_input = tf.keras.layers.Concatenate()([e1_start, e2_start])
    
    # post transformer layers (3): 
    # dense with linear activation, drop out, and prediction 
    dense = tf.keras.layers.Dense(256, activation='relu', name='dense')(dense_input)
    dense = tf.keras.layers.Dropout(rate=0.1)(dense)
    predictions = tf.keras.layers.Dense(2, activation='softmax', name='sre')(dense)
    
    # build model
    model = tf.keras.Model(inputs=inputs, outputs=predictions, name='sre_start')
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    accuracy = tf.keras.metrics.Accuracy()
    binary_accuracy = tf.keras.metrics.BinaryAccuracy()
    recall = tf.keras.metrics.Recall()
    precision = tf.keras.metrics.Precision()
    
    model.compile(loss="binary_crossentropy", optimizer=optimizer, 
                  metrics=[accuracy, binary_accuracy, recall, precision])
    
    print()
    print("=== SRE Start Entity Model ===")
    print('BERT layer output:', bert_output)
    print('Prediction:', predictions)
    print()
    
    model.summary()
    
    return model



def sre_pool_model(bert_model, max_length, train_layers=0):
    """
    Implementation of a Single Relation Extraction (SRE) Model 
    with Mention Pooling: obtains a fixed length relation representation
    by max pooling the final hidden layers corresponding to the two entities
    and concatenating the two vectors
    Reference: https://arxiv.org/pdf/1906.03158.pdf
    
    Variables:
    bert_model = pre-trained BERT model to be used as bert_layer
    max_length = number of tokens per snippet entry
    train_layers = number of layers to be retrained (optional)
    
    Returns: 
    Keras model
    """
    
    # input placeholders
    in_id = tf.keras.layers.Input(shape=(max_length,), dtype='int32', name='input_ids')
    in_mask = tf.keras.layers.Input(shape=(max_length,), dtype='int32', name='input_masks')
    in_segment = tf.keras.layers.Input(shape=(max_length,), dtype='int32', name='segment_ids')
    e1_mask = tf.keras.layers.Input(shape=(max_length,), dtype=tf.bool, name='e1_mask')
    e2_mask = tf.keras.layers.Input(shape=(max_length,), dtype=tf.bool, name='e2_mask')
    
    inputs = [in_id, in_mask, in_segment, e1_mask, e2_mask]
    bert_inputs = [inputs[0], inputs[2]]
    bert_layer = bert_model
    
    # optional: freeze layers, i.e. only train number of layers specified, starting from the top
    if not train_layers == -1:
        retrain_layers = []
        for retrain_layer_number in range(train_layers):
            layer_code = '_' + str(11 - retrain_layer_number)
            retrain_layers.append(layer_code)
        for w in bert_layer.weights:
            if not any([x in w.name for x in retrain_layers]):
                w._trainable = False
    # end of freezing section

    bert_output = bert_layer(bert_inputs)
    
    # apply masks to pick out outputs for mention tokens    
    e1_mention = tf.ragged.boolean_mask(bert_output, e1_mask, name='e1_mention')
    e1_mention = e1_mention.to_tensor()
    e2_mention = tf.ragged.boolean_mask(bert_output, e2_mask, name='e2_mention')
    e2_mention = e2_mention.to_tensor()
    
    # max pool entity mentions
    e1_max = tf.math.reduce_max(e1_mention, axis=1, name='e1_max')
    e2_max = tf.math.reduce_max(e2_mention, axis=1, name='e2_max')
    
    # concatenate max pooled entity mentions
    dense_input = tf.keras.layers.Concatenate()([e1_max, e2_max])
    
    # post transformer layers (3): 
    # dense with linear activation, drop out, and prediction
    dense = tf.keras.layers.Dense(256, activation='relu', name='dense')(dense_input)
    dense = tf.keras.layers.Dropout(rate=0.1)(dense)
    predictions = tf.keras.layers.Dense(2, activation='softmax', name='sre')(dense)
    
    # build model
    model = tf.keras.Model(inputs=inputs, outputs=predictions, name='sre_pool')
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    accuracy = tf.keras.metrics.Accuracy()
    binary_accuracy = tf.keras.metrics.BinaryAccuracy()
    recall = tf.keras.metrics.Recall()
    precision = tf.keras.metrics.Precision()
    
    model.compile(loss="binary_crossentropy", optimizer=optimizer, 
                  metrics=[accuracy, binary_accuracy, recall, precision])
    print()
    print("=== SRE Max Pool Model ===")
    print('BERT layer output:', bert_output)
    print('Prediction:', predictions)
    print()
    
    model.summary()
    
    return model
