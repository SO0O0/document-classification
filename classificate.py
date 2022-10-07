import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from bert import BertModelLayer
from bert.loader import (StockBertConfig, load_stock_weights,
                         map_stock_config_to_params)
from bert.tokenization.bert_tokenization import FullTokenizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from utils import *


def main():
    parser = argparse.ArgumentParser(description='Document Classification using BERT')
    parser.add_argument('--train', type=str, default='data/train.csv', help='path to the train data')
    parser.add_argument('--test', type=str, default='data/test.csv', help='path to the test data')
    parser.add_argument('--bert', type=str, default='uncased_L-12_H-768_A-12', help='path to the BERT model')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--max_len', type=int, default=128, help='maximum length of the input')
    args = parser.parse_args()

    # Load the data
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)

    # Split the data into train and validation
    train, val = train_test_split(train, test_size=0.1, random_state=42)

    # Load the BERT model
    bert_model = load_bert(args.bert)

    # Build the model
    model = build_model(bert_model, args.max_len)

    # Train the model
    train_model(model, train, val, args.epochs, args.batch_size, args.lr, args.max_len)

    # Evaluate the model
    evaluate_model(model, test, args.max_len)

def load_bert(bert_path):
    # Load the BERT model
    bert_model = BertModelLayer.from_params(
        bert_path,
        trainable=True
    )

    # Load the BERT weights
    bert_ckpt_file = os.path.join(bert_path, 'bert_model.ckpt')
    bert_config_file = os.path.join(bert_path, 'bert_config.json')

    bert_params = bert_model.params_from_pretrained_ckpt(bert_path)
    bert = bert_model(bert_params)
    bert.load_weights_from_checkpoint(bert_ckpt_file)

    return bert

def build_model(bert, max_len):
    # Build the model
    input_ids = tf.keras.layers.Input(shape=(max_len,), dtype='int32', name='input_ids')
    bert_output = bert(input_ids)

    print('bert shape', bert_output.shape)

    cls_out = tf.keras.layers.Lambda(lambda seq: seq[:, 0, :])(bert_output)
    cls_out = tf.keras.layers.Dropout(0.5)(cls_out)
    logits = tf.keras.layers.Dense(units=2, activation='softmax')(cls_out)

    model = tf.keras.models.Model(inputs=input_ids, outputs=logits)
    model.build(input_shape=(None, max_len))

    return model

def train_model(model, train, val, epochs, batch_size, lr, max_len):
    # Train the model
    train_x, train_y = preprocess(train, max_len)
    val_x, val_y = preprocess(val, max_len)

    model.compile(
        optimizer=optimizers.Adam(lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        train_x,
        train_y,
        validation_data=(val_x, val_y),
        epochs=epochs,
        batch_size=batch_size
    )

def evaluate_model(model, test, max_len):
    # Evaluate the model
    test_x, test_y = preprocess(test, max_len)

    y_pred = model.predict(test_x)
    y_pred = np.argmax(y_pred, axis=1)

    print(classification_report(test_y, y_pred))
    print(confusion_matrix(test_y, y_pred))

if __name__ == '__main__':
    main()
