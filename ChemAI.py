import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import json
import pickle
import ast
import re


class ChemAI():
    def __init__(self, ghs_path, dict_path=r'smiles_char_dict.json'):
        print('Created ChemAI instance')
        # default: predict label from smiles
        self.with_smiles = True

        if self.with_smiles:
            # load smiles_char_dict
            # {unit_of_interest: unique_int}
            # e.g. {'C': 12, 'Cl': 20}
            with open(dict_path, 'r') as dict_file:
                self.smiles_char_dict = json.load(dict_file)

            # {'C': 12} -> {12:'C'}
            self.inverted_smile_dict = {
                (value, key) for key, value in self.smiles_char_dict.items()}

        # load main data frame
        convert = {'IntSMILES': eval, 'GHS': eval}
        self.df = pd.read_csv(ghs_path, converters=convert)

    def format_features(self):
        print('Formating features')

        if self.with_smiles:
            int_smiles = self.df['IntSMILES'].tolist()

        # pad/truncate all smiles to the same length
        # 'X' = padding character; smiles_len defined in Chiled class
            int_smiles = tf.keras.preprocessing.sequence.pad_sequences(
                int_smiles, maxlen=self.smile_len, dtype='int16', padding='post', truncating='post',
                value=self.smiles_char_dict['X']
            )
            self.X = np.asarray(int_smiles)

    def shuffle_data(self):
        print('Shuffeling your data')
        # self.y is defined in child class
        # self.y needs to be np.array type

        assert len(self.y) == len(self.X)

        # combine into one helper array
        shuff_placeholder = np.c_[self.X.reshape(
            len(self.X), -1), self.y.reshape(len(self.y), -1)]

        # shuffle without losing connection between features and labels
        np.random.shuffle(shuff_placeholder)

        # reassign self.X and self.y to the shuffled form
        shuff_X = shuff_placeholder[:, :self.X.size //
                                    len(self.X)].reshape(self.X.shape)
        shuff_y = shuff_placeholder[:, self.X.size //
                                    len(self.X):].reshape(self.y.shape)

        self.X = shuff_X
        self.y = shuff_y

    def fit_model(self):
        # split train_data into train_data and validation_data
        # better value to determine how good the model is doing
        # (testing accuracy on examples it didnt learn from)

        nr_to_val = int(0.1 * len(self.X))
        x_val = self.X[:nr_to_val].astype('float64')
        x_train = self.X[nr_to_val:].astype('float64')

        y_val = self.y[:nr_to_val]
        y_train = self.y[nr_to_val:]

        # define opitimzer and loss function
        opt = tf.keras.optimizers.Adam(learning_rate=0.0007)
        loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)

        # start model
        model = keras.Sequential()

        # __define input layer
        if self.with_smiles:
            # defines 8 dimensional vector for each of the 32 possible units (random initialized)
            model.add(keras.layers.Embedding(32, 8))
            # reduces dimensions
            model.add(keras.layers.GlobalAveragePooling1D())

        else:
            # input layer when smiles are not used
            model.add(keras.layers.Dense(len(x_train[0]), activation='relu'))

        # __magic
        model.add(keras.layers.Dense(32, activation='relu'))

        # __define output layer
        if self.to_predict == 'GHS':
            # 9 possible GHS to predict
            model.add(keras.layers.Dense(9, activation='softmax'))

        elif self.to_predict == 'LogP':
            # 12 possible LogP values to predict (-5 to 6)
            model.add(keras.layers.Dense(12, activation='softmax'))

        model.build(input_shape=x_train.shape)
        model.summary()

        model.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=3, batch_size=4,
                  validation_data=(x_val, y_val), verbose=1)

        self.model = model

    def decode_int_smile(self, int_smile):
        # takes list repr of a smile and returns a readable format
        print('The list you entered translates to:')
        decoded_smile = ''
        for num in int_smile:
            if num == -1:
                continue
            decoded_smile += self.inverted_smile_dict[num]
        print(decoded_smile)
        # return decoded_smile
