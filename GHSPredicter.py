from ChemAI import ChemAI
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import json
import pickle
import ast
import re


class GHSPredicter(ChemAI):
    def __init__(self):
        ghs_path = r'.\Chem_data\ghs_with_int_smiles.csv'
        super().__init__(ghs_path)
        self.to_predict = 'GHS'
        # predict GHS with smiles
        self.with_smiles = True
        # set max length of the smiles strings
        self.smile_len = 50

    def build_classifier(self):
        super().format_features()

        # if prediction with values change features accordingly
        if not self.with_smiles:
            self.fine_tune_features()
        self.format_labels()
        super().shuffle_data()
        super().fit_model()

    def fine_tune_features(self):
        self.X = self.df.drop(
            ['IUPACName', 'CID', 'MolecularFormula', 'IntSMILES'], axis=1, inplace=False)

        # the larger dataset doesnt have these columns
        try:
            self.X.drop(['GHS', 'CAS'], axis=1, inplace=True)
        except:
            pass

        # features to correct shape and type
        self.X = np.asarray(self.X)
        for i in range(len(self.X)):
            self.X[i] = np.asarray(self.X[i])
        self.X = np.float32(self.X)

    def format_labels(self):
        # labels to correct shape and type
        self.y = self.df['GHS'].tolist()
        self.y = np.asarray(self.y)

    def predict_ghs(self, smile):
        # takes one smile string

        print('The predictions to your requests:')

        def translate_to_int(smile):
            # returns list of ints corresponding to self.smiles_char_dict 'C'->12; Cl->13; '='->5 etc
            int_smile = []

            # get all groups
            atoms = re.finditer(r'[A-Z]{1}[a-z]{0,1}', smile)
            signs = re.finditer('\W', smile)
            nums = re.finditer('\d{1}', smile)

            order_dict = {}

            # fill the dict
            for atom in atoms:
                order_dict[atom.span()[0]] = atom.group()
            for sign in signs:
                order_dict[sign.span()[0]] = sign.group()
            for num in nums:
                order_dict[num.span()[0]] = num.group()

            # sort the dict by starting of span (their first index)
            order_dict = dict(sorted(order_dict.items()))

            for index, unit in order_dict.items():
                int_smile.append(self.smiles_char_dict[unit])

            return int_smile

        int_smile = translate_to_int(smile)

        # pad/truncate the smile to the correct length
        # 'X' = padding character
        to_pred = tf.keras.preprocessing.sequence.pad_sequences(
            [int_smile], maxlen=self.smile_len, dtype='int16', padding='post', truncating='post',
            value=self.smiles_char_dict['X']
        )

        # predict and print GHS classifications with descending probability
        predictions = self.model.predict(to_pred)
        predictions = predictions[0].tolist()

        ghs_pred = []
        for _ in range(len(predictions)):
            m = max(predictions)
            curr_prediction = predictions.index(m)+1
            ghs_pred.append(curr_prediction)
            predictions[predictions.index(m)] = -1
        print(f'Predicted GHS with decending probability {ghs_pred}')
