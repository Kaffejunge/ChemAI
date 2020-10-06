import pandas as pd
import numpy as np
import json
import pickle
import re


class ChemDataModifier():
    def __init__(self, filepath):
        self.smiles_char_dict = {}

        # execute all needed operations
        self.load_data(filepath)
        self.create_smiles_char_dict()
        self.encode_smiles()

    def load_data(self, filepath):
        self.main_df = pd.read_csv(filepath)

        # drop useless or repetetiv data
        self.main_df.drop(['ExactMass', 'IsomericSMILES',
                           'RotatableBondCount'], axis=1, inplace=True)

        # this line eliminates all compounds with any metal in it
        # since they hava 'NaN' values -> 852593 compounds remain
        self.main_df.dropna(inplace=True)
        self.smiles = self.main_df['CanonicalSMILES']

    def create_smiles_char_dict(self):
        # dict will look like:
        # dict = {unit_of_interest: unique_int}
        # e.g. {'X':0, 'C':11, 'Cl':20, '(':30}

        print('Creating smiles char dict')

        # padding character = X
        self.smiles_char_dict['X'] = 0

        # numbers are themself
        for i in range(1, 10):
            self.smiles_char_dict[str(i)] = i

        self.smiles_char_dict['0'] = 10
        # in case of unknown replace with
        self.smiles_char_dict['UNK'] = 11
        i += 3

        for smile in self.smiles:

            # add atoms
            atoms = re.findall(r'[A-Z]{1}[a-z]{0,1}', smile)
            for atom in atoms:
                if atom not in self.smiles_char_dict.keys():
                    self.smiles_char_dict[atom] = i
                    i += 1

            # add extra signs
            signs = re.findall('\W', smile)
            for sign in signs:
                if sign not in self.smiles_char_dict.keys():
                    self.smiles_char_dict[sign] = i
                    i += 1

    def save_smile_char_dict(self, filename):
        json_dict = json.dumps(self.smiles_char_dict)
        with open(filename, "w") as dict_file:
            dict_file.write(json_dict)
        print(f'Saved the smiles_char_dict as {filename}.')

    def encode_smiles(self):
        print('Converting Smiles to integer list')

        def translate_to_int(smile):
            # takes one smile string
            # returns list of ints corresponding to self.smiles_char_dict 'C'->12; Cl->13; '='->5 etc
            int_smile = []

            # get all groups
            atoms = re.finditer(r'[A-Z]{1}[a-z]{0,1}', smile)
            signs = re.finditer('\W', smile)
            nums = re.finditer('\d{1}', smile)

            order_dict = {}

            # fill the dict with present units
            for atom in atoms:
                order_dict[atom.span()[0]] = atom.group()
            for sign in signs:
                order_dict[sign.span()[0]] = sign.group()
            for num in nums:
                order_dict[num.span()[0]] = num.group()

            # sort the dict by starting of span (their first index)
            order_dict = dict(sorted(order_dict.items()))

            # add the sorted ints to the return list
            for index, unit in order_dict.items():
                int_smile.append(self.smiles_char_dict[unit])

            return int_smile

        #! main code for encode_smiles
        self.int_smiles = []
        for smile in self.smiles:
            self.int_smiles.append(translate_to_int(smile))

    def update_dataframe(self):
        self.main_df['IntSMILES'] = self.int_smiles
        self.main_df.drop(['CanonicalSMILES'], inplace=True, axis=1)

    def save_dataframe(self, filename, file_type='csv'):
        self.update_dataframe()
        if file_type == 'csv':
            self.main_df.to_csv(filename, index=False)
            print(f'Saved the Data Frame as csv file: {filename}.')
        if file_type == 'pickle':
            pickle.dump(self.main_df, filename)
            print(f'Saved the Data Frame as pickle file: {filename}.')


if __name__ == '__main__':
    # This list will expand with different Predicters
    list_to_load = [r'.\Raw_data\all_chem_with_ghs.csv']
    list_to_save = [r'.\Chem_data\ghs_with_int_smiles.csv']

    # NOTE to self: only vary i nothing else :)
    i = 0

    myModifier = ChemDataModifier(list_to_load[i])
    myModifier.save_smile_char_dict('smiles_char_dict.json')
    myModifier.save_dataframe(list_to_save[i], file_type='csv')
