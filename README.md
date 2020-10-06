# ChemAI
Goal: Predict different chemical properties.

## Disclaimer
* This is my first AI
* Right now the model does not give reliable predictions (See Problem section)
* If you have suggestions how to improve my code or do things more efficiently I am happy to hear from you



## Files explained
### ./Raw_data/*.csv
Right now only the file 'all_chem_with_ghs.csv' is available. It contains all chemicals from https://pubchem.ncbi.nlm.nih.gov/ with a CID < 1.000.000 and a GHS symbol.

### Modify_data.py
Formats raw_data files into a usable format for the predicter classes.
Has to be executed before the first call of any predicter to ensure the needed data is avaiable
* creates a dict of all appearing units in the used smiles strings. `'C=O' -> {'C':12, '=': 13, 'O':14}`
* formats smiles strings into a list of integer representation: `'CC' -> [12,12]`
* formats ghs list into list of boleans: `[2,7] -> [0,1,0,0,0,0,1,0,0]`
* saves formated data in the main folder as csv or pickle

### ChemAI.py
`class ChemAI()`
Parent Class of all following Predicters to reduce redundant code.
Model build is strongly influenced by the type of data that is to be predicted. 

### GHSPredicter.py
For non-chemists: There are 9 GHS symbols. They represent dangerous properties of chemical substances (See https://pubchem.ncbi.nlm.nih.gov/ghs/)
`class GHSPredicter(ChemAI)`
When executed, trains a model that predicts GHS symbols.

`def predict_ghs(self, smile)`
When given any smiles string, prints a list of GHS symbols with descending probability for the given compound.


## HOW TO use
clone or download repository 
make sure you have the most commen ML packages installed (tensorflow, numpy, pandas)
execute modify_data.py
`python modify_data.py`

make instance of GHSPredicter:
```
chem_model = GHSPredicter()
your_smile = 'CC(=O)C'
chem_model.predict_ghs(your_smile)
```
## Problems
The current data set has GHS7 for about 80% of the ~27.000 compounds. 
This leads to the model predicting GHS7 for every molecule first and neglect the others. 
A larger more balanced data set would be needed to eradicate this unflattering behavior.

