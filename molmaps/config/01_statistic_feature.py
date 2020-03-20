from molmaps import distances, calculator, summary
import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm import tqdm
tqdm.pandas(ascii=True)


dfx = pd.read_csv('./molecule_open_data/candidate_train.csv')
dfx = dfx.set_index('id')
dfy = pd.read_csv('./molecule_open_data/train_answer.csv')
dfy = dfy.set_index('id')

df = dfx.join(dfy)
data = dfx.values


if '__name__' == '__main__':
    dfres = summary.Summary2(data, n_jobs=4)
    dfres.index = dfx.columns #index is string
    dfres = dfres.astype('float32')
    dfres.to_pickle('./feature_scale.cfg', compression = 'gzip')
