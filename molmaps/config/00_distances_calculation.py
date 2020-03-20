from molmaps import distances, calculator, summary
import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm import tqdm
tqdm.pandas(ascii=True)




def caldis(data, idx, tag, methods = ['correlation', 'cosine', 'jaccard']):
    
    
    ##############################################################
    Nf = len(feature.fingerprint.Extraction().bitsinfo)
    data0 = loadnpy('./data/fingerprint_8206960.npy', N = Nf, dtype = np.bool)
    groups = data0.sum(axis=1)
    from sklearn.model_selection import GroupKFold
    G  = GroupKFold(n_splits=10)
    sp = G.split(X = data0, groups=groups)
    spl = list(sp)
    sidx = spl[0][1]
    del data0
    print(len(sidx))
    
    data = data[sidx]
    data = data.astype(np.float32,copy=False)
    #############################################################
    
    for method in methods:
        res = calculator.pairwise_distance(data, n_cpus=16, method=method)
        res = np.nan_to_num(res,copy=False)
        df = pd.DataFrame(res,index=idx,columns=idx)
        df = df.astype('float32')
        df.to_pickle('./data/%s_%s.cfg' % (tag, method), compression = 'gzip')




if __name__ == '__main__':
    
    #discriptors distance
    dfx = pd.read_csv('./molecule_open_data/candidate_train.csv')
    dfx = dfx.set_index('id')

    data = dfx.values
    idx = dfx.columns
    
    tag = 'feature'
    caldis(data, idx, tag, methods = ['correlation', 'cosine'])
