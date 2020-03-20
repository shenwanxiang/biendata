import pandas as pd
import os


def load_config(ftype = 'feature', metric = 'cosine'):
    
    name = '%s_%s.cfg.gzip' % (ftype, metric)
    
    dirf = os.path.dirname(__file__)
    filename = os.path.join(dirf, name)

    df = pd.read_pickle(filename, compression = 'gzip')

    return df