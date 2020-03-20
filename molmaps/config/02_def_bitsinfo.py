import pandas as pd
import numpy as np

color_maps = {'descriptor': '#ff6a00', 
             'fingerprint':'#a700ff' }


df = pd.read_pickle('feature_scale.cfg')


bitsinfo = pd.DataFrame(df.index,columns = ['IDs'])
subtypes =  ['fingerprint' for i in range(len(df) - 27)]
subtypes.extend(['descriptor' for i in range(27)])



bitsinfo['Subtypes'] = subtypes
bitsinfo['colors'] = bitsinfo['Subtypes'].map(color_maps)

bitsinfo.to_pickle('feature_bitsinfo.cfg', compression = 'gzip')