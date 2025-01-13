import pandas as pd
import numpy as np
import ConfigSpace
import sklearn.impute

'''R = []
l = []
dc = pd.read_csv('/Users/jiaxuanyu/Code/AML/A1/lcdb_configs.csv')
config_space = ConfigSpace.ConfigurationSpace.from_json('/Users/jiaxuanyu/Code/AML/A1/lcdb_config_space_knn.json')
df = np.array(dc)
df = df[np.argsort(df[:,-1])]

cf = df[:5, :]
for c in cf:
    config = c[:-1]
    performance = c[-1]
    l.append((config, performance))

for (c, p) in l:
     z = list(c)
     z.append(p)
     R.append(z)


key = dc.keys()
d = pd.DataFrame(R, columns=key)
print(d)'''

config_space = np.array([[1,2,3], [4, 3, 2], [2, 3, 4],[4, 5, 6]])
e = np.array([4, 5])
mask = (config_space[:, 0] == 4) & (config_space[:, 1] == 3)
print(np.where(mask))
matching_indices = np.where(mask)[0]
print(matching_indices)



