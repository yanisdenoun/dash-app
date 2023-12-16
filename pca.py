import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

df = pd.read_csv('data/2021-2022 Football Player Stats.csv', encoding='ISO-8859-1', delimiter=';') 
pca = PCA(n_components=10)
pca.fit(df)

print(pca.components_)