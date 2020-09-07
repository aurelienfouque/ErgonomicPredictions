#!/usr/local/bin/python3.7

import pandas as pd
import libra
from libra import client

clous = client('housing.csv')

data = pd.read_csv('housing.csv')
data.columns
data.head(10)

clous.neural_network_query('estimate ocean proximity', epochs = 30)
clous.analyze()

clous.accuracy()
clous.losses()
clous.info()

clous.neural_network_query('model median house value',
    drop = ['ocean_proximity'],
    save_model = True)

