import pandas as pd

path = '/home/gordon/work/small_construct/data/raw/personal_assets/'

chuckBrokerage_positions = pd.read_csv(path + 'chuckBrokerage_positions.csv')
print(chuckBrokerage_positions)