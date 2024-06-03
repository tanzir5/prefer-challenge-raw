import pandas as pd
from tqdm import tqdm
def fix_train_background(path=None):
  train_background = pd.read_csv('other_data/PreFer_train_background_data.csv')

  unique_waves = train_background['wave'].unique()
  columns_to_augment = train_background.columns.difference(
    ['nomem_encr', 'wave']
  )
  new_data = []
  for nomem_encr, group in tqdm(train_background.groupby('nomem_encr')):
    new_row = {'nomem_encr': nomem_encr}
    for wave in unique_waves:
      wave_group = group[group['wave'] == wave]
      for col in columns_to_augment:
        col_name = f'bg_wave_{wave}_{col}'
        assert wave_group.empty or len(wave_group[col].values == 1)
        new_row[col_name] = wave_group[col].values[0] if not wave_group.empty else None
    new_data.append(new_row)
  return pd.DataFrame(new_data)

fixed_train_background = fix_train_background()
fixed_train_background.to_csv('fixed_train_background.csv',index=False)

