import pandas as pd
sub1 = pd.read_csv('lgbm_submission.csv')
sub2 = pd.read_csv('tabnet_submission.csv')

sub1['per_square_meter_price'] = sub1['per_square_meter_price']*0.75+sub2['per_square_meter_price']*0.25
sub1.to_csv('lgbm_tabnet_submission.csv', index=False)