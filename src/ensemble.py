import pandas as pd
import numpy as np
import datetime
import glob

def averaging(pred_list, submission_name='Ensemble'):

    cols = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']

    pred_ens = pd.DataFrame(np.zeros(79726*10).reshape(79726,10), columns=cols)
    for i in pred_list:
        files = glob.glob('./subm/submission_' + i +'*.csv')
        a = pd.read_csv(files[0])
        pred_ens[cols] += a[cols]

    now = datetime.datetime.now()
    time_now = str(now.strftime("%Y-%m-%d-%H-%M"))
    pred_ens = pred_ens / len(pred_list)
    pred_ens['img'] = a['img'].values
    pred_ens.to_csv('./subm/submission_' + submission_name + '_distracted_driver_' + time_now + '.csv', index=False)
    print('done!')
    
if __name__ == "__main__":

    
    pred_lists = [
    ]

    averaging(pred_lists, 'ensemble_models')
