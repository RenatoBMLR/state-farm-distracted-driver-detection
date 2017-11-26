import pandas as pd
import os
import datetime
import torch
import numpy as np

def create_submission(result, info):
    
    m = torch.nn.Softmax()
    predictions =  np.around(m(result['pred']).cpu().data.numpy(), decimals=1).tolist()
    
    test_id = result['true'].tolist()
    for i in range(0, len(test_id)):
        test_id[i] = 'img_'+ str(test_id[i]) + '.jpg'

    result_sample = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    result_sample.loc[:, 'img'] = pd.Series(test_id, index=result_sample.index)

    cols = result_sample.columns.tolist()
    cols = cols[-1:] + cols[:-1]

    result_sample = result_sample[cols]
    now = datetime.datetime.now()
    if not os.path.isdir('subm'):
        os.mkdir('subm')
    suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join('subm', 'submission_' + suffix + '.csv')
    result_sample.to_csv(sub_file, index=False)
    print('done!')
    
def save_results(results, info):    
    for key in results.keys():

        data ={'pred':results[key]['pred'].cpu().numpy(),
                   'true':results[key]['true'].cpu().numpy()}

        now = datetime.datetime.now()
        if not os.path.isdir('results'):
            os.mkdir('results')
        suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
        sub_file = os.path.join('results', 'results_' + suffix + '.npz')
        np.savez(sub_file,**data)
        print('Save result from '+ key + ' set ' + 'done!')
    
def metrics2csv(trainer, info):    
    now = datetime.datetime.now()
    if not os.path.isdir('metrics'):
        os.mkdir('metrics')
    suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join('metrics', 'metrics_' + suffix + '.csv')
    df = pd.DataFrame(trainer.metrics)

    df.to_csv(sub_file, index=False)
    print('done!')