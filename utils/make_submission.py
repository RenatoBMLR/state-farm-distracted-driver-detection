import pandas as pd
import os
import datetime


def create_submission(result, info):
    
    predictions = result['pred'].tolist()
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