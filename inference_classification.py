import numpy as np
from build_model import BasicClasModel
from basic_config import config
from datetime import datetime
import pandas as pd
import json

from utils import seed_everything
from dataset import CustomDataset, get_valid_transforms_img, get_valid_transforms_npy

import torch
import os

def inference(model, data_loader, device, require_prob=False):
    model.eval()
    preds = []
    for step, imgs in enumerate(data_loader):
        imgs = imgs.to(device).float()
        outputs = model(imgs).detach().cpu().numpy()
        preds.append(outputs)
    y_pred = np.concatenate(preds)
    if not require_prob:
        y_pred = np.argmax(y_pred, axis=1).astype(np.int)
    return y_pred


def run_inference():

    infer_mode = config.infer_mode

    device_ids = list(range(torch.cuda.device_count()))
    model = BasicClasModel(config).to(config.device)
    model = torch.nn.DataParallel(model, device_ids)

    if config.postfix == 'npy':
        tsfm_test = get_valid_transforms_npy(config)
    else:
        tsfm_test = get_valid_transforms_img(config)
    
    test_ds = CustomDataset(pd.read_csv(config.TEST_CSV),
                            config.TEST_IMAGE_DIR,
                            transforms=tsfm_test,
                            output_label=False,
                            postfix=config.postfix,
                            num_class=config.num_class,
                            mapp=config.classname_id_map
                            )

    data_loader = torch.utils.data.DataLoader(test_ds,
                                                batch_size=config.valid_bs,
                                                num_workers=config.num_workers,
                                                shuffle=False)
    test_df = pd.DataFrame()
    

    for ii, fold in enumerate(config.use_folds):

        print("[PREDICT] predict using {}th fold model".format(fold))
        model_path = os.path.join(config.OUTPUT_DIR, config.EXP_NAME,\
                 'arch_{}_best_ps_fold_{}.ckpt'.format(config.model_arch, fold))
        model.load_state_dict(torch.load(model_path)['model'])
        with torch.no_grad():
            pred = inference(model, data_loader, config.device)

        if infer_mode == 'mode':
            test_df = pd.concat([test_df, pd.DataFrame(pred)], axis=1)
        else:
            assert infer_mode == 'avg'
            if ii == 0:
                test_df_tmp = pred
            else:
                test_df_tmp += pred
            if ii == len(config.use_folds) - 1:
                test_df = pd.DataFrame(test_df_tmp)

    if infer_mode == 'mode':
        test_pred = test_df.mode(axis=1)
    else:
        test_pred = test_df.mean(axis=1)
    sub_df = pd.read_csv(config.TEST_CSV)
    sub_df['final_prediction'] = test_pred.astype(np.int)

    print(datetime.ctime(datetime.now()), 'prediction merged, saving ... ')
    now = datetime.now().strftime("%m%d-%H%M")
    os.makedirs('./submission', exist_ok=True)
    
    sub_df.to_csv('./submission/submission_model_{}_{}_{}.csv'\
        .format(config.EXP_NAME + '_' + config.model_arch, config.infer_mode, now), header=True, index=False)
    
    print(sub_df.head())
    print("saved, show statistics ... ")

    print("\nclassid \t percentage")
    print(sub_df['final_prediction'].value_counts(normalize=True))

    print("\nclassname \t percentage")
    reverse_map = dict([(config.classname_id_map[i], i) for i in config.classname_id_map])
    print(sub_df['final_prediction'].apply(lambda x: reverse_map[x]).value_counts(normalize=True))

if __name__ == "__main__":
    seed_everything(config.seed)
    run_inference()

