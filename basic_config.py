# configure file
# set all paramters about dataset/model/training in this class

class Config:

    # experiment name, identifier
    EXP_NAME = 'run_baseline'

    # fold and dataset related
    DEBUG = False
    fold_num = 3
    use_folds = [0]
    task_name = 'label_name'
    seed=20

    img_size_h = 128
    img_size_w = 128
    num_input_chs = 8
    num_class = 10
    postfix = 'npy'
    norm_mean = None
    norm_std = None
    classname_id_map = None

    # model related
    model_arch = 'tf_efficientnet_b5_ns'
    ckpt = './pretrained/tf_efficientnet_b5_ns-6f26d0cf.pth'

    # training settings
    epochs = 30
    train_bs=32
    valid_bs=32
    T_0=10
    lr=1e-4
    min_lr=1e-7
    weight_decay=1e-6
    num_workers=8
    accum_iter = 2
    verbose_step=50
    device='cuda:0'

    # inference (test) settings
    infer_mode = 'avg' # 'avg' or 'mode'

    # directories and pathes
    TRAIN_IMAGE_DIR = "../cerfacs_land_cover_dataset/"
    TRAIN_CSV = "../cerfacs_land_cover_csv/train_labels.csv"
    TEST_IMAGE_DIR = "../cerfacs_land_cover_dataset/"
    TEST_CSV = "../cerfacs_land_cover_csv/test_imids.csv"
    OUTPUT_DIR = '../output' # must no "/" in OUTPUT_DIR


config=Config()
