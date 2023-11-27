mode = 'test' # train or test
################################################
#                   training                   #
################################################
lr = 1e-4
epochs = 3000
weight_decay = 1e-5
weight_step = 500
betas = (0.5, 0.999)
gamma = 0.5
# dataset
crop_size_train = 256  # size for training
resize_size_test = 512  # size for testing
train_data_dir = ''
test_data_dir = ''
# Saving checkpoints
test_freq = 2
save_freq = 2
save_start_epoch = 1
model_save_dir = 'model_zoo'
# pusnet
pusnet_device_ids = '0, 1, 2, 3'
pusnet_batch_size_train = 8
pusnet_batch_size_test = 2
pusnet_sigma = 20
pusnet_lambda_S = 1.0
pusnet_lambda_R = 0.75
pusnet_lambda_DN = 0.25
sparse_ratio = 0.9
# pusnet-p
pusnet_p_device_ids = '3, 4'
pusnet_p_batch_size_train = 8
pusnet_p_batch_size_test = 2
################################################
#                   testing                   #
################################################
test_pusnet_path = ''
test_pusnet_p_path = ''
save_processed_img = True
resi_magnification = 5









