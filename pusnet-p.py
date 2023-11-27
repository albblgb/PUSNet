import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import logging
import numpy as np
import math
from torchvision.utils import save_image

from models.PUSNet import pusnet
from utils.terminal import MetricMonitor
from utils.logger import logger_info
from utils.image import calculate_psnr, calculate_ssim, calculate_mae, calculate_rmse
from utils.dataset import load_dataset
from utils.dirs import mkdirs
import config as c
from utils.model import load_model
from utils.proposed_mothod import generate_sparse_mask, init_weights, remove_adapter

os.environ["CUDA_VISIBLE_DEVICES"] = c.pusnet_device_ids
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mkdirs('results/pusnet-p')
logger_name = 'pusnet-p'
logger_info(logger_name, log_path=os.path.join('results', logger_name, c.mode + '.log'))
logger = logging.getLogger(logger_name)
logger.info('#'*50)
logger.info('model: pusnet')
logger.info('train data dir: {:s}'.format(c.train_data_dir))
logger.info('test data dir: {:s}'.format(c.test_data_dir))
logger.info('mode: {:s}'.format(c.mode))
logger.info('noisy level: {:s}'.format(str(c.pusnet_sigma)))
logger.info('sparse ration: {:s}'.format(str(c.sparse_ratio)))


model_tmp = pusnet()
init_weights(model_tmp, random_seed=1)
sparse_mask = generate_sparse_mask(model_tmp, sparse_ratio=c.sparse_ratio)
for idx in range(len(sparse_mask)):
    sparse_mask[idx] = sparse_mask[idx].to(device)
model = pusnet().to(device)
model = nn.DataParallel(model)
remove_adapter(model, sparse_mask)

train_loader, test_loader = load_dataset(c.train_data_dir, c.test_data_dir, c.pusnet_p_batch_size_train, c.pusnet_p_batch_size_test, c.pusnet_sigma)


if c.mode == 'test':
    
    model.load_state_dict(torch.load(c.test_pusnet_p_path))

    with torch.no_grad():
        N_psnr = []; N_ssim = []; N_mae = []; N_rmse = []
        DN_psnr = []; DN_ssim = []; DN_mae = []; DN_rmse = []
        
        model.eval()
        stream = tqdm(test_loader)
        for idx, (data, noised_data) in enumerate(stream):
            data = data.to(device)
            noised_data = noised_data.to(device)

            clean = data[data.shape[0]//2:]
            noised = noised_data[noised_data.shape[0]//2:]
    
            ################## forward ####################
            denoised = model(noised, None, 'denoising')

            ############### save images #################
            if c.save_processed_img == True:
                super_dirs = ['noisy', 'denoised']
                for cur_dir in super_dirs:
                    test_data_name = c.test_data_dir.split('/')[-1]
                    mkdirs(os.path.join('results/pusnet-p', test_data_name, cur_dir))   
                image_name = '%.4d.' % idx + 'png'
                save_image(noised, os.path.join('results/pusnet-p', test_data_name, super_dirs[0], image_name))
                save_image(denoised, os.path.join('results/pusnet-p', test_data_name, super_dirs[1], image_name))

            ############### calculate metircs #################
            clean = clean.detach().cpu().numpy().squeeze() * 255
            np.clip(clean, 0, 255)
            noised = noised.detach().cpu().numpy().squeeze() * 255
            np.clip(noised, 0, 255)
            denoised = denoised.detach().cpu().numpy().squeeze() * 255
            np.clip(denoised, 0, 255)

            psnr_temp = calculate_psnr(clean, noised)
            N_psnr.append(psnr_temp)
            psnr_temp = calculate_psnr(clean, denoised)
            DN_psnr.append(psnr_temp)

            mae_temp = calculate_mae(clean, noised)
            N_mae.append(mae_temp)
            mae_temp = calculate_mae(clean, denoised)
            DN_mae.append(mae_temp)

            rmse_temp = calculate_rmse(clean, noised)
            N_rmse.append(rmse_temp)
            rmse_temp = calculate_rmse(clean, denoised)
            DN_rmse.append(rmse_temp)

            ssim_temp = calculate_ssim(clean, noised)
            N_ssim.append(ssim_temp)
            ssim_temp = calculate_ssim(clean, denoised)
            DN_ssim.append(ssim_temp)

        logger.info('testing, noise_avg_psnr: {:.2f}, denoise_avg_psnr: {:.2f}'.format(np.mean(N_psnr), np.mean(DN_psnr)))
        logger.info('testing, noise_avg_mae: {:.2f}, denoise_avg_mae: {:.2f}'.format(np.mean(N_mae), np.mean(DN_mae)))
        logger.info('testing, noise_avg_rmse: {:.2f}, denoise_avg_rmse: {:.2f}'.format(np.mean(N_rmse), np.mean(DN_rmse)))
        logger.info('testing, noise_avg_ssim: {:.4f}, denoise_avg_ssim: {:.4f}'.format(np.mean(N_ssim), np.mean(DN_ssim)))

else:

    denoising_loss = nn.MSELoss().to(device)

    # according to pusnet_p #2 and pusnet_p #3
    opt2 = torch.optim.Adam(model.parameters(), lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
    opt3 = torch.optim.Adam(model.parameters(), lr=c.lr)

    optimizer = opt2 
    # optimizer = opt3 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, c.weight_step, gamma=c.gamma)

    for epoch in range(c.epochs):
        epoch += 1
        dn_loss = []
        loss_history=[]
        ###############################################################
        #                            train                            # 
        ###############################################################
        model.train()
        metric_monitor = MetricMonitor(float_precision=4)
        stream = tqdm(train_loader)

        for batch_idx, (data, noised_data) in enumerate(stream):
            data = data.to(device)
            noised_data = noised_data.to(device)
            
            clean = data[data.shape[0]//2:]
            noised = noised_data[noised_data.shape[0]//2:]
    
            ################## forward ####################
            denoised = model(noised, None, 'denoising')
            
            ################### loss ######################
            DN_loss = denoising_loss(clean, denoised)
            loss = c.pusnet_lambda_DN * DN_loss 
            ################### backword ##################
            loss.backward()
            idx_m = 0
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    m.weight.grad.data = torch.mul(m.weight.grad.data, sparse_mask[idx_m])
                    idx_m += 1
                elif isinstance(m, nn.Linear):
                    m.weight.grad.data = torch.mul(m.weight.grad.data, sparse_mask[len(sparse_mask)-1])
            optimizer.step()   
            optimizer.zero_grad()
            
            ################## record ##################
            dn_loss.append(DN_loss.item())
            loss_history.append(loss.item())
            metric_monitor.update("DN_loss", np.mean(np.array(dn_loss)))
            metric_monitor.update("T_Loss", np.mean(np.array(loss_history)))
            stream.set_description(
                "Epoch: {epoch}. Train.   {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
            )

        epoch_losses = np.mean(np.array(loss_history))

        ###############################################################
        #                              val                            # 
        ###############################################################
        model.eval()
        if epoch % c.test_freq == 0:
            with torch.no_grad():
                N_psnr = []
                DN_psnr = []
                for (data, noised_data) in test_loader:
                    data = data.to(device)
                    noised_data = noised_data.to(device)

                    clean = data[data.shape[0]//2:]
                    noised = noised_data[noised_data.shape[0]//2:]
            
                    ################## forward ####################
                    denoised = model(noised, None, 'denoising')

                    ############### calculate psnr #################
                    clean = clean.detach().cpu().numpy().squeeze() * 255
                    np.clip(clean, 0, 255)
                    noised = noised.detach().cpu().numpy().squeeze() * 255
                    np.clip(noised, 0, 255)
                    denoised = denoised.detach().cpu().numpy().squeeze() * 255
                    np.clip(denoised, 0, 255)

                    psnr_temp = calculate_psnr(clean, noised)
                    N_psnr.append(psnr_temp)
                    psnr_temp = calculate_psnr(clean, denoised)
                    DN_psnr.append(psnr_temp)
    
                logger.info('epoch: {}, training, T_loss: {:.5f}'.format(epoch, epoch_losses))
                logger.info('epoch: {}, noise_avg_psnr: {:.2f}, denoise_avg_psnr: {:.2f}'.format(epoch, np.mean(N_psnr), np.mean(DN_psnr)))

        if epoch % c.save_freq == 0 and epoch >= (c.save_start_epoch):
            model_save_dir = os.path.join(c.model_save_dir, 'pusnet-P-'+ str(c.sparse_ratio))
            mkdirs(model_save_dir)
            torch.save(model.state_dict(), os.path.join(model_save_dir, 'checkpoint_%.4i' % epoch + '.pt'))
            
        scheduler.step()


    
