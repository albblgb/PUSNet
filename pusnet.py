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
from utils.model import load_model
import config as c
from utils.proposed_mothod import generate_sparse_mask, init_weights, remove_adapter, insert_adapter


os.environ["CUDA_VISIBLE_DEVICES"] = c.pusnet_device_ids
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mkdirs('results/pusnet')
logger_name = 'pusnet'
logger_info(logger_name, log_path=os.path.join('results', logger_name, c.mode + '.log'))
logger = logging.getLogger(logger_name)
logger.info('#'*50)
logger.info('model: pusnet')
logger.info('train data dir: {:s}'.format(c.train_data_dir))
logger.info('test data dir: {:s}'.format(c.test_data_dir))
logger.info('mode: {:s}'.format(c.mode))
logger.info('noisy level: {:s}'.format(str(c.pusnet_sigma)))
logger.info('sparse ration: {:s}'.format(str(c.sparse_ratio)))


################## prepare ####################
model_hiding_seed = pusnet()
model_recover_seed = pusnet()

# mask generation accoding to random seed '1'
init_weights(model_hiding_seed, random_seed=1)
sparse_mask = generate_sparse_mask(model_hiding_seed, sparse_ratio=c.sparse_ratio)


for idx in range(len(sparse_mask)):
    sparse_mask[idx] = sparse_mask[idx].to(device)

# set hiding seed/key '10101' and recover seed/key '1010'
init_weights(model_hiding_seed, random_seed=10101)
init_weights(model_recover_seed, random_seed=1010)

model = pusnet().to(device)
# init_weights(model)
model_hiding_seed = model_hiding_seed.to(device)
model_recover_seed = model_recover_seed.to(device)

# multi GPUs
model = nn.DataParallel(model)
model_hiding_seed = nn.DataParallel(model_hiding_seed)
model_recover_seed = nn.DataParallel(model_recover_seed)

train_loader, test_loader = load_dataset(c.train_data_dir, c.test_data_dir, c.pusnet_batch_size_train, c.pusnet_batch_size_test, c.pusnet_sigma)

if c.mode == 'test':
    model.load_state_dict(torch.load(c.test_pusnet_path))
    # model = nn.DataParallel(model)

    with torch.no_grad():
        S_psnr = []; S_ssim = []; S_mae = []; S_rmse = []
        R_psnr = []; R_ssim = []; R_mae = []; R_rmse = []
        N_psnr = []; N_ssim = []; N_mae = []; N_rmse = []
        DN_psnr = []; DN_ssim = []; DN_mae = []; DN_rmse = []

        model.eval()
        stream = tqdm(test_loader)
        for idx, (data, noised_data) in enumerate(stream):
            data = data.to(device)
            noised_data = noised_data.to(device)
            
            secret = data[data.shape[0]//2:]
            cover = data[:data.shape[0]//2]
            clean = secret
            noised = noised_data[noised_data.shape[0]//2:]
    
            ################## forward ####################
            remove_adapter(model, sparse_mask)  
            denoised = model(noised, None, 'denoising') # pusnet-p
            insert_adapter(model, sparse_mask, model_hiding_seed)  
            stego = model(secret, cover, 'hiding') # pusnet-E
            insert_adapter(model, sparse_mask, model_recover_seed, is_sparse=False)  
            secret_rev = model(stego, None, 'recover') # pusnet-D

            cover_resi = abs(cover - stego) * c.resi_magnification
            secret_resi = abs(secret - secret_rev) * c.resi_magnification
            
            ############### save images #################
            if c.save_processed_img == True:
                super_dirs = ['cover', 'secret', 'stego', 'secret_rev', 'cover_resi', 'secret_resi', 'noisy', 'denoised']
                for cur_dir in super_dirs:
                    test_data_name = c.test_data_dir.split('/')[-1] # for example, c.test_data_dir = './testdata/coco' ==>  test_data_name = 'coco'
                    mkdirs(os.path.join('results/pusnet', test_data_name, cur_dir))    
                image_name = '%.4d.' % idx + 'png'
                save_image(cover, os.path.join('results/pusnet', test_data_name, super_dirs[0], image_name))
                save_image(secret, os.path.join('results/pusnet', test_data_name, super_dirs[1], image_name))
                save_image(stego, os.path.join('results/pusnet', test_data_name, super_dirs[2], image_name))
                save_image(secret_rev, os.path.join('results/pusnet', test_data_name, super_dirs[3], image_name))
                save_image(cover_resi, os.path.join('results/pusnet', test_data_name, super_dirs[4], image_name))
                save_image(secret_resi, os.path.join('results/pusnet', test_data_name, super_dirs[5], image_name))
                save_image(noised, os.path.join('results/pusnet', test_data_name, super_dirs[6], image_name))
                save_image(denoised, os.path.join('results/pusnet', test_data_name, super_dirs[7], image_name))

            ############### calculate metrics #################
            secret = secret.detach().cpu().numpy().squeeze() * 255
            np.clip(secret, 0, 255)
            secret_rev = secret_rev.detach().cpu().numpy().squeeze() * 255
            np.clip(secret_rev, 0, 255)
            
            cover = cover.detach().cpu().numpy().squeeze() * 255
            np.clip(cover, 0, 255)
            stego = stego.detach().cpu().numpy().squeeze() * 255
            np.clip(stego, 0, 255)
            
            noised = noised.detach().cpu().numpy().squeeze() * 255
            np.clip(noised, 0, 255)
            denoised = denoised.detach().cpu().numpy().squeeze() * 255
            np.clip(denoised, 0, 255)

            psnr_temp = calculate_psnr(cover, stego)
            S_psnr.append(psnr_temp)
            psnr_temp = calculate_psnr(secret, secret_rev)
            R_psnr.append(psnr_temp)
            psnr_temp = calculate_psnr(secret, noised)
            N_psnr.append(psnr_temp)
            psnr_temp = calculate_psnr(secret, denoised)
            DN_psnr.append(psnr_temp)

            mae_temp = calculate_mae(cover, stego)
            S_mae.append(mae_temp)
            mae_temp = calculate_mae(secret, secret_rev)
            R_mae.append(mae_temp)
            mae_temp = calculate_mae(secret, noised)
            N_mae.append(mae_temp)
            mae_temp = calculate_mae(secret, denoised)
            DN_mae.append(mae_temp)

            rmse_temp = calculate_rmse(cover, stego)
            S_rmse.append(rmse_temp)
            rmse_temp = calculate_rmse(secret, secret_rev)
            R_rmse.append(rmse_temp)
            rmse_temp = calculate_rmse(secret, noised)
            N_rmse.append(rmse_temp)
            rmse_temp = calculate_rmse(secret, denoised)
            DN_rmse.append(rmse_temp)

            ssim_temp = calculate_ssim(cover, stego)
            S_ssim.append(ssim_temp)
            ssim_temp = calculate_ssim(secret, secret_rev)
            R_ssim.append(ssim_temp)
            ssim_temp = calculate_ssim(secret, noised)
            N_ssim.append(ssim_temp)
            ssim_temp = calculate_ssim(secret, denoised)
            DN_ssim.append(ssim_temp)

        logger.info('testing, stego_avg_psnr: {:.2f}, secret_avg_psnr: {:.2f}, noise_avg_psnr: {:.2f}, denoise_avg_psnr: {:.2f}'.format(np.mean(S_psnr), np.mean(R_psnr), np.mean(N_psnr), np.mean(DN_psnr)))
        logger.info('testing, stego_avg_ssim: {:.4f}, secret_avg_ssim: {:.4f}, noise_avg_ssim: {:.4f}, denoise_avg_ssim: {:.4f}'.format(np.mean(S_ssim), np.mean(R_ssim), np.mean(N_ssim), np.mean(DN_ssim)))
        logger.info('testing, stego_avg_mae: {:.2f}, secret_avg_mae: {:.2f}, noise_avg_mae: {:.2f}, denoise_avg_mae: {:.2f}'.format(np.mean(S_mae), np.mean(R_mae), np.mean(N_mae), np.mean(DN_mae)))
        logger.info('testing, stego_avg_rmse: {:.2f}, secret_avg_rmse: {:.2f}, noise_avg_rmse: {:.2f}, denoise_avg_rmse: {:.2f}'.format(np.mean(S_rmse), np.mean(R_rmse), np.mean(N_rmse), np.mean(DN_rmse)))

else:
    secret_recover_loss = nn.MSELoss().to(device)
    stego_similarity_loss = nn.MSELoss().to(device)
    denoising_loss = nn.MSELoss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=c.lr, betas=(0.5, 0.999), eps=1e-6, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, c.weight_step, gamma=c.gamma)

    for epoch in range(c.epochs):
        epoch += 1
        s_loss = []
        r_loss = []
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

            secret = data[data.shape[0]//2:]
            cover = data[:data.shape[0]//2]
            clean = secret
            noised = noised_data[noised_data.shape[0]//2:]
            
            ################## forward ####################
            remove_adapter(model, sparse_mask)
            denoised = model(noised, None, 'denoising')
            insert_adapter(model, sparse_mask, model_hiding_seed)
            stego = model(secret, cover, 'hiding')
            insert_adapter(model, sparse_mask, model_recover_seed, is_sparse=False)
            secret_rev = model(stego, None, 'recover')

            ################### loss ######################
            S_loss = stego_similarity_loss(cover, stego)
            R_loss = secret_recover_loss(secret, secret_rev)
            DN_loss = denoising_loss(clean, denoised)
            loss =  c.pusnet_lambda_S * S_loss + c.pusnet_lambda_R * R_loss + c.pusnet_lambda_DN * DN_loss
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
            s_loss.append(S_loss.item())
            r_loss.append(R_loss.item())
            dn_loss.append(DN_loss.item())
            loss_history.append(loss.item())
            metric_monitor.update("S_loss", np.mean(np.array(s_loss)))
            metric_monitor.update("R_loss", np.mean(np.array(r_loss)))
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
                S_psnr = []
                R_psnr = []
                N_psnr = []
                DN_psnr = []
                for (data, noised_data) in test_loader:
                    data = data.to(device)
                    noised_data = noised_data.to(device)
                    
                    secret = data[data.shape[0]//2:]
                    cover = data[:data.shape[0]//2]
                    clean = secret
                    noised = noised_data[noised_data.shape[0]//2:]
            
                    ################## forward ####################
                    remove_adapter(model, sparse_mask)
                    denoised = model(noised, None, 'denoising')
                    insert_adapter(model, sparse_mask, model_hiding_seed)
                    stego = model(secret, cover, 'hiding')
                    insert_adapter(model, sparse_mask, model_recover_seed, is_sparse=False)
                    secret_rev = model(stego, None, 'recover')

                    ############### calculate psnr #################
                    secret = secret.detach().cpu().numpy().squeeze() * 255
                    np.clip(secret, 0, 255)
                    secret_rev = secret_rev.detach().cpu().numpy().squeeze() * 255
                    np.clip(secret_rev, 0, 255)
                    
                    cover = cover.detach().cpu().numpy().squeeze() * 255
                    np.clip(cover, 0, 255)
                    stego = stego.detach().cpu().numpy().squeeze() * 255
                    np.clip(stego, 0, 255)
                    
                    noised = noised.detach().cpu().numpy().squeeze() * 255
                    np.clip(noised, 0, 255)
                    denoised = denoised.detach().cpu().numpy().squeeze() * 255
                    np.clip(denoised, 0, 255)

                    psnr_temp = calculate_psnr(cover, stego)
                    S_psnr.append(psnr_temp)
                    psnr_temp = calculate_psnr(secret, secret_rev)
                    R_psnr.append(psnr_temp)
                    psnr_temp = calculate_psnr(secret, noised)
                    N_psnr.append(psnr_temp)
                    psnr_temp = calculate_psnr(secret, denoised)
                    DN_psnr.append(psnr_temp)
    
                # logger.info('epoch: {}, training, T_loss: {:.5f}'.format(epoch, epoch_losses))
                logger.info('epoch: {}, testing, stego_avg_psnr: {:.2f}, secret_avg_psnr: {:.2f}, noise_avg_psnr: {:.2f}, denoise_avg_psnr: {:.2f}'.format(epoch, np.mean(S_psnr), np.mean(R_psnr), np.mean(N_psnr), np.mean(DN_psnr)))

        if epoch % c.save_freq == 0 and epoch >= (c.save_start_epoch):
            remove_adapter(model, sparse_mask)
            model_save_dir = os.path.join(c.model_save_dir, 'pusnet-'+ str(c.sparse_ratio))
            mkdirs(model_save_dir)
            torch.save(model.state_dict(), os.path.join(model_save_dir, 'checkpoint_%.4i' % epoch + '.pt'))
            
        scheduler.step()


    
