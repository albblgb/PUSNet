import torch
import torch.nn as nn
import config as c
from utils.image import quantization

class pusnet(nn.Module): 
    def __init__(self) -> None:
        super(pusnet, self).__init__()
        
        self.in_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1 ,padding=1), 
            nn.GroupNorm(4, 64),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1 ,padding=1), 
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1 ,padding=1), 
        )
        
        self.pre_layers_1 = nn.Sequential(
            nn.GroupNorm(16, 256),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1 ,padding=1), 
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1 ,padding=1), 
        )    
        
        self.pre_layers_2 = nn.Sequential(
            nn.GroupNorm(16, 256),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1 ,padding=1), 
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1 ,padding=1), 
        )  
        
        self.pre_layers_3 = nn.Sequential(
            nn.GroupNorm(16, 256),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1 ,padding=1), 
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1 ,padding=1), 
            nn.GroupNorm(16, 256),
            nn.LeakyReLU(),
        ) 

        self.mid_secret_path = nn.Sequential( 
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1 ,padding=1),
        )

        self.mid_cover_path = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1 ,padding=1),
        )

        self.after_concat_layers_1 = nn.Sequential(
            nn.GroupNorm(16, 256),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1 ,padding=1), 
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1 ,padding=1), 
        ) 
        
        self.after_concat_layers_2 = nn.Sequential(
            nn.GroupNorm(16, 256),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1 ,padding=1), 
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1 ,padding=1), 
        ) 
        
        self.after_concat_layers_3 = nn.Sequential(
            nn.GroupNorm(16, 256),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1 ,padding=1), 
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1 ,padding=1), 
        ) 
        
        self.output_layer = nn.Sequential(
            nn.GroupNorm(16, 256),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1 ,padding=1), 
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1 ,padding=1), 
            nn.GroupNorm(4, 64),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1 ,padding=1),
        )

        self.sigmoid = nn.Sigmoid()


    def forward(self, input_1, input_2, task):
        if task == 'hiding':
            secret = input_1
            cover = input_2

            x_s = self.in_layers(secret)
            x_s = self.pre_layers_1(x_s) + x_s
            x_s = self.pre_layers_2(x_s) + x_s
            x_s = self.pre_layers_3(x_s) + x_s
            
            x_c = self.in_layers(cover)
            x_c = self.pre_layers_1(x_c) + x_c
            x_c = self.pre_layers_2(x_c) + x_c
            x_c = self.pre_layers_3(x_c) + x_c
            
            secret_feature = self.mid_secret_path(x_s)
            cover_feature = self.mid_cover_path(x_c)
            concat_feature = torch.cat((secret_feature, cover_feature), dim=1)

            x = self.after_concat_layers_1(concat_feature)
            x = self.after_concat_layers_2(x) + x
            x = self.after_concat_layers_3(x) + x
            
            stego = self.output_layer(x)
            if c.mode == 'test':
                stego = quantization(stego)
            # stego = self.sigmoid(stego)
            out = stego
        
        elif task == 'recover':
            stego = input_1

            x = self.in_layers(stego)
            x = self.pre_layers_1(x) + x
            x = self.pre_layers_2(x) + x
            x = self.pre_layers_3(x) + x
            
            stego_feature_1 = self.mid_secret_path(x)
            stego_feature_2 = self.mid_cover_path(x)
            concat_feature = torch.cat((stego_feature_1, stego_feature_2), dim=1)

            x = self.after_concat_layers_1(concat_feature)
            x = self.after_concat_layers_2(x) + x
            x = self.after_concat_layers_3(x) + x
            
            secret_rev = self.output_layer(x)
            if c.mode == 'test':
                secret_rev = quantization(secret_rev)
            # secret_rev = self.sigmoid(secret_rev)
            out = secret_rev
        
        else: 

            noised_img = input_1
            x = self.in_layers(noised_img)
            x = self.pre_layers_1(x) + x
            x = self.pre_layers_2(x) + x
            x = self.pre_layers_3(x) + x
            
            noised_feature_1 = self.mid_secret_path(x)
            noised_feature_2 = self.mid_cover_path(x)
            concat_feature = torch.cat((noised_feature_1, noised_feature_2), dim=1)

            x = self.after_concat_layers_1(concat_feature)
            x = self.after_concat_layers_2(x) + x
            x = self.after_concat_layers_3(x) + x
            
            noise_residual = self.output_layer(x)
            denoised_img = noised_img - noise_residual
            if c.mode == 'test':
                denoised_img = quantization(denoised_img)
            out = denoised_img
            
        return out


