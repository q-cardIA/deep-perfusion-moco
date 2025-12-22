"""
We train the affine registration network again, now with different input data.
Data: 
- Moving: result affine registration model 1
- Fixed: PCA of result affine registration model 1
"""

import faulthandler; faulthandler.enable()
# Import other modules
import Data_module

# Import libraries
import os
import time
import yaml
import torch
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import monai
from monai.utils import set_determinism
from monai.config import print_config, USE_COMPILED
from monai.networks.nets import GlobalNet
from monai.networks.blocks import Warp
from monai.losses import ssim_loss, BendingEnergyLoss
from monai.losses import LocalNormalizedCrossCorrelationLoss, GlobalMutualInformationLoss

os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

torch.manual_seed(42)
np.random.seed(42)
monai.utils.set_determinism(42)

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
def loss_function(config, fixed_image, predicted_image, ddf):
    """
    Function calculates loss function based on elements specified in config. 
    ncc: normalized cross correlation
    gmi: global mutual information
    regularization: bending energy on the predicted ddf

    return: loss_value, NCC_value, regularization_value
    """
    lam_ncc, lam_gmi, lam_reg = config["loss"]["LNCC"]["lam"], config["loss"]["GMI"]["lam"], config["loss"]["BE"]["lam"]
    if lam_ncc > 0:
        ncc_loss = LocalNormalizedCrossCorrelationLoss(spatial_dims=2, kernel_size=127)
    if lam_gmi > 0:
        gmi_loss = GlobalMutualInformationLoss()
    if lam_reg > 0:
        regularization = BendingEnergyLoss()
    
    ncc = ncc_loss(predicted_image, fixed_image) if lam_ncc > 0 else 0.0
    gmi = gmi_loss(predicted_image, fixed_image) if lam_gmi > 0 else 0.0
    reg = regularization(ddf) if lam_reg > 0 else 0.0

    total_loss = lam_ncc*ncc + lam_gmi*gmi + lam_reg*reg
    return total_loss, ncc.item(), reg

def save_images(type_training, epoch, fixed, moving, pred_image, ddf, warp_layer, save_model_path):
    # Monitor deformation (define square and apply ddf)
    array_mon = np.zeros((config["dataloader"][type_training]["batch_size"],1,128, 128))
    array_size_mon, square_size_mon = 128, 40
    for ind_mon in range(config["dataloader"][type_training]["batch_size"]):
        image_mon = np.zeros((128, 128))
        start_index_mon = (array_size_mon - square_size_mon) // 2
        end_index_mon = start_index_mon + square_size_mon
        image_mon[start_index_mon:end_index_mon, start_index_mon:end_index_mon] = 1
        array_mon[ind_mon,0,:,:] = image_mon
    image_pred_mon = warp_layer(torch.Tensor(array_mon).to(device), ddf)

    nr_rows = 8
    fig, ax = plt.subplots(nr_rows, 4, figsize=(12, 3 * nr_rows))
    for ind_mon in range(nr_rows):
        i = ind_mon
        ax[i, 0].set_title("Fixed image")
        ax[i, 0].axis("off")
        if type_training == "train":
            ax[i, 0].imshow(fixed[ind_mon][0], cmap="gray")
        else:
            ax[i, 0].imshow(fixed[ind_mon], cmap="gray")

        ax[i, 1].set_title("Moving image")
        ax[i, 1].axis("off")
        if type_training == "train":
            ax[i, 1].imshow(moving[ind_mon][0], cmap="gray")
        else:
            ax[i, 1].imshow(moving[ind_mon], cmap="gray")

        ax[i, 2].set_title("Predicted image")
        ax[i, 2].axis("off")
        if type_training == "train":
            ax[i, 2].imshow(pred_image[ind_mon][0], cmap="gray")
        else:
            ax[i, 2].imshow(pred_image[ind_mon], cmap="gray")
        
        ax[i, 3].set_title("Deformation")
        ax[i, 3].axis("off")
        ax[i, 3].imshow(image_pred_mon.detach().cpu().numpy()[ind_mon,0,:,:])
    plt.tight_layout()
    plt.savefig('{}{}_monitor_images_epoch{}.png'.format(save_model_path, type_training, epoch))          
    plt.clf()   
    plt.close()       

def main(config, save_model_path, loss_func="LNCC"):
    print_config()
    set_determinism(42)
        
    # Call dataloaders (train + valid)
    DL = Data_module.DataModule(config)
    train_loader = DL.train_dataloader()
    valid_loader = DL.valid_dataloader()
    
    # Model/training settings
    model = GlobalNet(image_size=(128,128), spatial_dims=2, in_channels=2, num_channel_initial=config["model"]["num_channel_initial"], depth=config["model"]["depth"]).to(device) 
    
    # Define loss function
    metrics = {"GMI" :GlobalMutualInformationLoss(),
               "SSIM":ssim_loss.SSIMLoss(spatial_dims=2)}
    monitor_metrics = {"training": {"LNCC":[], "GMI":[], "SSIM":[], "reg":[]},
                       "validation": {"LNCC":[], "GMI":[], "SSIM":[], "reg":[]}}

    if USE_COMPILED:
        warp_layer = Warp(3, "border").to(device)
    else:
        warp_layer = Warp("bilinear", "border").to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), float(config["training"]["lr"]))

    # ---- Training loop ----
    max_epochs = config["training"]["num_epochs"]

    # Initialize lists for monitoring
    epoch_loss_values = []
    epoch_loss_values_validation = []

    # Define early stopping and save_best_model
    early_stopping = EarlyStopper(patience=5, min_delta=0.05)
    best_loss = 1000

    for epoch in range(max_epochs):
        st = time.time()
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()

        # Initialize variables
        epoch_loss, step, step_valid = 0, 0, 0
        ssim_train_val, valid_epoch_loss, ssim_valid_val = 0, 0, 0
        lncc_train_val, lncc_valid_val = 0, 0
        gmi_train_val, gmi_valid_val = 0, 0
        reg_train_val, reg_valid_val = 0,0

        for batch_data in train_loader:
            step += 1
            if "00" in str(step):
                print("step {}/{}".format(step, len(train_loader)))
            optimizer.zero_grad()

            moving_training = batch_data["image"].to(torch.float).to(device).unsqueeze(1)
            fixed_training = batch_data["reference"].to(torch.float).to(device).unsqueeze(1)

            ddf = model(torch.cat((moving_training, fixed_training), dim=1))             # get deformation field from model = (bs, 2, 128, 128)
            pred_image_training = warp_layer(moving_training, ddf)                       # apply deformation field to moving image
            
            if step == 10:
                save_images("train", epoch, fixed_training.detach().cpu().numpy()[:8], moving_training.detach().cpu().numpy()[:8], pred_image_training.detach().cpu().numpy()[:8], ddf, warp_layer, save_model_path)

            # Calculate the multi-element loss 
            loss, NCC_train, reg_train = loss_function(config, fixed_training, pred_image_training, ddf) # calculate loss beween fixed and moved
            loss.backward()                                           
            optimizer.step()                
            epoch_loss += loss.item()
            
            # Monitor LNCC/GMI/SSIM during training
            ssim_train_val += metrics["SSIM"](pred_image_training, fixed_training).item()
            lncc_train_val += NCC_train
            reg_train_val += reg_train

        # Store training loss
        epoch_loss /= step                                      
        epoch_loss_values.append(epoch_loss)                    

        # Store LNCC/GMI/SSIM values for monitoring training
        monitor_metrics["training"]["SSIM"].append(ssim_train_val/step)
        monitor_metrics["training"]["LNCC"].append(lncc_train_val/step)
        monitor_metrics["training"]["GMI"].append(gmi_train_val/step)
        monitor_metrics["training"]["reg"].append(reg_train_val/step)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        # Visualize images
        moving_monitor, fixed_monitor, moved_monitor, square_monitor = [], [], [], []
        
        # ---- Validation loop ----   
        with torch.no_grad():  
            model.eval()
            # Calculate validation loss
            for batch_data in valid_loader:
                step_valid += 1

                moving_valid = batch_data["image"].detach().to(torch.float).to(device).unsqueeze(1)
                fixed_valid = batch_data["reference"].detach().to(torch.float).to(device).unsqueeze(1)
                
                ddf_valid = model(torch.cat((moving_valid, fixed_valid), dim=1))      
                pred_image_valid = warp_layer(moving_valid, ddf_valid)

                # Visualize squared deformation
                array_mon = np.zeros((config["dataloader"]["valid"]["batch_size"],1,128, 128))
                array_size_mon, square_size_mon = 128, 40
                for ind_mon in range(config["dataloader"]["valid"]["batch_size"]):
                    image_mon = np.zeros((128, 128))
                    start_index_mon = (array_size_mon - square_size_mon) // 2
                    end_index_mon = start_index_mon + square_size_mon
                    image_mon[start_index_mon:end_index_mon, start_index_mon:end_index_mon] = 1
                    array_mon[ind_mon,0,:,:] = image_mon
                image_pred_mon = warp_layer(torch.Tensor(array_mon).to(device), ddf_valid)

                moving_monitor.append(moving_valid.detach().cpu().numpy()[0,0])
                fixed_monitor.append(fixed_valid.detach().cpu().numpy()[0,0])
                moved_monitor.append(pred_image_valid.detach().cpu().numpy()[0,0])
                square_monitor.append(image_pred_mon.detach().cpu().numpy()[0,0])
            
                # Calculate the multi-element loss 
                valid_loss, NCC_valid, reg_valid = loss_function(config, fixed_valid, pred_image_valid, ddf_valid)
                valid_epoch_loss += valid_loss
                
                # Monitor LNCC/GMI/SSIM during validation
                ssim_valid_val += metrics["SSIM"](pred_image_valid, fixed_valid).item()
                lncc_valid_val += NCC_valid
                reg_valid_val += reg_valid
            
        # Save visualizations
        save_images("valid", epoch, fixed_monitor[:8], moving_monitor[:8], moved_monitor[:8], ddf_valid, warp_layer, save_model_path)

        # Store validation loss
        valid_epoch_loss = valid_epoch_loss/step_valid
        epoch_loss_values_validation.append(valid_epoch_loss.detach().cpu().numpy())

        # Store LNCC/GMI/SSIM values for monitoring
        monitor_metrics["validation"]["SSIM"].append(ssim_valid_val/step_valid)
        monitor_metrics["validation"]["LNCC"].append(lncc_valid_val/step_valid)
        monitor_metrics["validation"]["GMI"].append(gmi_valid_val/step_valid)
        monitor_metrics["validation"]["reg"].append(reg_valid_val/step_valid)
        print(f"epoch {epoch + 1} average validation loss: {valid_epoch_loss:.4f}")
        print("Time epoch: ",time.time()-st)
        print("monitor metrics",monitor_metrics)

        # Save best model
        if epoch > 10 and best_loss < valid_epoch_loss.detach().cpu().numpy():
            torch_input = torch.randn(1,2,128,128).to(device)
            torch.onnx.export(model, torch_input, "{}best_model.onnx".format(save_model_path), verbose=False, input_names=["input"], output_names=["output"], export_params=True, dynamic_axes={"input":{0:"batch_size"}, "output":{0:"batch_size"}})
            best_loss = valid_epoch_loss.detach().cpu().numpy()
        # Early stopping
        if early_stopping.early_stop(valid_epoch_loss): 
            print("We are at epoch:", epoch)
            break

    visualization_images = [fixed_training, moving_training, pred_image_training.detach().cpu().numpy(), fixed_valid, moving_valid, pred_image_valid.detach().cpu().numpy()]
    return epoch_loss_values, epoch_loss_values_validation, monitor_metrics, visualization_images

if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="Path to config file", type=str)
    parser.add_argument("--save_path", help="Path to save output", type=str)
    parser.add_argument("--config_nr", help="Nr of config file", type=int)
    parser.add_argument("--plot_nr", help="Nr of plot file", type=int)
    args = parser.parse_args()

    # Define test number
    test_number = args.config_nr
    print("I'm working on config_{}".format(test_number))
    # Define configurations for dataloader 
    config_path = Path("{}/config_{}.yaml".format(args.config_path, test_number)) 
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)

    save_image_path = "{}/Plots_{}/".format(args.save_path, args.plot_nr)

    # Call main function
    epoch_loss_values, epoch_loss_values_validation, monitor_metrics, visualization_images = main(config, save_image_path)
