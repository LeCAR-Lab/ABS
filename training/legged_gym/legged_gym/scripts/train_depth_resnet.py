import numpy as np
import os
from datetime import datetime

import torch
import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import copy

import torchvision
from torchvision import models
from tqdm import tqdm
import cv2

class CustomDataset(Dataset):
    def __init__(self, all_dataset):
        self.data = list(all_dataset.values())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, lidar_data = self.data[idx]
        image_tensor = torch.from_numpy(image).float()  # Assuming image is a numpy array
        lidar_tensor = torch.from_numpy(np.array(lidar_data)).float()  # Assuming lidar_data is a numpy array
        return image_tensor, lidar_tensor


class ResNetModel(torch.nn.Module):
    def __init__(self, resnet_type):
        super(ResNetModel, self).__init__()
        if resnet_type == 'resnet18':
            self.resnet = models.resnet18(weights='IMAGENET1K_V1')
            print("Using resnet18")

        elif resnet_type == 'resnet34':
            self.resnet = models.resnet34(weights='IMAGENET1K_V1')
            print("Using resnet34")
        else:
            raise NotImplementedError
        self.num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Linear(self.num_ftrs, 11)

    def forward(self, x):
        x_1 = self.resnet(x)
        return x_1

def train(args):
    datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_folder = '../depth_logs/{}-{}-{}'.format(datetime_now, args.resnet_type, args.exp_name)
    tensorboard_writer = SummaryWriter(log_dir=log_folder)

    
    # ============================== load data ==============================
    print("==============================loading data ==============================")
    data_folder = "../depth_data/rec_cam/"
    all_test_folder = os.listdir(data_folder)
    all_test_folder.sort()
    print(all_test_folder)
    all_dataset = {}

    if args.leftright_augmentation:
        print("Using left-right image augmentation, will double the number of data.")
    for folder_this_test in all_test_folder:
        data_folder_for_this_test = data_folder + folder_this_test
        with open(data_folder_for_this_test+ '/label.pkl', 'rb') as f:
            _label = pickle.load(f)
            print("Loaded label for {}".format(folder_this_test))
            print("Number of data for this test: {}".format(len(_label)))
        all_image = os.listdir(data_folder_for_this_test)
        all_image = [x for x in all_image if x.endswith('.npy')]
        for image_name in all_image:
            image_idx = image_name.split('.')[0]
            assert image_idx in _label.keys()
            image_path = data_folder_for_this_test + '/' + image_name

            image = np.load(image_path, allow_pickle=True)
            all_dataset["{}_{}".format(folder_this_test, image_idx)] = [np.log2(image), np.log2(_label[image_idx])]
      
        print("Total number of data: {}".format(len(all_dataset)))
            
    print("Total number of data: {}".format(len(all_dataset)))
    # Create a custom dataset instance
    dataset = CustomDataset(all_dataset)

    # Determine the size of the test set
    test_size = int(0.1 * len(dataset))
    train_size = len(dataset) - test_size

    # Split the dataset into training and test sets
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoader instances for training and test sets
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize the model and other necessary components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = ResNetModel(args.resnet_type).to(device)  # Move the model to the GPU
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    criterion = torch.nn.MSELoss()  # Mean Squared Error (MSE) loss for regression task
    os.makedirs(log_folder, exist_ok=True)
    # Training loop
    DEBUG_IMAGE = False
    for epoch in range(args.num_epochs):
        model.train()
        for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader)):
            apply_left_right_augmentation = np.random.uniform()<0.5 and args.leftright_augmentation
            if DEBUG_IMAGE:
                cv2.imshow('Depth', inputs[0].numpy()/np.log2(6))

            if apply_left_right_augmentation:
                inputs = flipped_inputs = torch.flip(inputs, dims=[-1]) # Reverse the tensor along the last dimension (columns)
                targets = flipped_targets = torch.flip(targets, dims=[-1]) # Reverse the tensor along the last dimension (columns)    
            
            # apply_gaussian_blur_augmentation
            apply_gaussian_blur_augmentation = np.random.uniform()<args.gaussian_blur_augmentation_prob and args.gaussian_blur_augmentation
            if apply_gaussian_blur_augmentation:
                kernel_size = args.gaussian_blur_augmentation_kernel_size
                # Define the GaussianBlur transform
                blur_transform = torchvision.transforms.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0))

                # Apply Gaussian blur to the reshaped images
                inputs_ = blur_transform(inputs)
                inputs = inputs_

            # apply_noise_augmentation 
            apply_noise_augmentation = np.random.uniform()<args.noise_augmentation_prob and args.noise_augmentation
            if apply_noise_augmentation:
                noise_std = np.random.uniform()*args.noise_augmentation_std_max
                noise = torch.randn_like(inputs) * noise_std + args.noise_augmentation_mean
                inputs_ = inputs + noise
                inputs = inputs_
                
            # apply_random_erasing_augmentation
            apply_random_erasing_augmentation = np.random.uniform()<args.random_erasing_augmentation_prob and args.random_erasing_augmentation
            if apply_random_erasing_augmentation:
                x = np.random.randint(0, inputs.size(2) - 5)  # Adjust the range as needed
                y = np.random.randint(0, inputs.size(1) - 5)  # Adjust the range as needed

                h = np.random.randint(args.random_erasing_area_min, args.random_erasing_area_max) 
                w = np.random.randint(args.random_erasing_area_min, args.random_erasing_area_max)   # Adjust the range as needed

                random_erased_value = np.random.choice([np.log2(0.25), np.log2(6)])  # Adjust the range as needed
                inputs_ = torchvision.transforms.functional.erase(inputs, x, y, h, w, v=random_erased_value)
                inputs = inputs_
            if DEBUG_IMAGE:
                cv2.imshow('Depth_augmented', inputs[0].numpy()/np.log2(6))
                cv2.waitKey(1)
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to the GPU
            inputs = inputs.unsqueeze(1).repeat(1, 3, 1, 1)
            optimizer.zero_grad()
            outputs = model(inputs)  
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Print training statistics
            if batch_idx % args.log_interval == 0:
                print('Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, args.num_epochs, batch_idx * len(inputs), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                tensorboard_writer.add_scalar('Train Loss', loss.item(), epoch * len(train_loader)+ batch_idx)
        
        # Validation phaseblurred_reshaped_images
        model.eval()  # Set the model to evaluation mode
        test_loss = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                inputs = inputs.unsqueeze(1).repeat(1, 3, 1, 1)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()

        # Calculate average test loss
        test_loss /= len(test_loader)

        # Log test set loss
        print('Test set: Average Loss: {:.6f}'.format(test_loss))
        tensorboard_writer.add_scalar('Test Loss', test_loss, epoch)

        # Save the entire model
        if epoch % args.save_interval == 0:
            model_file_path = os.path.join(log_folder, 'depth_lidar_model_{}_{}.pt'.format(datetime_now, epoch))
            model_to_save = copy.deepcopy(model).to('cpu')
            traced_script_module = torch.jit.script(model_to_save)
            traced_script_module.save(model_file_path)
            # torch.save(model_to_save, model_file_path)
            print(f'Model saved at: {model_file_path}')

    # Close the TensorBoard writer
    tensorboard_writer.close()
        
    #torch.save(model.state_dict(), '../depth_logs/trained_model.pth')  

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Training script for LiDAR prediction using depth images.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for the optimizer.')
    parser.add_argument('--num_epochs', type=int, default=302, help='Number of training epochs.')
    parser.add_argument('--log_interval', type=int, default=50, help='Interval for logging training statistics.')
    parser.add_argument('--save_interval', type=int, default=50, help='Interval for saving the trained model.')
    parser.add_argument('--batch_size', type=int, default=320, help='Batch size for training the model.')
    parser.add_argument('--leftright_augmentation', type=bool, default=True, help='Whether to use left-right image augmentation.')
    parser.add_argument('--exp_name', type=str, default='', help='Experiment name.')
    parser.add_argument('--resnet_type', type=str, default='resnet18', help='ResNet model.')
    # Noise augmentation
    parser.add_argument('--noise_augmentation', type=bool, default=True, help='Whether to use noise augmentation.')
    parser.add_argument('--noise_augmentation_std_max', type=float, default=0.3, help='Standard deviation of the noise augmentation.')
    parser.add_argument('--noise_augmentation_mean', type=float, default=0.0, help='Mean of the noise augmentation.')
    parser.add_argument('--noise_augmentation_prob', type=float, default=0.5, help='Probability of the noise augmentation.')
    # Gaussian blur augmentation
    parser.add_argument('--gaussian_blur_augmentation', type=bool, default=True, help='Whether to use Gaussian blur augmentation.')
    parser.add_argument('--gaussian_blur_augmentation_kernel_size', type=int, default=5, help='Kernel size of the Gaussian blur augmentation.')
    parser.add_argument('--gaussian_blur_augmentation_prob', type=float, default=0.5, help='Probability of the Gaussian blur augmentation.')
    # Random erasing augmentation
    parser.add_argument('--random_erasing_augmentation', type=bool, default=True, help='Whether to use random erasing augmentation.')
    parser.add_argument('--random_erasing_augmentation_prob', type=float, default=0.5, help='Probability of the random erasing augmentation.')
    parser.add_argument('--random_erasing_area_min', type=int, default=5, help='Minimum area of the random erasing augmentation.')
    parser.add_argument('--random_erasing_area_max', type=int, default=10, help='Maximum area of the random erasing augmentation.')


    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    train(args)
