from dataloader_definition import EnsembleDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import os

#test data loader image creation
def test_dataloader_image():
    # Path to the data directory
    root_dir = "../data/single_objects_training"

    # Transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    # Create the dataset
    dataset = EnsembleDataset(root_dir=root_dir, transform=transform)

    # Create the DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Test the DataLoader by selecting the first 
    for batch_idx, (ensemble_images, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        print(f"Batch Shape: {ensemble_images.shape}")  # (batch_size, 3, 244, 244)
        print(f"Batch Labels: {labels}")  # (batch_size,)

        # Visualize the first image in the batch
        ensemble_image = ensemble_images[0]  # Select the first image
        ensemble_image = ensemble_image.permute(1, 2, 0).numpy()  # Convert to (H, W, C) for plotting
        ensemble_image = ensemble_image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # Denormalize

        plt.imshow(ensemble_image)
        ratio_conditions = ['10:2', '9:3', '8:4', '7:5']
        plt.title(f"Label: {labels[0]}, ratio condition: {ratio_conditions[labels[0]]}")
        plt.axis("off")
        plt.show()

        break  # Test only the first batch

#test dataloader dataframe condition creation
def test_dataloader_conditions():
    # Path to the data directory
    root_dir = "../data/single_objects_training"

    # Transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    # Create the dataset
    dataset = EnsembleDataset(root_dir=root_dir, transform=transform)

    # Create the DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Test the DataLoader by running all the batches and checking the proportion of each condition
    proportion = []
    proportion_binarized = []
    for batch_idx, (ensemble_images, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        #print(f"Batch Shape: {ensemble_images.shape}")  # (batch_size, 3, 244, 244)
        #print(f"Batch Labels: {labels}")  # (batch_size,)
        lab = labels.numpy().tolist()
        lab = [0 if int(x) == 0 else 1 for x in lab]
        #print(f"Batch Labels: {lab}")  # (batch_size,)
        proportion.extend(labels.numpy().tolist())
        proportion_binarized.extend(lab)

    #Calculate the proportion of times a 0, 1, 2, 3 appears
    print('-----------------------------------------------------')
    print('Proportion 10:2', proportion.count(0)/len(proportion))
    print('Proportion 9:3', proportion.count(1)/len(proportion))
    print('Proportion 8:4', proportion.count(2)/len(proportion))
    print('Proportion 7:5', proportion.count(3)/len(proportion))

    #Calculate the proportion of times a catch trials and normal trials appear
    print('-----------------------------------------------------')
    print('Proportion catch trial', proportion_binarized.count(0)/len(proportion_binarized))
    print('Proportion normal trial', proportion_binarized.count(1)/len(proportion_binarized))

if __name__ == "__main__":

    #Test how images looks like
    #test_dataloader_image()

    #Test the number of times each condition appears
    test_dataloader_conditions()

    #the number of batches is the product of the number items per class divided by the bacth size. 
    #In thuis case we have 3 different classes (animacy, shape and orientation) with two classes




