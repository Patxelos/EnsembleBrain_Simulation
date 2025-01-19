import random
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os 
import numpy as np

class EnsembleDataset(Dataset):
    def __init__(self, root_dir, image_size=(30, 30), canvas_size=(244, 244), transform=None):
        """
        Custom Dataset to generate ensemble images with class and subfolder predominance.
        Args:
            root_dir (str): Root directory containing class subfolders, each with `A` and `B`.
            image_size (tuple): Size to which each image is resized (width, height).
            canvas_size (tuple): Size of the final output image (width, height).
            transform (callable, optional): Transform to apply to the images.
        """
        self.root_dir = root_dir #path to where the folders to where animacy, shape and orientation subfolders are located containing single objects 
        self.image_size = image_size #single object size
        self.canvas_size = canvas_size #final image size, adjust to 
        self.transform = transform #tranformations to apply to the images (toTensor, Normalize)
        self.sampling_num = 10000 #number of samples to generate per class. Add a large number to generate arbitrarily infinite samples
        #this controls the numbers of batches to generate. 1000 * 3(classes) / 32(batch_size) = 93 batches

        # Define ensemble conditions and their probabilities
        self.conditions = {
            "10:2": (10, 2),
            "9:3": (9, 3),
            "8:4": (8, 4),
            "7:5": (7, 5),
        }
        self.condition_labels = {key: idx for idx, key in enumerate(self.conditions.keys())}
        self.probabilities = [0.5, 0.166, 0.166, 0.166]  # Probabilities for each condition

        # Load image paths grouped by class and subfolder (A and B)
        classes = os.listdir(root_dir)  # List of class subfolders
        self.classes = [cls for cls in classes if not '.DS_Store' in cls] #remove .DS_Store from the list
        
        self.image_paths = {cls: {"A": [], "B": []} for cls in self.classes}
        for cls in self.classes:
            for subfolder in ["A", "B"]:
                subfolder_path = os.path.join(root_dir, cls, subfolder)
                if os.path.exists(subfolder_path):
                    self.image_paths[cls][subfolder] = [
                        os.path.join(subfolder_path, img) for img in os.listdir(subfolder_path)
                        if img.endswith(('.jpg', '.png'))
                    ]

    #This function defines the number of samples created 
    def __len__(self):
        # Arbitrary large number for infinite sampling
        return len(self.classes) * self.sampling_num 

    #This function checks that single objects within ensembles do not overlap
    def is_overlapping(self, pos, placed_positions, margin=10):
        """
        Check if a new position overlaps with any existing positions.
        """
        for existing_pos in placed_positions:
            if (abs(pos[0] - existing_pos[0]) < self.image_size[0] + margin and
                abs(pos[1] - existing_pos[1]) < self.image_size[1] + margin):
                return True
        return False

    #This function scatters images on the canvas
    def scatter_images(self, canvas, image_paths, cls):
        """
        Scatter images onto the canvas while avoiding overlaps.
        """
        if cls == 'orientation':
            img_size = tuple(item + 20 for item in self.image_size)
        else:
            img_size = self.image_size
        
        # List of positions where images are placed
        placed_positions = []
        for path in image_paths:
            image = Image.open(path).convert("RGBA")
            image = image.resize(img_size)

            for _ in range(100):  # Try up to 100 times to place the image
                # Random position on the canvas
                pos_x = random.randint(0, self.canvas_size[0] - img_size[0])
                pos_y = random.randint(0, self.canvas_size[1] - img_size[1])

                if not self.is_overlapping((pos_x, pos_y), placed_positions):
                    placed_positions.append((pos_x, pos_y))
                    canvas.paste(image, (pos_x, pos_y))
                    break
    
    def __getitem__(self, idx):
        """
        Generate a single scattered ensemble image based on a randomly chosen condition and predominant subfolder.
        Returns:
            scatter_image (torch.Tensor): The generated 244x244 scatter image.
            label (int): Class label for the ensemble.
        """
        try:
            # Randomly select a class
            cls = random.choice(self.classes)
            #label = self.classes.index(cls)  # Class label is the index of the class folder

            # Randomly select a condition based on probabilities
            condition_name = random.choices(list(self.conditions.keys()), weights=self.probabilities, k=1)[0]
            majority_count, minority_count = self.conditions[condition_name]
            condition_label = self.condition_labels[condition_name]

            # Randomly decide which subfolder (A or B) is predominant
            predominant_class = random.choice(["A", "B"])
            minority_class = "A" if predominant_class == "B" else "B"

            # Randomly sample majority and minority images
            sampled_paths_majority = random.sample(self.image_paths[cls][predominant_class], majority_count)
            sampled_paths_minority = random.sample(self.image_paths[cls][minority_class], minority_count)
            combined = sampled_paths_majority + sampled_paths_minority
            combined = np.random.permutation(combined)

            # Create a blank canvas
            canvas = Image.new("RGB", self.canvas_size, (0, 0, 0))  # Black background

            # Scatter majority and minority images
            self.scatter_images(canvas, combined, cls)

            # Apply transformations
            if self.transform:
                scatter_image = self.transform(canvas)
            else:
                scatter_image = transforms.ToTensor()(canvas)

            return scatter_image, condition_label

        except Exception as e:
            print(f"Error occurred at index {idx}")
            print(f"Class: {cls}")
            print(f"Condition: {condition_name} (Majority: {majority_count}, Minority: {minority_count})")
            print(f"Predominant Class: {predominant_class}, Minority Subfolder: {minority_class}")
            print(f"Paths (Majority): {sampled_paths_majority if 'sampled_paths_majority' in locals() else 'Not generated'}")
            print(f"Paths (Minority): {sampled_paths_minority if 'sampled_paths_minority' in locals() else 'Not generated'}")
            print(f"Error Message: {e}")
            raise e