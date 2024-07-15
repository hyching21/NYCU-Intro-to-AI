import os
import cv2

def load_images(data_path):
    """
    Load all Images in the folder and transfer a list of tuples. 
    The first element is the numpy array of shape (m, n) representing the image.
    (remember to resize and convert the parking space images to 36 x 16 grayscale images.) 
    The second element is its classification (1 or 0)
      Parameters:
        dataPath: The folder path.
      Returns:
        dataset: The list of tuples.
    """
    # Begin your code (Part 1)
    dataset = []
    labels = {"car": 1, "non-car": 0}
    for folder, label in labels.items():
        folder_path = os.path.join(data_path, folder)
        for file in os.listdir(folder_path):
            img = cv2.imread(os.path.join(folder_path, file))
            resized = cv2.resize(img,(36,16)) # resize to 36 x 16
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            dataset.append((gray, label))
    #raise NotImplementedError("To be implemented")
    # End your code (Part 1)
    return dataset

