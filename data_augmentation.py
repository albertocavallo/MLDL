import os
from imgaug import augmenters as iaa
import cv2
import numpy as np

def zoom_image(image, scale):
    height, width = image.shape[:2]
    zoom_matrix = np.array([[scale, 0, (1 - scale) * width / 2],
                            [0, scale, (1 - scale) * height / 2],
                            [0, 0, 1]])
    return cv2.warpPerspective(image, zoom_matrix, (width, height))

def augment_images(input_folder, output_folder, input_folder_l, output_folder_l, augmentation_factor):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # Flip images horizontally with a 50% chance
        iaa.Flipud(0.5),  # Flip images vertically with a 50% chance
        iaa.Affine(rotate=(-20, 20)),  # Rotate images by -20 to +20 degrees
        iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)})
    ])
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):  
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename.split('.')[0])
            input_path_l = os.path.join(input_folder_l, filename)
            output_path_l = os.path.join(output_folder_l, filename.split('.')[0])  

            image = cv2.imread(input_path)
            label = cv2.imread(input_path_l)

            for i in range(augmentation_factor):
                # IMAGES
                augmented_image = seq.augment_image(image)
                output_filename = f'{output_path}_augmented_{i}.jpg'
                cv2.imwrite(output_filename, augmented_image)

                # LABELS
                augmented_label = seq.augment_image(label)
                augmented_label = cv2.cvtColor(augmented_label, cv2.COLOR_BGR2GRAY)
                output_filename_l = f'{output_path_l}_augmented_{i}.jpg'
                cv2.imwrite(output_filename_l, augmented_label)

    print('Data augmentation completed.')



input_folder = '/content/drive/MyDrive/dataset/train'
output_folder = '/content/drive/MyDrive/dataset/train'
input_folder_l = '/content/drive/MyDrive/dataset/labels/train'
output_folder_l = '/content/drive/MyDrive/dataset/labels/train'
augmentation_factor = 1  # Generate augmentation_factor images for each input image

augment_images(input_folder, output_folder,input_folder_l, output_folder_l, augmentation_factor)
