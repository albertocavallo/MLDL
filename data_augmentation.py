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
        iaa.Fliplr(0.5),  # horizontally flip 50% of images
        iaa.GaussianBlur(sigma=(0, 1.0)),  # apply Gaussian blur with sigma between 0 and 1.0
        iaa.Affine(rotate=(0, 90))  # rotate images between -45 and 45 degrees
    ])
    x=0
    for filename in os.listdir(input_folder):
        x+=1
        if(x==2): exit
        if filename.endswith('.jpg') or filename.endswith('.png'):  
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename.split('.')[0])
            image = cv2.imread(input_path)
            for i in range(augmentation_factor):
                augmented_image = seq.augment_image(image)
                augmented_image=zoom_image(augmented_image, 1.5)
                output_filename = f'{output_path}_augmented_{i}.jpg'
                cv2.imwrite(output_filename, augmented_image)

                print(f'Saved augmented image: {output_filename}')

    print('Data augmentation, train images completed.')

    for filename in os.listdir(input_folder_l):
        if filename.endswith('.jpg') or filename.endswith('.png'): 
            input_path = os.path.join(input_folder_l, filename)
            output_path = os.path.join(output_folder_l, filename.split('.')[0])  

            image = cv2.imread(input_path)

            for i in range(augmentation_factor):
                augmented_image = seq.augment_image(image)
                augmented_image=zoom_image(augmented_image, 1.5)
                augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2GRAY)

                output_filename = f'{output_path}_augmented_{i}.jpg'
                cv2.imwrite(output_filename, augmented_image)

    print('Data augmentation, train labels completed.')


input_folder = '/content/drive/MyDrive/dataset/train'
output_folder = '/content/drive/MyDrive/dataset/train'
input_folder_l = '/content/drive/MyDrive/dataset/labels/train'
output_folder_l = '/content/drive/MyDrive/dataset/labels/train'
augmentation_factor = 1  # Generate augmentation_factor images for each input image

augment_images(input_folder, output_folder,input_folder_l, output_folder_l, augmentation_factor)
