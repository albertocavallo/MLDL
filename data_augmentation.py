import os
from imgaug import augmenters as iaa
import cv2

def augment_images(input_folder, output_folder, augmentation_factor):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Define augmentation sequence
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # horizontally flip 50% of images
        iaa.GaussianBlur(sigma=(0, 1.0)),  # apply Gaussian blur with sigma between 0 and 1.0
        iaa.Affine(rotate=(-45, 45))  # rotate images between -45 and 45 degrees
        # Add more augmentation techniques as needed
    ])

    # Iterate over input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # consider only image files
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename.split('.')[0])  # remove file extension

            # Read the image
            image = cv2.imread(input_path)

            # Apply data augmentation multiple times
            for i in range(augmentation_factor):
                augmented_image = seq.augment_image(image)

                # Save augmented image
                output_filename = f'{output_path}_augmented_{i}.jpg'
                cv2.imwrite(output_filename, augmented_image)

                print(f'Saved augmented image: {output_filename}')

    print('Data augmentation completed.')


# Example usage
input_folder = '/content/drive/MyDrive/dataset/train'
output_folder = '/content/drive/MyDrive/dataset/train'
augmentation_factor = 1  # Generate 5 augmented images for each input image

augment_images(input_folder, output_folder, augmentation_factor)
