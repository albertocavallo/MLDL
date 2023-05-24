import os

def delete_augmented_images(folder_path1, folder_path2):
    for filename in os.listdir(folder_path1):
        if "augmented" in filename:
            file_path = os.path.join(folder_path1, filename)
            os.remove(file_path)
    print("Images deleted")
    for filename in os.listdir(folder_path2):
        if "augmented" in filename:
            file_path = os.path.join(folder_path2, filename)
            os.remove(file_path)
    print("Labels deleted")
# Example usage
folder_path1 = "/content/drive/MyDrive/dataset/train"  # Replace with the actual folder path
folder_path2 = "/content/drive/MyDrive/dataset/labels/train"  # Replace with the actual folder path

delete_augmented_images(folder_path1, folder_path2)
