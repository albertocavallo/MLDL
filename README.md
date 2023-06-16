# Resource Constraint Recyclable Waste Segmentation
This repository include the setup for Resource Constraint Recyclable Waste Segmentation. It provides the codes for training ENet, BiSeNet and ICNet on the ReSort dataset for binary and instance class segmentation.

## Usage

### Data Preparation
* Download the [ResortIT dataset.](https://drive.google.com/file/d/14ThGc53okYC61AnTXFAofiYYY8PTZYtl/view?usp=share_link).
* Unzip the ```dataset.zip``` into the project folder.
* Modify the root path of the dataset by changing ```__C.DATA.DATA_PATH``` in ```config.py```.

### Requirements installation
* Run the following command to install all the requirements:  
       `pip install -r requirements.txt`

### Training
* Move to the folder of the desired segmentation model among ENet, BiSeNet, or ICNet, for binary segmentation. execute the related train file, for example for ENet run: `train_E.py`
* To use the instance segmentation modify:
- the ```__C.DATA.NUM_CLASSES``` in ```config.py``` from `2` to `5`.
- Comment line 45 in ```resortit.py```

### For data augmentation
* To apply data augmentation to the dataset run the ```data_augmentation.py``` file.
* To remove data augmentation and return the original dataset run the ```remove_data.py``` file.
