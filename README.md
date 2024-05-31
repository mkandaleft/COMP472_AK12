
# SmartClass A.I.ssistant Project

## Group Information
- **Team Name:** AK12


- **Team Members:**
  
 |   Name |   Student ID    |  Specialization  |
 |---|---|---|
| Ayoub Kchaou | 27894112 | Data Specialist |
| Mark Kandaleft | 40126013 |Training Specialist|
| Valentyna Tsilinchuk| 40046092 | Evaluation Specialist |


## Project Overview
Welcome to the SmartClass A.I.ssistant project. This project aims to develop a Deep Learning Convolutional Neural Network (CNN) using PyTorch to analyze images of students in a classroom or online meeting setting and categorize them into distinct states or activities.

## Project Parts
The project is divided into three parts:
1. **Data Collection, Cleaning, Labeling & Preliminary Analysis**
2. **CNN Model Development and Basic Evaluation**
3. **Bias Analysis, Model Refinement & Deep Evaluation**


---

## Part 1: Data Collection, Cleaning, Labeling & Preliminary Analysis

### Prerequisites

To successfully run the code, you will need the following libraries:

* pip
  ```sh
  pip install numpy
  pip install matplotlib
  pip install pillow
  pip install imagededup
  pip install scikit-learn
  pip install scikit-image
  pip install opencv-python
  pip install torchvision
  pip install seaborn
  ```

To read the images in the correct format, use LFS extension:

* git
  ```sh
  git lfs install
  git lfs pull
  ```
### Data
 * [FER Dataset](https://github.com/mkandaleft/COMP472_AK12/blob/main/data/FER_dataset/fer2013.csv) : the original dataset from kaggle containing 7 classes of emotions
 * [angry_happy_neutral_dataset](https://github.com/mkandaleft/COMP472_AK12/blob/main/data/extracted%20dataset/angry_happy_neutral.csv) : the extracted csv with only the emotions needed : angry,happy and neutral.
 * [angry (0)](https://github.com/mkandaleft/COMP472_AK12/tree/main/data/classes/0) : angry class final data.
 * [happy (1)](https://github.com/mkandaleft/COMP472_AK12/tree/main/data/classes/1) : happy class final data.
 * [neutral (2)](https://github.com/mkandaleft/COMP472_AK12/tree/main/data/classes/2) : neutral class final data.
 * [focused (3)](https://github.com/mkandaleft/COMP472_AK12/tree/main/data/classes/3) : focused class final data.
### Data Cleaning Programs

> ⚠️ **Warning:** *Certain programs below permanenty alter directories by deleting files.
> Make sure folder paths are specified correctly before running.*

* [split_emot.py](https://github.com/mkandaleft/COMP472_AK12/blob/278511d96c0fffa820d965cd1e7217938cdafdfc/data%20cleaning/split_emot.py)
  * **Description:** Program visualizes the images for each emotion from the FER-2013 dataset to know what emotion each class represents
* [transform.py](https://github.com/mkandaleft/COMP472_AK12/blob/278511d96c0fffa820d965cd1e7217938cdafdfc/data%20cleaning/transform.py)
  * **Description:** This script will organize and save the images based on their emotion labels. The program gets specified pixel lists from [angry_happy_neutral.csv](https://github.com/mkandaleft/COMP472_AK12/blob/278511d96c0fffa820d965cd1e7217938cdafdfc/data/extracted%20dataset/angry_happy_neutral.csv)
and converts them to images under a created new directory 'emotion_images' having 3 subdirectories '0' , '3' ,and '6' corresponding to 'angry' , 'happy' , 'neutral' .
* [imagededup_visualizer.py](https://github.com/mkandaleft/COMP472_AK12/blob/278511d96c0fffa820d965cd1e7217938cdafdfc/data%20cleaning/imagedup_visualizer.py)
  * **Description:** This script identifies and removes duplicate images in a specified directory using a CNN-based similarity detection method.
* [light_processing.py](https://github.com/mkandaleft/COMP472_AK12/blob/278511d96c0fffa820d965cd1e7217938cdafdfc/data%20cleaning/light_processing.py)
  * **Description:** Implements histogram specification by getting a reference pixel intensity from specified class and interpolates with images in target directory
* [resizer.py](https://github.com/mkandaleft/COMP472_AK12/blob/278511d96c0fffa820d965cd1e7217938cdafdfc/data%20cleaning/resizer.py)
  * **Description:** Modifies resolution of all images in specified folder to 48 x 48 pixels
  * **Demo only:** [resizer_demo.py](https://github.com/mkandaleft/COMP472_AK12/blob/278511d96c0fffa820d965cd1e7217938cdafdfc/data%20cleaning/resizer_demo.py) used to select a downscaling approach
* [to_png.py](https://github.com/mkandaleft/COMP472_AK12/blob/278511d96c0fffa820d965cd1e7217938cdafdfc/data%20cleaning/to_png.py)
  * **Description:** Changes extension of images in specified directory to png and removes the old format

### Data Visualization Programs

* [aggregate_pixel_intensity.py](https://github.com/mkandaleft/COMP472_AK12/blob/06ce0e2326ea255625329833b2af5f7e7d0b8e47/data%20visualization/aggregate_pixel_intensity.py)
  * **Description:** Calculates aggregate pixel intensities of outlined classes and displays a plot per class
  * **Demo only:** [cd_tester.py](https://github.com/mkandaleft/COMP472_AK12/blob/06ce0e2326ea255625329833b2af5f7e7d0b8e47/data%20visualization/cd_tester.py) displays cumulative pixel intensity of each class on the same plot. Implemented for an alternative way to compare aggregate histograms
* [class_distribution.py](https://github.com/mkandaleft/COMP472_AK12/blob/06ce0e2326ea255625329833b2af5f7e7d0b8e47/data%20visualization/class_distribution.py)
  * **Description:** This script counts the images in each class folder and plots the distribution in a histogram. 
* [display_samples.py](https://github.com/mkandaleft/COMP472_AK12/blob/06ce0e2326ea255625329833b2af5f7e7d0b8e47/data%20visualization/display_samples.py)
  * **Description:** This program will process each class folder, randomly select sample images, and plot the sample images and their pixel intensity histograms in a (5x3) grid.
* [knn2.py](https://github.com/mkandaleft/COMP472_AK12/blob/06ce0e2326ea255625329833b2af5f7e7d0b8e47/data%20visualization/knn2.py)
   * **Description:**  this program is clustering images into different directories based on their pixel values and visualizing the clusters using PCA and t-SNE.
* [resnet.py](https://github.com/mkandaleft/COMP472_AK12/blob/06ce0e2326ea255625329833b2af5f7e7d0b8e47/data%20visualization/resnet.py)
  * **Description:** This script trains a ResNet-18 model on the image dataset using PyTorch. It loads images from a class directory, applies transformations, and splits the data into training and testing sets. The model is fine-tuned to classify images into two categories. It uses cross-entropy loss and the Adam optimizer for training. After each epoch, the model's performance is evaluated using accuracy and F1 score metrics.

### Execution Instructions

The code can be executed in the PyCharm IDE:

```sh
PS C:\PATH_TO_PROJECT> python 'program_name.py'
```

