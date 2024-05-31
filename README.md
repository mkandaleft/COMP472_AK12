
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
  ```
### Available Programs

#### Data cleaning

* [converter.py](https://github.com/mkandaleft/COMP472_AK12/blob/278511d96c0fffa820d965cd1e7217938cdafdfc/data%20cleaning/converter.py)
  * **Description:** Program gets speciified pixel lists from a FER-2013 CSV and converts them to images, sorting directly into "angry", "neutral", and "happy" classes
* [imagededup_visualizer.py](https://github.com/mkandaleft/COMP472_AK12/blob/278511d96c0fffa820d965cd1e7217938cdafdfc/data%20cleaning/imagedup_visualizer.py)
* [light_processing.py](https://github.com/mkandaleft/COMP472_AK12/blob/278511d96c0fffa820d965cd1e7217938cdafdfc/data%20cleaning/light_processing.py)
  * **Description:** Implements histogram specification by getting a reference pixel intensity from specified class and interpolates with images in target directory
* [resizer.py](https://github.com/mkandaleft/COMP472_AK12/blob/278511d96c0fffa820d965cd1e7217938cdafdfc/data%20cleaning/resizer.py)
  * **Description:** Modifies resolution of all images in specified folder to 48 x 48 pixels
  * [resizer_demo.py](https://github.com/mkandaleft/COMP472_AK12/blob/278511d96c0fffa820d965cd1e7217938cdafdfc/data%20cleaning/resizer_demo.py) used to select a downscaling approach
* [split_emot.py](https://github.com/mkandaleft/COMP472_AK12/blob/278511d96c0fffa820d965cd1e7217938cdafdfc/data%20cleaning/split_emot.py)
* [to_png.py](https://github.com/mkandaleft/COMP472_AK12/blob/278511d96c0fffa820d965cd1e7217938cdafdfc/data%20cleaning/to_png.py)
  * **Description:** Changes extension of images in specified directory to png and removes the old format
* [transform.py](https://github.com/mkandaleft/COMP472_AK12/blob/278511d96c0fffa820d965cd1e7217938cdafdfc/data%20cleaning/transform.py)

