# EuroSAT: Land Use and Land Cover Classification with Sentinel-2

![EuroSAT overview image](https://github.com/phelber/EuroSAT/blob/master/eurosat_overview_small.jpg?raw=true)

## Short Description

In this study, we address the challenge of land use and land cover classification using Sentinel-2 satellite images. The Sentinel-2 satellite images are openly and freely accessible through the Copernicus Earth observation program. We present a novel dataset based on Sentinel-2 images, covering 13 spectral bands and consisting of 10 classes, with a total of 27,000 labeled and geo-referenced images. Benchmarks for this dataset were established using state-of-the-art deep Convolutional Neural Networks (CNNs), with ResNet-50 as the highest-performing model. Using this architecture, we achieved an overall classification accuracy of 95.88%. The resulting classification system enables numerous Earth observation applications, demonstrating its effectiveness in detecting land use and land cover changes and assisting in the improvement of geographical maps.

### Dataset
The dataset is available via [Zenodo](https://zenodo.org/record/7711810#.ZAm3k-zMKEA).

##  Project Structure

- [EuroSAT Benchmark for Land Use Classification](#eurosat-benchmark-for-land-use-classification)
  - [Land Type Classification using Sentinel-2 Satellite Images](#eurosat-paper-1pdf)
- [notebook](#notebook)
  - [DL Project.ipynb](#dl-projectipynb)
- [src](#src)
  - [Data preprocessing](#data-preprocessing)
    - [preprocessing.py](#preprocessingpy)
  - [Evaluation](#evaluation)
    - [test.py](#testpy)
  - [GUI](#gui)
    - [app.py](#apppy)
  - [Models](#models)
    - [CNN.py](#cnnpy)
    - [Googlenet.py](#googlenetpy)
    - [Resent50.py](#resent50py)
  - [Training](#training)
    - [train.py](#trainpy)
- [DL Project.ipynb](#dl-projectipynb-root)
- [app.py](#apppy-root)
- [requirements.txt](#requirementstxt)
- [README.md](#readmemd)

---

##  Folder & File Description
### EuroSAT Benchmark for Land Use Classification
<a name="eurosat-benchmark-for-land-use-classification"></a>
- [Land Type Classification using Sentinel-2 Satellite Images](#eurosat-paper-1pdf)
  
  <a name="eurosat-paper-1pdf"></a>
  Contains the research paper documenting the benchmark results of the EuroSAT dataset. 

### notebook
<a name="notebook"></a>
- [DL Project.ipynb](#dl-projectipynb)  
  <a name="dl-projectipynb"></a>
  Jupyter notebook for experiments, data visualization, and model evaluation.

### src
<a name="src"></a>

#### Data preprocessing
<a name="data-preprocessing"></a>
- [preprocessing.py](#preprocessingpy)  
  <a name="preprocessingpy"></a>
  Handles loading, cleaning, and preprocessing the EuroSAT images for model training.

#### Evaluation
<a name="evaluation"></a>
- [test.py](#testpy)  
  <a name="testpy"></a>
  Evaluates trained models on unseen data and produces performance metrics.

#### GUI
<a name="gui"></a>
- [app.py](#apppy)  
  <a name="apppy"></a>
  Graphical interface to run the classification model and visualize predictions interactively.

#### Models
<a name="models"></a>
- [CNN.py](#cnnpy)  
  <a name="cnnpy"></a>
  Custom Convolutional Neural Network implementation.
- [Googlenet.py](#googlenetpy)  
  <a name="googlenetpy"></a>
  GoogleNet model implementation for classification.
- [Resent50.py](#resent50py)  
  <a name="resent50py"></a>
  ResNet50 model implementation for classification.

#### Training
<a name="training"></a>
- [train.py](#trainpy)  
  <a name="trainpy"></a>
  Training pipeline for models, including data loading, model initialization, and training loops.

### Root Directory
- [DL Project.ipynb](#dl-projectipynb-root)  
  <a name="dl-projectipynb-root"></a>
  Main notebook combining preprocessing, training, and evaluation.
- [app.py](#apppy-root)  
  <a name="apppy-root"></a>
  Entry point for launching the GUI application.
- [requirements.txt](#requirementstxt)  
  <a name="requirementstxt"></a>
  List of required Python packages for running the project.
- [README.md](#readmemd)  
  <a name="readmemd"></a>
  Documentation for the project.


 ## 👥 Team Contributions

### Bassem Akram
**Worked on:**  
- Data cleaning and Analysis 
- Feature engineering

### Hend Ramadan
**Worked on:**
- Model design

### Menna Nour
**Worked on:**   
- Model training  
    
### Mina Ibrahim
**Worked on:** 
- Model testing and inference

### Ahmed Khaled
**Worked on:**  
- GUI implementation and application running

---

## 🧰 Technologies Used

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)  
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)  
[![Torchvision](https://img.shields.io/badge/Torchvision-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/vision/stable/index.html)  
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)  
[![Rasterio](https://img.shields.io/badge/Rasterio-4B9CD3?style=for-the-badge&logo=&logoColor=white)](https://rasterio.readthedocs.io/en/latest/)  
[![TQDM](https://img.shields.io/badge/TQDM-4B9CD3?style=for-the-badge&logo=&logoColor=white)](https://tqdm.github.io/)  
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)](https://matplotlib.org)  
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/)  
[![Seaborn](https://img.shields.io/badge/Seaborn-4B7BEC?style=for-the-badge&logo=&logoColor=white)](https://seaborn.pydata.org)
 

---



## Contact 

For any questions, reach out to the team members:

- Ahmed Khaled  
[Email](mailto:ahmed.ghaith979@gmail.com) | [LinkedIn](https://www.linkedin.com/in/ahmedkhaled369/) | [Kaggle](https://www.kaggle.com/ahmedkhaled369)

- Hend Ramadan  
[Email](mailto:hendtalba@gmail.com) | [LinkedIn](https://www.linkedin.com/in/hend-ramadan-72a9712a5) | [Kaggle](https://www.kaggle.com/hannod)

- Mina Ibrahim  
[Email](mailto:minaibrahim365@gmail.com) | [LinkedIn](https://www.linkedin.com/in/mina-ibrahim-ab7472313/) | [Kaggle](https://www.kaggle.com/minaibrahim22)

- Menna Nour  
[Email](mailto:menna.nour@example.com) | [LinkedIn](https://www.linkedin.com/in/menna-eldemerdash/)  | [Kaggle](https://www.kaggle.com/minaibrahim22)

- Bassem Akram  
[Email](mailto:basemakram560@gmail.com) | [LinkedIn](https://www.linkedin.com/in/basem-younis-7837723a1/)  |  [Kaggle](https://www.kaggle.com/basemakram)

<p align="right">(<a href="#readme-top">back to top</a>)</p>  
  


