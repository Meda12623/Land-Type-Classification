# EuroSAT: Land Use and Land Cover Classification with Sentinel-2

![EuroSAT overview image](https://github.com/phelber/EuroSAT/blob/master/eurosat_overview_small.jpg?raw=true)

## Short Description

In this study, we address the challenge of land use and land cover classification using Sentinel-2 satellite images. The Sentinel-2 satellite images are openly and freely accessible provided in the Earth observation program Copernicus. We present a novel dataset based on Sentinel-2 satellite images covering 13 spectral bands and consisting out of 10 classes with in total 27,000 labeled and geo-referenced images. We provide benchmarks for this novel dataset with its spectral bands using state-of-the-art deep Convolutional Neural Network (CNNs). With the proposed novel dataset, we achieved an overall classification accuracy of 95.88\%. The resulting classification system opens a gate towards a number of Earth observation applications. We demonstrate how this classification system can be used for detecting land use and land cover changes and how it can assist in improving geographical maps. The geo-referenced dataset EuroSAT is made publicly available [here](#).

### Dataset
The dataset is available via [Zenodo](https://zenodo.org/record/7711810#.ZAm3k-zMKEA).

##  Project Structure

- [EuroSAT Benchmark for Land Use Classification](#eurosat-benchmark-for-land-use-classification)
  - [eurosat_paper (1).pdf](#eurosat-paper-1pdf)
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
- [eurosat_paper (1).pdf](#eurosat-paper-1pdf)
  
  <a name="eurosat-paper-1pdf"></a>
  Contains the research paper documenting the benchmark results of the EuroSAT dataset. 
ظبطلي دي


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


