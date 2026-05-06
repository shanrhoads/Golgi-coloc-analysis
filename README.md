# Golgi-coloc-analysis
Quantitative assessment of fluorescent protein localization to the Golgi in confocal microscopy images.

## **Computer setup (only do these things once per computer)**
To begin using the analysis included in the package, you will need to install the following Python packages. It's recommended to do so using a Python environment manager, like Anaconda. See how below:

**Anaconda and Git install:**
1. Download Anaconda from this site: https://www.anaconda.com/download. Run the installer.
    
    **IMPROTANT for Windows computers:** during the install check the box that adds the installation to the Windows PATH variable. This is not "recommended", but is essential for success!! Do not skip this!!!
2. **If you have never used Git before on your computer:** download Git from this site: git-scm.com. Run the installer.
    
    **IMPROTANT for Windows computers:** Select "Use Git and optional UNIX tolls from the Command Prompt" to ensure that Git gets added to path.

**Setting up a conda environment:**
1. Open Command Prompt, or your terminal of choice.
2. Run the following commands line-by-line in order:
    ```Python
    conda create -n golgi-coloc python=3.10
    conda activate golgi-coloc

    pip install infer-subc
    pip install bioio bioio-czi
    pip install napari[all]

    python -m ipykernel install --user --name=golgi-coloc --display-name "Python golgi-coloc"
    ```

**Download the Golgi-coloc-analysis repository:**
1. Open your terminal (e.g. Command Prompt on Windows).
2. Navigate to the folder location you want to download the GitHub repository to.
    - Use the "cd {folder-name}" command to move into the specified folder location
    - Use the "cd .." command to move backwards one folder location
    - Use the "dir" command to print the folders and files in the current repository
3. Clone the golgi-coloc-analysis repository to your local computer by exicuting the following command:
    ```Python
    git clone https://github.com/shanrhoads/Golgi-coloc-analysis.git
    ```

## **Running the Golgi colocalization analysis**

### **Analysis workflow:**

This repository has several components that are meant to function as a pipeline. Their purpose and the intended order of opporations are as follows:

1. Segmentation of the Golgi
    - [1.4_infer_golgi](./analysis-pipelines/1.4_infer_golgi.ipynb): This workflow is used to determine the best settings for segmentation.
    - [batch_process_segmentations](./analysis-pipelines/batch_process_segmentations.ipynb): This notebook allows you to apply the segmentation settings determined in notebook 1.4_infer-golgi to a batch of images.
    - [quality_check_segmentations](./analysis-pipelines/quality_check_segmentations.ipynb): This notebook walks you through a quality checking process where each image and the associated Golgi segmentations are visualized. Edits can be made image-by-image in this notebook, if necessary. If an entire batch needs to be reprocessed, the batch_process_segmentation notebook can be rerun instead.
2. Quantification of intensity inside the Golgi regions
    - <mark> UPDATE THIS ONCE THE FINAL NOTEBOOK IS AVAILABLE

***NOTE:** this repository also includes a cell/nucleus segmentation notebook that is *optional*. If a per cell analysis is desired, the [1.1_infer_masks_from-composite_with_nuc](./analysis-pipelines/1.1_infer_masks_from-composite_with_nuc.ipynb) can be used in the same fashion as the 1.4_infer-golgi notebook and then the settings can be batch processed in the batch_process_segmentation notebook as described above.*

### **How to open and run the analysis through Jupyter Notebooks:**

You can use your favorite code editor, such as [Visual Studio Code]() or access everything through Jupyter Labs. The instructions for Jupyter Lab access is listed here: 
1. Open your terminal (e.g., Command Prompt on Windows).
2. Activate your conda environment and initiate Jupyter labs:
    ```Python
    conda activate golgi-coloc
    juptyer lab
    ```
3. A new window in your webbrowser should appear with a Jupyter Lab space. Use the built in file navigation system to find the Golgi-coloc-analysis repository that you cloned.
4. Carry out your analysis using the Jupyter notebooks found in 'Golgi-coloc-analysis' > 'analysis-pipelines'


## **Development notes**

This package was based on the existing framework of the [*infer-subc*](https://github.com/SCohenLab/infer-subc) organelle analysis package. The segmentation workflows and morphology quantification pipeline from *infer-subc* v2.0.0b1 were copied and modified for use here following their [license](https://github.com/SCohenLab/infer-subc/blob/main/LICENSE).
