NordAxon Code Monkeys Astra Zeneca Hackathon
==============================

The overall structure of the repo is described below.

To run inference with our models, we have prepared a notebook under notebooks/End-to-endpipeline.ipynb where  we  detail  the  data  preparation  and  inference  given  the raw data.

For a training pipeline, we would suggest that you follow the following steps:
- 1.  Start by processing your raw input data by running the scriptprepare_training_data.py. At the bottom of the script, you can define the paths to your raw data and where you wantto store the training data. Please make sure that you follow how the already defined pathsare structured - there will be a middle step folder created.
- 2.  Follow this with training your first model!  You will start with training the segmentationmodel for the nuclei masks by running thetrain_segmentation.pyscript. Masks werecreated with the maskSLIC algorithm in the previous step, and now you can use these forthe training.  If you did not change the paths in the previous script (except for the oneto your raw data), you should not have to change anything in this script. Otherwise, youmay have to change the paths.
- 3.  Now you have the GAN models left!  Here you have a multitude of parameters that youcan tune directly from the command line.  You run the scripttrain_pix2pix.pywitharguments that fit your data. Please note that theA1model will need 8 channels as input,whereas the other two targets only need 7. This is because of the masks from step 2.

Overall project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   ├── raw            <- The original, immutable data dump.
    |   └── models         <- Saved models
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── reports            <- Report hand-in for the competition
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to modify data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts that define models

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
