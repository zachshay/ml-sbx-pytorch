# ml-sbx-pytorch
This is a personal sandbox used for various experiements with PyTorch:
* CNN via MNIST: classify images
* RNN via IMDB: sentiment analysis
* RNN via <tbd>: forecast a weather attribute
* Transformer via SST-2
* Shootouts via Yelp: RNN vs. Transformer on sentiment analysis

The root directory of this repo should include:
* This README
* Folders for each broad experiement


Sub-Folders should include:
* A README describing:
    * Purpose
    * Current State
    * Completed Tasks
    * Remaining Tasks
* data: empty folder where data is downloaded
* models: python code defining models
* notebooks: more expressive medium for completed work *(tbd)*
* runs: empty folder to hold TensorBoard logs
* scripts: utility functions supporting model code