# CNNs via MNIST
Leverage PyTorch and the MNIST (handwritten digits) to practice building Classification CNN models.

Current State: on hold (working on other data sets)

Completed Tasks:
* generalized data loader & model trainer created in scripts/
* basic CNN models built for both cpu & gpu training in models/
* tensorboard logs populating in runs/

TODO (no priority):
* define more models (mlp; resnet8)
* enhance training loop with cross-validation & hyperparameter tuning
    * scikit-learn
    * or
    * skorch
* analyze models in a notebook
* containerize best model, wrap with GUI, deploy on cloud for interactive usage