# Directory Content:

## Components: 
  - mainModel.py, Variant 3 and oldModel.py which are the model architectures.
  - dataLoader.py loads data into tensors and splits for training, validating and testing
  - training.py contains the training loop
  - evaluate.py will report metrics and confusion Matrices for a given set

## Main:
  - main.py will load mainModel.py ---> load the pre-processed data with dataLoader.py<br>
      ---> train using training.py ---> evaluate model and training using saved weights

## Other:
  - fane_experiment contains faneModel.py which is our experiment model.
  - model_weights.pth saves the weight after training.
  - gridSearch.py does a hyperparameter search to find best hyperparameters.

## Results:
  - contains screenshots of metrics and confusion matrices
