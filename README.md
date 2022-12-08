# rss_regressor
### Overview ###
This project applies deep learning to wireless indoor localization.
The data used to train the model is the same as that used in trilateration and triangulation, i.e., the received signal strength (RSS) and angle of arrival (AOA).
A feed-forward neural network is used in this implementation.

### Libraries ###
The required libraries are
- tensorflow
- keras
- scikit-learn
- numpy
- matplotlib
- joblib


### Running the code ###
To run the project:
1. run the 'main_ensemble_train.py' file to generate ensemble models.
(Please change the path to the data files according to your settings.)
- To generate training dataset, set the flag 'gendataTrain = True'.
- To generate testing datset, set the flag 'gendataTest = True'.
- To train/re-train the model, set the flag 'trainmodel = True'.
2. run the 'main_ensemble_test.py' file to combine the ensemble models to produce the target's location.

*** First time running the code, please set all flags to True to generate the training and testing data, and train the model.

### Reference ###
If you find this code useful, please cite this work as

Wisanmongkol, J., et al.: An ensemble approach to deep-learning-based wireless indoor localization. IET Wirel. Sens. Syst. 12( 2), 33– 55. (2022). https://doi.org/10.1049/wss2.12035

https://ietresearch.onlinelibrary.wiley.com/share/X4DVDZH5VNXJIYX44V9G?target=10.1049/wss2.12035
