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
To run the project, run the 'main.py' file. (Please change the path to the data files according to your settings.)
>> To generate training dataset, set the flag 'gendataTrain = True'.
>> To generate testing datset, set the flag 'gendataTest = True'.
>> To train/re-train the model, set the flag 'trainmodel = True'.

### Reference ###
https://ietresearch.onlinelibrary.wiley.com/share/X4DVDZH5VNXJIYX44V9G?target=10.1049/wss2.12035
