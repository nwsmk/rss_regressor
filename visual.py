import matplotlib.pyplot as plt
import joblib

fname = "traindata_e20-20-200.save"
epochlist, trainrmselist, testrmselist = joblib.load(fname)
plt.plot(epochlist, trainrmselist, 'r-x', label="train")
plt.plot(epochlist, testrmselist, 'k-x', label="test")
plt.xlabel("Epochs")
plt.ylabel("Root mean squared error")
plt.legend()
plt.grid(True)
plt.show()
