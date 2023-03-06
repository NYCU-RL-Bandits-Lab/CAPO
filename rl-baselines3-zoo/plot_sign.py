
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy.signal import savgol_filter

FILE = "./sign.txt"
corrects = []
incorrects = []
with open(FILE, 'r') as f:
    for i, l in enumerate(f):
        correct, incorrect = l.strip().split(',')
        correct = int(correct.strip())
        incorrect = int(incorrect.strip())
        corrects.append(correct)
        incorrects.append(incorrect)
n = len(corrects)
t = np.arange(n)
print(n)
corrects = savgol_filter(corrects, 401, 3) # window size 51, polynomial order 3
incorrects = savgol_filter(incorrects, 401, 3) # window size 51, polynomial order 3

# plt.figure()
plt.plot(t, corrects, label='correct')
plt.plot(t, incorrects, label='incorrect')
plt.legend()


plt.savefig("./test.png")
