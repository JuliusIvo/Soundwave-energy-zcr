from tkinter import Tk
from tkinter.filedialog import askopenfilename
from scipy.io import wavfile
import numpy
import matplotlib.pyplot as plt

def plotSignal(data, time, filename):
    if(numpy.ndim(data.shape)==1):
        plt.plot(time, data[:]) 
    else:
        for dimension in data.shape:
            plt.plot(time, dimension[:])
    plt.legend()
    plt.title(f"soundfile - {filename.split('/')[-1]}")
    plt.show()

def calculateEnergy(data, window_size):
    energy = [numpy.sum(numpy.square(segment, dtype=numpy.int64)) for segment in data]
    return energy

def plotEnergy(data, time, filename, window_size):
    energy = calculateEnergy(data, window_size)
    plt.plot(time, energy)
    plt.xlabel("Time (s)")
    plt.ylabel("Energy")
    plt.title(f"Energy of {filename.split('/')[-1]}")
    plt.show()

def calculateNKS(data, window_size):
    NKS = []
    for i in range(0, len(data), window_size):
        segment = data[i:i+window_size]
        sign_changes = numpy.where(numpy.diff(numpy.sign(segment)))[0]
        nks = len(sign_changes) / (2 * len(segment))
        NKS.append(nks)
    return NKS

def plotNKS(data, time, filename, window_size):
    if numpy.ndim(data.shape) > 1:
        data = data.mean(axis=1)
    NKS = calculateNKS(data, window_size)
    time_zcr = numpy.arange(0, len(data), window_size) / samplerate
    plt.plot(time_zcr, NKS)
    plt.xlabel("Time (s)")
    plt.ylabel("NKS")
    plt.title(f"NKS diagram of {filename.split('/')[-1]}")
    plt.show()

def plotEnergyWithThreshold(data, time, filename, window_size, threshold):
    energy = calculateEnergy(data, window_size)
    sound_events = [t for t, e in zip(time, energy) if e > threshold]
    
    plt.plot(time, energy)
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    plt.scatter(sound_events, [threshold] * len(sound_events), color='g', label='Sound Events')
    
    plt.xlabel("Time (s)")
    plt.ylabel("Energy")
    plt.title(f"Energy with Threshold of {filename.split('/')[-1]}")
    plt.legend()
    plt.show()

window_size = int(input("Enter frame length for calculations: "))

threshold = float(input("Enter the energy threshold: "))

filename = askopenfilename()
samplerate, data = wavfile.read(filename)

time = numpy.linspace(0., data.shape[0] / samplerate, data.shape[0])

plotSignal(data, time, filename)

plotEnergy(data, time, filename, window_size)

plotNKS(data, time, filename, window_size)

plotEnergyWithThreshold(data, time, filename, window_size, threshold)
