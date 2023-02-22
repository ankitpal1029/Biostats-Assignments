import numpy as np
import math

data = np.array([0.496171, 5.26533, 0.931403, 3.31149, 3.41641, 1.55618, 2.04802, 1.42595, 0.369204, 2.38854, 5.11589, 0.0638427, 0.303104, 3.3126, 0.751968, 2.16072, 0.715723, 0.537688, 1.31197, 6.03044, 0.0852217, 0.0641738, 1.33244, 1.28574, 0.130804, 1.08743, 3.63306, 0.561859, 0.315556, 0.164096, 0.0102462, 0.538957, 6.78743, 0.475209, 1.47687, 0.530609, 0.500678, 0.25049, 0.0358761, 2.14403, 1.3575, 0.0634059, 0.686267, 0.740474, 2.59658, 3.087, 2.66754, 1.35586, 0.694144, 0.756533, 0.26969, 0.119303, 0.0228468, 0.050462, 0.504946, 6.85364, 2.93545, 5.03093, 4.33381, 1.20729, 0.17341, 4.38488, 0.034147, 3.166, 0.742843, 0.00990518, 4.53424, 7.0845, 2.26287, 0.805308, 0.0234403, 1.99731, 1.7815, 0.510237, 1.08118, 1.81082, 0.89951, 0.813067, 5.05697, 0.795866, 0.0366557, 0.0185079, 0.89917, 5.7816, 0.767073, 2.26308, 3.88149, 4.61512, 5.46474, 0.0795005, 1.92033, 0.628585, 0.64974, 0.240192, 5.19963, 2.76195, 0.686384, 2.66974, 2.1002, 0.817489, 0.000662403, 6.64782, 1.13877, 0.357796, 1.93296, 0.07817, 2.60229, 1.77667, 0.249727, 1.07697, 1.60397, 0.562914, 1.34751, 0.884525, 1.8234, 3.77961, 0.171684, 2.30696, 1.3162, 0.3764967, 0.0710807, 1.89867, 1.9169, 1.03711, 0.670067, 0.0983501, 1.66534, 3.92959, 0.350353, 8.72518, 0.775613, 2.77873, 1.32846, 6.6946, 0.868716, 0.206313, 0.00830072, 2.43967, 5.12259, 6.56197, 0.547719, 3.63278, 1.93439, 0.940522, 9.2161, 0.196014, 2.21003, 1.19311, 0.578256, 1.18532])

print(len(data))


def firstMoment():
    sum = np.sum(data)
    
    return sum/data.size

def secondMoment():
    mean = firstMoment()
    np_mean = np.full((len(data), 1), mean)
    """
    var = 
    """
    var  = np.sum((data - mean) ** 2) / len(data) - 1
    return var

def thirdMoment():
    mean = firstMoment()
    var = secondMoment()
    var_sqrt = math.sqrt(var)

    third = np.sum((data - mean) ** 3) / (len(data) * var_sqrt ** 3)
    return third

def fourthMoment():
    mean = firstMoment()
    var = secondMoment()
    var_sqrt = math.sqrt(var)

    fourth = np.sum((data - mean) ** 4)/ (len(data) * var_sqrt ** 4)
    return fourth

# Pearsons B1 and B2
def beta1Pearson():
    return thirdMoment()/ math.sqrt(secondMoment()) ** 3

def beta2Pearson():
    return fourthMoment()/ math.sqrt(secondMoment()) ** 4

first = firstMoment()
print(f"first moment of data is {first}")

second = secondMoment()
print(f"second moment of data is: {second}")

third = thirdMoment()
print(f"third moment of data is: {third}")

fourth = fourthMoment()
print(f"fourth moment of data is: {fourth}")

beta1 = beta1Pearson()
print(f"beta1 of data is: {beta1}")

beta2 = beta2Pearson()
print(f"beta2 of data is: {beta2}")

# Pearson classfication
def pearsonClassification():
    sample_skew = np.sum((data - firstMoment())**3) / ((len(data) - 1) * firstMoment()**(3/2))
    sample_kurt = np.sum((data - firstMoment())**4) / ((len(data) - 1) * firstMoment()**2) - 3

    return sample_skew**2 / sample_kurt


pearsonClass = pearsonClassification()
print(f"pearson classification of data is: {pearsonClass}")
