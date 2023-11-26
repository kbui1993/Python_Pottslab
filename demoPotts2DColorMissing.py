from utils import *
import numpy as np
import matplotlib.pyplot as plt
from Potts2D import minL2Potts2DADMM

# load image
img_pil = get_image('Data/church.jpg',-1)[0]
f = np.array(img_pil)/255

# add noise
sigma = 0.2
imgNoisy = f + np.random.normal(0,sigma, f.shape)

# set weights and destroy image
m = f.shape[0]
n = f.shape[1]
missingFraction = 0.6
weights = (np.random.rand(m,n) >0.6)*1

for i in range(0,3):
    imgNoisy[:,:,i] = imgNoisy[:,:,i] * weights


# Potts restoration
gamma = 0.4
u = minL2Potts2DADMM(imgNoisy, gamma, weights = weights)

# show results
plt.figure(figsize=(20,20))
plt.subplot(1,3,1)
plt.imshow(f)
plt.axis('off')
plt.title('Original')

plt.subplot(1,3,2)
plt.imshow(imgNoisy)
plt.axis('off')
plt.title('Noisy and Corrupted')

plt.subplot(1,3,3)
plt.imshow(u)
plt.axis('off')
plt.title('Restored')