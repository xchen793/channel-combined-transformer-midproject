import numpy as np
z = np.random.randint(0, 2, size=(100, 100))
bit_size = 100 * 100
nz = z^(np.random.random(z.shape) < 0.15)  #binary  symmetric channel 0.15 is the exact probability you specify 
bit_numerrs = np.sum(z != nz)
bit_pcterrs = bit_numerrs/float(bit_size)