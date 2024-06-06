import torch
import time
start_time = time.time()

device = torch.device('cpu')
x = torch.rand((10000, 10000), dtype=torch.float32)
y = torch.rand((10000, 10000), dtype=torch.float32)
x = x.to(device)
y = y.to(device)
x*y
print("Time for CPU --- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
device = torch.device('mps')
x = torch.rand((10000, 10000), dtype=torch.float32)
y = torch.rand((10000, 10000), dtype=torch.float32)
x = x.to(device)
y = y.to(device)
x*y
print("Time for GPU --- %s seconds ---" % (time.time() - start_time))