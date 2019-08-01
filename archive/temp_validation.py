# %%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid")

# %%
l_ttl = 8.4
A = (7.5e-4)**2*np.pi #m^2
I = 5 #A
sigma = 3.46e6 #S/m
c = 296 #J/Kg/K
rho = 6440 #kg/m^3
Q = 1.5e-8 #m^3/s

# %%
dl = 1e-3
l = np.arange(0,l_ttl,dl)
temps = 20 + (I*l)/(c*rho*A*sigma*Q)

#%%
plt.plot(l,temps)
plt.xlabel("Length")
plt.ylabel("Temperature")

#%%

#%%
