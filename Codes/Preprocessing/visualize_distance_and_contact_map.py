import numpy as np
import os
import matplotlib.pyplot as plt

residue_name = input()
dist_map = np.load("distance_maps/" + residue_name + ".npy")

plt.imshow(dist_map, cmap="viridis")
plt.colorbar(label="Ångström Distance")
plt.title("Distance Map: " + residue_name)
plt.xlabel("Residue index")
plt.ylabel("Residue index")
plt.show()

cutoff = 8.0 # Ångströms
contact_map = (dist_map < cutoff).astype(int)

plt.imshow(contact_map, cmap="gray_r")
plt.title(f"Contact Map: {residue_name} (cutoff={cutoff} Å)")
plt.xlabel("Residue index")
plt.ylabel("Residue index")
plt.show()