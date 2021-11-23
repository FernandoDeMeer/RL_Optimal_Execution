import numpy as np
from gym import spaces

space = spaces.Box(low=np.array([0., 0.0]), high=np.array([np.inf, 100.0]), shape=(2,), dtype=np.float64)
sample = np.array([0., 0.5], dtype=np.float64)

print(space.contains(sample))

# assert space.contains(sample)
# assert sample in space  # Alternatively
