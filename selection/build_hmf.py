from read_grid import read_grid
from read_halos import read_halos_region


overd, coods = read_grid()

masses, coods = read_halos_region()

idx1 = np.random.choice(overd.shape[0])
idx2 = np.random.choice(overd.shape[1])
idx2 = np.random.choice(overd.shape[2])

region = overd[idx1, idx2, idx3]




