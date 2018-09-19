# Written by David McDougall, 2018

import numpy as np
from sdr import SDR
import math
import random
import hexy

class GridCellEncoder:
    def __init__(self,
        n              = 100,
        sparsity       = .30,
        module_periods = [6 * (2**.5)**i for i in range(5)],
        ):
        assert(min(module_periods) >= 4)
        self.n              = n
        self.sparsity       = sparsity
        self.module_periods = module_periods
        self.grid_cells     = SDR((n,))
        self.offsets = np.random.uniform(0, max(self.module_periods)*9, size=(n, 2))
        module_partitions       = np.linspace(0, n, num=len(module_periods) + 1)
        module_partitions       = list(zip(module_partitions, module_partitions[1:]))
        self.module_partitions  = [(int(round(start)), int(round(stop)))
                                        for start, stop in module_partitions]
        self.scales   = []
        self.angles   = []
        self.rot_mats = []
        for period in module_periods:
            self.scales.append(period)
            angle = random.random() * 2 * math.pi
            self.angles.append(angle)
            c, s = math.cos(angle), math.sin(angle)
            R    = np.array(((c,-s), (s, c)))
            self.rot_mats.append(R)
        self.reset()

    def reset(self):
        pass

    def encode(self, location):
        # Find the distance from the location to each grid cells nearest
        # receptive field center.
        # Convert the units of location to hex grid with angle 0, scale 1, offset 0.
        displacement = location - self.offsets
        radius       = np.empty(self.n)
        for mod_idx in range(len(self.module_partitions)):
            start, stop = self.module_partitions[mod_idx]
            R           = self.rot_mats[mod_idx]
            displacement[start:stop] = R.dot(displacement[start:stop].T).T
            radius[start:stop] = self.scales[mod_idx] / 2
        # Convert into and out of hexagonal coordinates, which rounds to the
        # nearest hexagons center.
        nearest = hexy.cube_to_pixel(hexy.pixel_to_cube(displacement, radius), radius)
        # Find the distance between the location and the RF center.
        distances = np.hypot(*(nearest - displacement).T)
        # Activate the closest grid cells in each module.
        index = []
        for start, stop in self.module_partitions:
            z = int(round(self.sparsity * (stop - start)))
            index.extend( np.argpartition(distances[start : stop], z)[:z] + start )
        self.grid_cells.flat_index = np.array(index)
        return self.grid_cells

if __name__ == '__main__':
    gc = GridCellEncoder()
    print('Module Periods', gc.module_periods)

    sz = 40
    rf = np.empty((gc.n, sz, sz))
    for x in range(sz):
        for y in range(sz):
            r = gc.encode([x, y])
            rf[:, x, y] = r.dense.ravel()

    import matplotlib.pyplot as plt
    plt.figure('GRID CELL TEST')
    samples = random.sample(range(gc.n), 100)
    samples = sorted(samples)
    for row in range(10):
        for col in range(10):
            i = row*10+col
            plt.subplot(10, 10, i + 1)
            plt.imshow(rf[samples[i]], interpolation='nearest')
    plt.show()
