import numpy as np
from UnstructuredLaneChange import UnstructuredLaneChange

class ThreeCarMerge(UnstructuredLaneChange):
    def __init__(self):
        super().__init__(car_count=3)
        self.x0 = np.array([[3.0,0, 2.0, 0], [0.0, 0, 2.0, 0], [1.5, 1.5, 2.0, 0]])
        self.target_y = [0,0, 0]

if __name__=="__main__":
    main = ThreeCarMerge()
    main.solve(save_gif=False,visualize=False,animate=True)
    main.final()

