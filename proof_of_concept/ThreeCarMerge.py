import numpy as np
from UnstructuredLaneChange import UnstructuredLaneChange

class ThreeCarMerge(UnstructuredLaneChange):
    def __init__(self):
        super().__init__(car_count=2)
        # simplest
        #self.x0 = np.array([[3.0,0, 2.0, 0], [0.0, 0, 2.0, 0], [1.5, 1.5, 2.0, 0]])
        #self.target_y = [0,0, 0]

        # merging that require rear car to slow down
        #self.x0 = np.array([[3.0,0, 2.1, 0], [1.0, 0, 2.1, 0], [1.5, 1.5, 2.0, 0]])
        #self.target_y = [0,0, 0]

        # collision resolution, longitudinal
        self.x0 = np.array([[1.0, 0, 2.3, 0], [1.5, 0, 2.2, 0]])
        self.target_y = [0, 0]
        self.J_Qr = np.diag([0,10,0,1])

if __name__=="__main__":
    main = ThreeCarMerge()
    main.solve(save_gif=False,visualize=True,animate=True)
    main.final()

