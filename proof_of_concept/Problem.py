import numpy as np
import matplotlib.pyplot as plt
import vnoise
noise = vnoise.Noise()

class Problem:
    ''' base class for a problem '''
    def __init__(self):
        # dimension of problem
        self.n = 0
        # total function evaluations
        self.eval_count = 0

    def groundTruth(self):
        ''' get ground truth for optimization, (x, fun_x) '''
        return 0,0

    def reset(self):
        ''' reset eval_count etc '''
        self.eval_count = 0

    def evaluate(self,val):
        ''' evaluate function '''
        self.eval_count += 1
        return val

    def visualize(self,val_vec):
        ''' visualize the function '''
        return None

class ParabolaWithSineNoise(Problem):
    def __init__(self):
        # dimension of problem
        self.n = 1
        # parameters
        self.A = 2
        self.B = 10
        self.C = 0.4

        # total function evaluations
        self.eval_count = 0


    def groundTruth(self):
        ''' get ground truth for optimization, (x, fun_x) '''
        return 0,0

    def evaluate(self,val):
        ''' evaluate function '''
        self.eval_count += 1
        assert(np.array(val).shape==()) # single value evaluation
        return self._evaluate(val)

    def _evaluate(self,val_vec):
        return self.A*val_vec**2 + self.C*np.sin(self.B*val_vec)

    def visualize(self,val_vec):
        ''' visualize the function '''
        xx = np.linspace(-1,1,1000)
        plt.plot(xx,self._evaluate(xx))
        plt.show()
        return None

class PerlinNoise(Problem):
    def __init__(self):
        # dimension of problem
        self.n = 2
        # parameters
        self.scale = 1  # Adjust this for different terrain scales
        self.octaves = 6
        self.persistence = 0.5

        # total function evaluations
        self.eval_count = 0

    def evaluate(self,val):
        ''' evaluate function '''
        self.eval_count += 1
        assert(np.array(val).shape==(2)) # single value evaluation
        value = noise.noise2(val * self.scale, val * self.scale, octaves=self.octaves, persistence=self.persistence)
        return value

    def _evaluate(self,val_vec):
        #value = np.vectorize(noise.noise2)(val_vec[:,[0]] * self.scale, val_vec[:,[1]] * self.scale, octaves=self.octaves, persistence=self.persistence)
        value = noise.noise2(val_vec[:,0] * self.scale, val_vec[:,1] * self.scale,grid_mode=False)
        return value

    def visualize(self,val_vec):
        ''' visualize the function '''
        xx,yy = np.meshgrid(np.linspace(-1,1),np.linspace(-1,1))
        zz = self._evaluate(np.vstack([xx.flatten(), yy.flatten()]).T)
        zz = zz.reshape(xx.shape)
        fig, ax = plt.subplots()
        z_min = np.min(zz.flatten())
        z_max = np.max(zz.flatten())
        #plt.imshow(zz,cmap='RdBu')
        #ax.axis([xx.min(), xx.max(),yy.min(), yy.max()])
        c = ax.pcolormesh(xx,yy,zz, cmap='RdBu', vmin=z_min, vmax=z_max)
        fig.colorbar(c,ax=ax)
        plt.show()
        return None

if __name__=='__main__':
    #problem = ParabolaWithSineNoise()
    problem = PerlinNoise()
    problem.visualize([])


