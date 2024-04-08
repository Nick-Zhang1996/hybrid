import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import collections  as mc
from util import *
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
        #assert(np.array(val).shape==(self.n,)) # single value evaluation
        return self._evaluate(val).item()

    def _evaluate(self,val_vec):
        ''' val_vec.shape = (N,n), return: (N,1) '''
        return self.A*val_vec**2 + self.C*np.sin(self.B*val_vec)

    def jacobian(self,val):
        ''' return jacobian evaluated at val as a row vector '''
        return 2*self.A*val + self.C*self.B*np.cos(self.B*val)

    def hessian(self,val):
        ''' return hessian evaluated at val '''
        return 2*self.A - self.C*self.B**2*np.sin(self.B*val)

    def visualize(self,val_vec=None,rollout_cost=None,dir_vec=None):
        ''' visualize the function '''
        xx = np.linspace(-1,1,1000)
        plt.plot(xx,self._evaluate(xx))
        if (val_vec is not None):
            if (rollout_cost is None):
                rollout_cost = self._evaluate(val_vec)
            plt.plot(val_vec,rollout_cost,'o')

        '''
        # DEBUG check jacobian and hessian
        x = 0.2
        xx = np.linspace(x-0.2,x+0.2)
        yy = self.evaluate(x) + self.jacobian(xx)*(xx-x) + 0.5*self.hessian(xx)*(xx-x)**2
        plt.plot(xx,yy)
        '''

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
        ''' evaluate function val.shape = (n), return: float'''
        self.eval_count += 1
        assert(np.array(val).shape==(self.n,)) # single value evaluation
        value = noise.noise2(val[0] * self.scale, val[1] * self.scale, octaves=self.octaves, persistence=self.persistence)
        return value.item()

    def _evaluate(self,val_vec):
        ''' val_vec.shape = (N,n), return: (N,1) '''
        #value = np.vectorize(noise.noise2)(val_vec[:,[0]] * self.scale, val_vec[:,[1]] * self.scale, octaves=self.octaves, persistence=self.persistence)
        value = noise.noise2(val_vec[:,0] * self.scale, val_vec[:,1] * self.scale,grid_mode=False)
        return value

    def jacobian(self,val):
        ''' return jacobian evaluated at val as a row vector '''
        return linearizeNumerical(lambda x:self.evaluate(x), val)

    def hessian(self,val):
        ''' return hessian evaluated at val '''
        return None

    def visualize(self,val_vec=None, fun_val=None,dir_vec=None):
        ''' visualize the function 
        val_vec/fun_val values to highlight
        dir_vec: directions to note for val_vec

        '''
        xx,yy = np.meshgrid(np.linspace(-1,1,100),np.linspace(-1,1,100))
        zz = self._evaluate(np.vstack([xx.flatten(), yy.flatten()]).T)
        zz = zz.reshape(xx.shape)
        fig, ax = plt.subplots()
        z_min = np.min(zz.flatten())
        z_max = np.max(zz.flatten())
        #plt.imshow(zz,cmap='RdBu')
        c = ax.pcolormesh(xx,yy,zz, cmap='RdBu', vmin=z_min, vmax=z_max)
        fig.colorbar(c,ax=ax)

        # plot val_vec
        if (val_vec is not None):
            plt.plot(val_vec[:,0], val_vec[:,1],'ok')
            if (dir_vec is not None):
                x0 = val_vec[:,0]
                y0 = val_vec[:,1]
                x1 = val_vec[:,0] + dir_vec[:,0]
                y1 = val_vec[:,1] + dir_vec[:,1]
                lines = np.hstack([np.column_stack([x0,y0])[:,np.newaxis,:], np.column_stack([x1,y1])[:,np.newaxis,:]])
                if (val_vec.shape[0]==1):
                    x_vec = np.vstack([x0,x1]).T
                    y_vec = np.vstack([y0,y1]).T
                    ax.plot(x_vec.flatten(),y_vec.flatten(),'-k')
                else:
                    #ax.plot(x_vec,y_vec,'-k')
                    lc = mc.LineCollection(lines,linewidths=2,color='black')

        ax.add_collection(lc)
        ax.axis([xx.min(), xx.max(),yy.min(), yy.max()])
        plt.show()

        '''
        # DEBUG visualization
        fig = plt.figure()
        ax = Axes3D(fig)
        #ax.plot_surface(xx,yy,zz)
        # plot val_vec
        if (val_vec is not None):
            z_vec = self._evaluate(val_vec)
            plt.plot(val_vec[:,0], val_vec[:,1],z_vec,'ok')
            if (dir_vec is not None):
                x0 = val_vec[:,0]
                y0 = val_vec[:,1]
                x1 = val_vec[:,0] + dir_vec[:,0]
                y1 = val_vec[:,1] + dir_vec[:,1]
                x_vec = np.vstack([x0,x1]).T
                y_vec = np.vstack([y0,y1]).T
                if (x_vec.shape[0]==1):
                    ax.plot(x_vec.flatten(),y_vec.flatten(),z_vec.flatten(),'-k')
                else:
                    ax.plot(x_vec,y_vec,'-k')
        # visualize jacobian
        val = val_vec[0]
        N = 10
        xx_vec, yy_vec = np.meshgrid(np.linspace(-0.1,0.1,N),np.linspace(-0.1,0.1,N))
        xx_vec += val[0]
        yy_vec += val[1]
        zz_vec = []
        jac_vec = []
        J = self.jacobian(val)
        for x,y in zip(xx_vec.flatten(), yy_vec.flatten()):
            zz_vec.append(self.evaluate(np.array([x,y])))
            jac_vec.append(self.evaluate(val)+ J@(np.array([x,y])-val).T)

        ax.plot_surface(np.array(xx_vec),np.array(yy_vec),np.array(zz_vec).reshape(N,N),color=(1.0,0,0))
        ax.plot_surface(np.array(xx_vec),np.array(yy_vec),np.array(jac_vec).reshape(N,N),color=(0,1.0,0))

        plt.show()
        '''
        return None

class ParabolaWithSineNoise2D(Problem):
    def __init__(self):
        # dimension of problem
        self.n = 2
        # parameters
        self.A = 2
        self.B = 10
        self.C = 0.4

        # total function evaluations
        self.eval_count = 0

    def evaluate(self,val):
        ''' evaluate function '''
        self.eval_count += 1
        assert(np.array(val).shape==(self.n,)) # single value evaluation
        return self._evaluate(val.reshape(-1,self.n)).item()

    def _evaluate(self,val_vec):
        ''' val_vec.shape = (N,n), return: (N,1) '''
        x = (val_vec[:,0]**2+val_vec[:,1]**2)**0.5
        return self.A*x**2 + self.C*np.sin(self.B*val_vec[:,0]) + self.C*np.cos(self.B*val_vec[:,1])

    def jacobian(self,val):
        ''' return jacobian evaluated at val as a row vector '''
        return linearizeNumerical(lambda x:self.evaluate(x), val)

    def hessian(self,val):
        ''' return hessian evaluated at val '''
        return None

    def visualize(self,val_vec=None, fun_val=None,dir_vec=None):
        ''' visualize the function 
        val_vec/fun_val values to highlight
        dir_vec: directions to note for val_vec

        '''
        xx,yy = np.meshgrid(np.linspace(-1,1,100),np.linspace(-1,1,100))
        zz = self._evaluate(np.vstack([xx.flatten(), yy.flatten()]).T)
        zz = zz.reshape(xx.shape)
        fig, ax = plt.subplots()
        z_min = np.min(zz.flatten())
        z_max = np.max(zz.flatten())
        #plt.imshow(zz,cmap='RdBu')
        c = ax.pcolormesh(xx,yy,zz, cmap='RdBu', vmin=z_min, vmax=z_max)
        fig.colorbar(c,ax=ax)

        # plot val_vec
        if (val_vec is not None):
            plt.plot(val_vec[:,0], val_vec[:,1],'ok')
            if (dir_vec is not None):
                x0 = val_vec[:,0]
                y0 = val_vec[:,1]
                x1 = val_vec[:,0] + dir_vec[:,0]
                y1 = val_vec[:,1] + dir_vec[:,1]
                x_vec = np.vstack([x0,x1]).T
                y_vec = np.vstack([y0,y1]).T
                if (x_vec.shape[0]==1):
                    ax.plot(x_vec.flatten(),y_vec.flatten(),'-k')
                else:
                    ax.plot(x_vec,y_vec,'-k')
        #ax.plot(np.array([[0.1,0.2]]),np.array([[0.1,0.2]]),'-k')
        #ax.plot(np.array([[0.90022836,0.92629124]]),np.array([[ 0.27586163,-0.72379867]]),'-k')
        ax.axis([xx.min(), xx.max(),yy.min(), yy.max()])
        plt.show()

        return None

if __name__=='__main__':
    problem = ParabolaWithSineNoise()
    #problem = PerlinNoise()
    problem.visualize()


