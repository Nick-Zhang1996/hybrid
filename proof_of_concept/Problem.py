import numpy as np
from math import sin,cos
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import collections  as mc
from utilities.util import *
import vnoise
import os
from PIL import Image
from scipy.optimize import fsolve

noise = vnoise.Noise()

class Problem:
    ''' base class for a problem '''
    def __init__(self):
        # dimension of problem
        self.n = 0
        # total function evaluations
        self.evaluations = 0
        self.frame_vec = []

    def groundTruth(self):
        ''' get ground truth for optimization, (x, fun_x) '''
        return 0,0

    def reset(self):
        ''' reset evaluations etc '''
        self.evaluations = 0

    def evaluate(self,val):
        ''' evaluate objective function, 
        val.shape = (N,1) 
        type(return): float
        '''
        self.evaluations += 1
        return val

    def _evaluate(self,val_vec):
        '''
        batch evaluation, not counted towards self.evaluations
        val_vec.shape = (N,n), 
        return.shape = (N,1) 
        '''
        return np.zeros(val_vec.shape[0])

    def getLx(self):
        return []
    def getHx(self):
        return []

    def setConstraints(self, solver):
        for lx in self.getLx():
            solver.addLx(lx)
        for hx in self.getHx():
            solver.addHx(hx)
        return

    def visualize(self,val_vec=None, fun_val_vec=None,dir_vec=None,visualize=False,save_gif=False):
        ''' visualize the function 
        val_vec: vector of sampled points
        fun_val_vec: vector of cost for sampled points, usually not used
        dir_vec: step direction vector for sampled points '''
        if (not visualize and not save_gif):
            return
        else:
            fig = self._visualize(val_vec, fun_val_vec, dir_vec)
            if (save_gif):
                fig.canvas.draw()
                frame = Image.frombytes('RGB',
                fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
                self.frame_vec.append(frame)
            if (visualize):
                plt.show()

        return

    def resolveLogname(self,):
        # setup log file
        # log file will record state of the vehicle for later analysis
        logFolder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','gifs/'))
        logPrefix = "iteration"
        logSuffix = ".gif"
        no = 1
        while os.path.isfile(logFolder+logPrefix+str(no)+logSuffix):
            no += 1

        log_no = no
        logFilename = logFolder+logPrefix+str(no)+logSuffix
        return logFilename

    def final(self):
        if (len(self.frame_vec)>0):
            gif_filename = self.resolveLogname()
            self.frame_vec[0].save(fp=gif_filename,format='GIF',append_images=self.frame_vec,save_all=True,duration = 200,loop=0)
            print(f'GIf saved to {gif_filename}')

class ParabolaWithSineNoise(Problem):
    def __init__(self):
        Problem.__init__(self)
        # dimension of problem
        self.n = 1
        # parameters
        self.A = 2
        self.B = 10
        self.C = 0.4

        # total function evaluations
        self.evaluations = 0

    def groundTruth(self):
        ''' get ground truth for optimization, (x, fun_x) '''
        return 0,0

    def evaluate(self,val):
        ''' evaluate function '''
        self.evaluations += 1
        #assert(np.array(val).shape==(self.n,)) # single value evaluation
        return self._evaluate(val).item()
    def evaluateNoCount(self,val):
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

    def _visualize(self,val_vec=None,fun_val_vec=None,dir_vec=None):
        ''' visualize the function '''
        xx = np.linspace(-1,1,1000)
        fig, ax = plt.subplots()
        ax.plot(xx,self._evaluate(xx))
        if (val_vec is not None):
            if (fun_val_vec is None):
                fun_val_vec = self._evaluate(val_vec)
            ax.plot(val_vec,fun_val_vec,'o')

        '''
        # DEBUG check jacobian and hessian
        x = 0.2
        xx = np.linspace(x-0.2,x+0.2)
        yy = self.evaluate(x) + self.jacobian(xx)*(xx-x) + 0.5*self.hessian(xx)*(xx-x)**2
        ax.plot(xx,yy)
        '''
        return fig

class PerlinNoise(Problem):
    def __init__(self):
        Problem.__init__(self)
        # dimension of problem
        self.n = 2
        # parameters
        self.scale = 1  # Adjust this for different terrain scales
        self.octaves = 6
        self.persistence = 0.5

        # total function evaluations
        self.evaluations = 0

    def evaluate(self,val):
        ''' evaluate function val.shape = (n), return: float'''
        self.evaluations += 1
        assert(np.array(val).shape==(self.n,)) # single value evaluation
        value = noise.noise2(val[0] * self.scale, val[1] * self.scale, octaves=self.octaves, persistence=self.persistence)
        return value.item()
    def evaluateNoCount(self,val):
        value = noise.noise2(val[0] * self.scale, val[1] * self.scale, octaves=self.octaves, persistence=self.persistence)
        return value.item()

    def _evaluate(self,val_vec):
        ''' val_vec.shape = (N,n), return: (N,1) '''
        #value = np.vectorize(noise.noise2)(val_vec[:,[0]] * self.scale, val_vec[:,[1]] * self.scale, octaves=self.octaves, persistence=self.persistence)
        value = noise.noise2(val_vec[:,0] * self.scale, val_vec[:,1] * self.scale,grid_mode=False)
        return value

    def jacobian(self,val):
        ''' return jacobian evaluated at val as a row vector '''
        return jacobianNumerical(lambda x:self.evaluate(x), val)

    def hessian(self,val):
        ''' return hessian evaluated at val '''
        return None

    def _visualize(self,val_vec=None, fun_val_vec=None,dir_vec=None):
        ''' visualize the function 
        val_vec/fun_val_vec values to highlight
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

        # DEBUG check min in gridsearch
        '''
        min_idx = np.argmin(zz.flatten())
        min_x = xx.flatten()[min_idx]
        min_y = yy.flatten()[min_idx]
        min_z = zz.flatten()[min_idx]
        print(f'x={min_x},y={min_y},val={min_z}')
        '''

        # plot val_vec
        if (val_vec is not None):
            ax.plot(val_vec[:,0], val_vec[:,1],'ok')
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
        N = 50
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
        return fig

class ParabolaWithSineNoise2D(Problem):
    def __init__(self):
        Problem.__init__(self)
        # dimension of problem
        self.n = 2
        # parameters
        self.A = 2
        self.B = 10
        self.C = 0.4

        # total function evaluations
        self.evaluations = 0
    def getHx(self):
        return [lambda x:((x[0]-0.15)**2+(x[1]-0.05)**2-0.3**2)*100]

    def evaluate(self,val):
        ''' evaluate function '''
        self.evaluations += 1
        assert(np.array(val).shape==(self.n,)) # single value evaluation
        return self._evaluate(val.reshape(-1,self.n)).item()
    def evaluateNoCount(self,val):
        return self._evaluate(val.reshape(-1,self.n)).item()

    def _evaluate(self,val_vec):
        ''' val_vec.shape = (N,n), return: (N,1) '''
        x = (val_vec[:,0]**2+val_vec[:,1]**2)**0.5
        return self.A*x**2 + self.C*np.sin(self.B*val_vec[:,0]) + self.C*np.cos(self.B*val_vec[:,1])

    def jacobian(self,val):
        ''' return jacobian evaluated at val as a row vector '''
        return jacobianNumerical(lambda x:self.evaluate(x), val)

    def hessian(self,val):
        ''' return hessian evaluated at val '''
        return None

    def _visualize(self,val_vec=None, fun_val_vec=None,dir_vec=None):
        ''' visualize the function 
        val_vec/fun_val_vec values to highlight
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
            ax.plot(val_vec[:,0], val_vec[:,1],'ok')
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

        # DEBUG plot constraint
        xx = 0.3*np.cos(np.linspace(0,2*np.pi)) + 0.15
        yy = 0.3*np.sin(np.linspace(0,2*np.pi)) + 0.05
        ax.plot(xx,yy)
        return fig

class ParabolaWithSineNoise3D(Problem):
    def __init__(self):
        Problem.__init__(self)
        # dimension of problem
        self.n = 3
        # parameters
        self.A = 2
        self.B = 10
        self.C = 0.4

        # total function evaluations
        self.evaluations = 0

        # create circle (x-0.2)**2 + (y-0.1)**2 + z**2 = 1**2, in plane sin(x) + y + z = 0
        theta_vec = np.linspace(0,2*np.pi)
        x0 = 0.2; y0 = 0.1
        r_vec = []
        residual_vec = []
        for theta in theta_vec:
            fun = lambda r:(r*cos(theta))**2 + (r*sin(theta))**2 + ((sin(x0+r*cos(theta)) + y0+r*sin(theta)))**2 - 1
            res = fsolve(fun,1.0)
            r_vec.append(res.item())
            residual_vec.append(fun(res.item()))
        r_vec = np.array(r_vec)
        self.r_vec = r_vec
        
    def getHx(self):
        return [lambda u: (u[0]-0.2)**2 + (u[1]-0.1)**2 + u[2]**2 - 1.0**2]
    def getLx(self):
        return [lambda u: sin(u[0]) + u[1] + u[2]]


    def evaluate(self,val):
        ''' evaluate function '''
        self.evaluations += 1
        assert(np.array(val).shape==(self.n,)) # single value evaluation
        return self._evaluate(val.reshape(-1,self.n)).item()
    def evaluateNoCount(self,val):
        return self._evaluate(val.reshape(-1,self.n)).item()

    def _evaluate(self,val_vec):
        ''' val_vec.shape = (N,n), return: (N,1) '''
        x = (val_vec[:,0]**2+val_vec[:,1]**2)**0.5
        return self.A*x**2 + self.C*np.sin(self.B*val_vec[:,0]) + self.C*np.cos(self.B*val_vec[:,1]) + (val_vec[:,2]-0.2)**3

    def jacobian(self,val):
        ''' return jacobian evaluated at val as a row vector '''
        return jacobianNumerical(lambda x:self.evaluate(x), val)

    def hessian(self,val):
        ''' return hessian evaluated at val '''
        return None

    def _visualize(self,val_vec=None, fun_val_vec=None,dir_vec=None):
        ''' visualize the function 
        Since this is a 3D function, we visualize the x,y plane subject to constraint sin(x) + y + z = 0
        val_vec/fun_val_vec values to highlight
        dir_vec: directions to note for val_vec

        '''
        xx,yy = np.meshgrid(np.linspace(-1,1,100),np.linspace(-1,1,100))
        zz = (0 - np.sin(xx) - yy)
        f_vec = self._evaluate(np.vstack([xx.flatten(), yy.flatten(),zz.flatten()]).T)
        f_vec = f_vec.reshape(xx.shape)

        fig, ax = plt.subplots()
        z_min = np.min(f_vec.flatten())
        z_max = np.max(f_vec.flatten())
        #plt.imshow(val_vec,cmap='RdBu')
        c = ax.pcolormesh(xx,yy,f_vec, cmap='RdBu', vmin=z_min, vmax=z_max)
        fig.colorbar(c,ax=ax)

        # plot val_vec
        if (val_vec is not None):
            ax.plot(val_vec[:,0], val_vec[:,1],'ok')
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

        # DEBUG plot constraint
        xx = self.r_vec*np.cos(np.linspace(0,2*np.pi)) + 0.2
        yy = self.r_vec*np.sin(np.linspace(0,2*np.pi)) + 0.1
        ax.plot(xx,yy)
        return fig

if __name__=='__main__':
    #problem = ParabolaWithSineNoise()
    #problem = PerlinNoise()
    problem = ParabolaWithSineNoise3D()
    problem.visualize(visualize=True)


