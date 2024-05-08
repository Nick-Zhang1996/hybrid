from Problem import Problem
from Solver import *

# example: unstructured lane change
class UnstructuredLaneChange(Problem):
    def __init__(self,car_count=3):
        # u_i = [ax,ay] longitudinal, lateral acceleration
        # x_i = [x,y,vx,vy]
        # x+_i = f(x_i,u_i) = [x + vx*dt + 0.5*ax*dt*dt, y + vy*dt + 0.5*ay*dt*dt ]
        # s.t. [(xi-xj)/dx]**2 + [(yi-yj)/dy]**2 >= 1 
        # agent count: N, time step: 1..T+1
        # X (game state) = concatenated state, first by agent, then by time)
        # state p of agent i at time k: X[k,i,p] or X.flatten()[k*N*m + i*m + p]
        # U (control) = concatenated control  dim: T*N*mm
        # control p of agent i at time k: U[k,i,p] or U.flatten()[k*N*m + i*m + p]

        # Problem formulation
        # decision variables:
        self.N = car_count
        self.T = 20
        self.track_width = 5
        self.track_length = 20
        self.dt = 0.5

        # dimension of u and x for single agent
        self.mm = 2
        self.nn = 4

        # for Problem class
        self.m = car_count*self.mm
        self.n = car_count*self.nn

        # initial state, stated in unit of car size
        self.x0 = x0 = np.array([[0,0,1,0],[2,1,1.4,0],[1,-2,1,0.1]])
        # initial control guess
        U = np.zeros((self.T,self.N,self.mm))
        X = self.rollout(x0,U)

    def rollout(self,x0,U):
        U = U.reshape(self.T,self.N,self.mm)
        X = np.zeros((self.T+1,self.N,self.nn))
        X[0,:,:] = x0.reshape(self.N,self.nn)
        # x+ = x + vx*dt + 0.5*ax*dt*dt
        # vx+ = vx + ax*dt
        for i in range(self.N):
            for k in range(1,self.T+1):
                X[k,i,0] = X[k-1,i,0] + X[k-1,i,2]*self.dt + 0.5*self.dt*self.dt*U[k-1,i,0]
                X[k,i,2] = X[k-1,i,2] + self.dt*U[k-1,i,0]/10
                X[k,i,1] = X[k-1,i,1] + X[k-1,i,3]*self.dt + 0.5*self.dt*self.dt*U[k-1,i,1]
                X[k,i,3] = X[k-1,i,3] + self.dt*U[k-1,i,1]/10
        return X[1:,:,:]





    # TODO
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

    def visualize(self,U):
        X = np.vstack([self.x0[np.newaxis,:,:],self.rollout(self.x0,U)])
        plt.vlines(x=-self.track_width/2,ymin=-1,ymax=self.track_length)
        plt.vlines(x=self.track_width/2,ymin=-1,ymax=self.track_length)
        for i in range(self.N):
            xx = X[:,i,0]
            yy = X[:,i,1]
            plt.plot(yy,xx,'*-')
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.show()
        return

if __name__=="__main__":
    main = UnstructuredLaneChange(3)
    U = np.zeros((main.T,main.N,main.mm))
    U[:,0,0] = 1.0
    U[:,1,1] = 1.0
    main.visualize(U)
