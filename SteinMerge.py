# demo to show stien can find two equilibriums
from util import *
from TimeUtil import TimeUtil
from src.build.car_merge_kinematic_bicycle import CarMergeKinematicBicycle as cpp_CarMergeKinematicBicycle
from SteinGame import SteinGame
from ResidualGame import ResidualGame
from CarMergeKinematicBicycle import CarMergeKinematicBicycle

class SteinMerge(CarMergeKinematicBicycle):
    USE_CPP = True
    def __init__(self):
        super().__init__(car_count=2)

        main_lane_n = 1
        merge_lane_n = 2 - main_lane_n
        x_pos_main_lane = np.array( [ 1.5] )
        x_pos_merge_lane = np.array( [ 1.5] )
        v_main_lane = 2.0
        v_merge_lane = 2.0
        x0_main_lane = np.vstack([x_pos_main_lane,self.track_width/2*np.ones(main_lane_n)+0.2,v_main_lane, np.zeros(main_lane_n)]).T
        x0_merge_lane = np.vstack([x_pos_merge_lane,-self.track_width/2*np.ones(merge_lane_n)-0.2,v_merge_lane, np.zeros(merge_lane_n)]).T
        self.x0 = np.vstack([x0_main_lane, x0_merge_lane])
        self.target_y = [1]*(main_lane_n+merge_lane_n)

    def setup(self):
        # subclass responsible for loading cpp/eigen module
        # and setting x0
        if (self.USE_CPP or self.CPP_DEBUG):
            self.cpp = cpp_CarMergeKinematicBicycle(self.N, self.T, self.dt, self.rho, self.rho_b, self.bc_a, self.bc_b, self.tolerance, self.backtracking_max_iter, self.J_Qr, self.J_Q, self.J_R, self.h_Qh, self.target_y,self.iterations, False)
            self.cpp.set_x0(self.x0)


    def Testsolve(self,save_gif=False,visualize=False,animate=False):
        dim_u = (self.T, self.N, self.m)
        u_ref = np.zeros(dim_u)
        #u_ref[:,1,0] = 0.5
        #u_ref[:,0,0] = -0.4
        #u_ref[:,1,0] = -0.5
        #u_ref[:,0,0] = 0.4

        u_ref, full_x_ref, has_converged = ResidualGame.solve(self,u_ref = u_ref.reshape(dim_u))
        return u_ref, full_x_ref, has_converged

if __name__=="__main__":
    #np.random.seed(2)
    main = SteinMerge()
    main.setup()
    u_ref, full_x_ref, has_converged = main.solve(save_gif=False,visualize=True,animate=False)
    #main.final()
    print(f'u_ref mean {np.mean(u_ref.flatten())} std {np.std(u_ref.flatten())}')
    #main.testAnimation()

