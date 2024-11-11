# demo to show stien can find two equilibriums
from util import *
from TimeUtil import TimeUtil
from src.build.car_merge_kinematic_bicycle import CarMergeKinematicBicycle as cpp_CarMergeKinematicBicycle
from SteinGame import SteinGame
from ResidualGame import ResidualGame
from CarMergeKinematicBicycle import CarMergeKinematicBicycle

class SteinMerge(CarMergeKinematicBicycle):
    USE_CPP = False
    def __init__(self):
        super().__init__(car_count=2)

        main_lane_n = 1
        merge_lane_n = 2 - main_lane_n
        x_pos_main_lane = np.array( [ 1.5] )
        x_pos_merge_lane = np.array( [ 1.5] )
        v_main_lane = 2.0
        v_merge_lane = 2.0
        x0_main_lane = np.vstack([x_pos_main_lane,self.track_width/2*np.ones(main_lane_n),v_main_lane, np.zeros(main_lane_n)]).T
        x0_merge_lane = np.vstack([x_pos_merge_lane,-self.track_width/2*np.ones(merge_lane_n),v_merge_lane, np.zeros(merge_lane_n)]).T
        self.x0 = np.vstack([x0_main_lane, x0_merge_lane])
        self.target_y = [1]*(main_lane_n+merge_lane_n)

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
    np.random.seed(2)
    main = SteinMerge()
    main.setup()
    u_ref, full_x_ref, has_converged = main.solve(save_gif=False,visualize=False,animate=True)
    main.visualize(u_ref,full_x_ref,visualize=False,save_gif=False,animate=True,gif_prefix='after')
    #main.final()
    print(f'u_ref mean {np.mean(u_ref.flatten())} std {np.std(u_ref.flatten())}')
    #main.testAnimation()

