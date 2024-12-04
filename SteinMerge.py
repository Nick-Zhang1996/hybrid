# demo to show stien can find two equilibriums
from util import *
from TimeUtil import TimeUtil
from src.build.car_merge_kinematic_bicycle import CarMergeKinematicBicycle as cpp_CarMergeKinematicBicycle
from SteinGame import SteinGame
from ResidualGame import ResidualGame
from CarMergeKinematicBicycle import CarMergeKinematicBicycle

class SteinMerge(CarMergeKinematicBicycle):
    USE_CPP = True
    def __init__(self, car_count):
        '''
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
        '''
        super().__init__(car_count)
        self.T = 20
        # multiple car merge, car_count: main_lane_n + merge_lane_n
        main_lane_n = min(int(0.67*car_count),car_count-1)
        merge_lane_n = car_count - main_lane_n
        x_pos_main_lane = np.linspace(0,(main_lane_n-1)*5.4,main_lane_n) + np.random.random(main_lane_n)
        #x_pos_merge_lane = 2.5 + np.linspace(0,(merge_lane_n-1)*5.4,merge_lane_n) + np.random.random(merge_lane_n)
        x_pos_merge_lane = np.linspace(0,(merge_lane_n-1)*5.4,merge_lane_n) + np.random.random(merge_lane_n)
        v_main_lane = 2.0 + np.random.random(main_lane_n)
        v_merge_lane = 2.0 + np.random.random(merge_lane_n)
        x0_main_lane = np.vstack([x_pos_main_lane,self.track_width/2*np.ones(main_lane_n),v_main_lane, np.zeros(main_lane_n)]).T
        x0_merge_lane = np.vstack([x_pos_merge_lane,-self.track_width/2*np.ones(merge_lane_n),v_merge_lane, np.zeros(merge_lane_n)]).T
        self.x0 = np.vstack([x0_main_lane, x0_merge_lane])
        self.target_y = [1]*(main_lane_n+merge_lane_n)

    def setup(self):
        # subclass responsible for loading cpp/eigen module
        # and setting x0
        if (self.USE_CPP or self.CPP_DEBUG):
            self.cpp = cpp_CarMergeKinematicBicycle(self.N, self.T, self.dt, self.rho, self.rho_b, self.bc_a, self.bc_b, self.tolerance, self.backtracking_max_iter, self.J_Qr, self.J_Q, self.J_R, self.h_Qh, self.target_y,self.collision_radius,self.iterations, False)
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
    main = SteinMerge(car_count=5)
    main.setup()
    u_ref, full_x_ref, has_converged = main.solve(save_gif=False,visualize=True,animate=False)
    #main.final()
    print(f'u_ref mean {np.mean(u_ref.flatten())} std {np.std(u_ref.flatten())}')
    #main.testAnimation()

    # test stein game's prediction
    chosen_id = 0
    observed_u_ref = main.belief_support[chosen_id].reshape((main.T,main.N,main.m))
    u_size = observed_u_ref.flatten().shape[0]
    noise = np.random.multivariate_normal(np.zeros(u_size), np.diag([1e-4]*u_size)).reshape(observed_u_ref.shape)
    observed_u_ref += noise

    print(f'showing chosen NE, residual = {main.belief_support_residual[chosen_id]}, social cost = {main.belief_support_cost[chosen_id]}')
    main.visualize(main.belief_support[chosen_id],visualize=True, animate=False,gif_prefix='before')
    # Randomly select a NE for "opponent"
    for k in range(main.T//2):
        print(f'step {k}')
        # send the control of the opponent to bayesian
        main.update(observed_u_ref[k,:,:], k)

        # show top 3 scenarios, are they the NE opponent is using?
        high_likelihood_index = np.argsort(main.belief_weight)[-3:]
        for i in high_likelihood_index:
            print(f'showing top 3: id {i}, prob {main.belief_weight[i]:.4f}, residual = {main.belief_support_residual[i]}, social cost = {main.belief_support_cost[i]}')
            main.visualize(main.belief_support[i],visualize=True, animate=False,gif_prefix='before')

