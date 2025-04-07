# benchmark stein merge, simulate and find collision rate with and without stein
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pickle
from examples.SteinMerge import SteinMerge

def checkCollisions(u_ref):
    x_ref = main.rollout(main.x0,u_ref)
    h_plus_mask = main.getHplusMask(x_ref)
    return np.sum(h_plus_mask)


def displayResults(data):
    #  no.cars, case 1 - collision % , case 1 - avg collision, case 2...
    print('  no.cars, case 1 - collision % , case 1 - avg collision, case 2...')
    for records in data:
        repeats = len(records)
        records = np.array(records)
        car_count = records[0,0]
        case1_col_ratio = np.sum(records[:,1]>0)/repeats
        case1_avg_col = np.mean(records[:,1])
        case2_col_ratio = np.sum(records[:,2]>0)/repeats
        case2_avg_col = np.mean(records[:,2])
        case3_col_ratio = np.sum(records[:,3]>0)/repeats
        case3_avg_col = np.mean(records[:,3])
        print(f'{car_count}\t, {case1_col_ratio*100:.0f}%\t, {case1_avg_col:.2f}\t,{case2_col_ratio*100:.0f}%\t, {case2_avg_col:.2f}\t,{case3_col_ratio*100:.0f}%\t, {case3_avg_col:.2f}\t')




if __name__=='__main__':
    # solve random games with stein
    data = []
    for car_count in range(2,8):
        # one record for each experiment
        # (car_count, case_1_collision, case_2_col, case_3)
        records = []
        for i in range(100):
            main = SteinMerge(car_count=car_count)
            main.silent_mode_enable()
            main.setup()
            u_ref, full_x_ref, has_converged = main.solve(save_gif=False,visualize=True,animate=False)
            #main.final()

            # Case 1: what if just randomly pick two solutions
            # ego agent
            chosen_ego = np.random.randint(0,len(main.belief_support))
            # non-ego, N-1 agents will share this NE
            chosen_nonego = np.random.randint(0,len(main.belief_support))
            ego_u_ref = main.belief_support[chosen_ego].reshape((main.T,main.N,main.m)).copy()
            nonego_u_ref = main.belief_support[chosen_nonego].reshape((main.T,main.N,main.m)).copy()
            # assemble joint-control, we assume agent 0 is ego
            nonego_u_ref[:,0,:] = ego_u_ref[:,0,:]
            case_1_collisions = checkCollisions(nonego_u_ref)

            # Case 2: what if ego agent use "best" NE from SVG, while non-ego randomly pick a solution
            # ego agent
            chosen_ego = 0
            # non-ego, N-1 agents will share this NE
            chosen_nonego = np.random.randint(0,len(main.belief_support))
            ego_u_ref = main.belief_support[chosen_ego].reshape((main.T,main.N,main.m)).copy()
            nonego_u_ref = main.belief_support[chosen_nonego].reshape((main.T,main.N,main.m)).copy()
            # assemble joint-control, we assume agent 0 is ego
            nonego_u_ref[:,0,:] = ego_u_ref[:,0,:]
            case_2_collisions = checkCollisions(nonego_u_ref)


            # Case 3: What if ego learn what non-egos are doing (full alg)
            chosen_nonego = np.random.randint(0,len(main.belief_support)//3)
            observed_u_ref = main.belief_support[chosen_nonego].reshape((main.T,main.N,main.m))

            # adding noise to non-ego's action so learning more challenging
            u_size = observed_u_ref.flatten().shape[0]
            noise = np.random.multivariate_normal(np.zeros(u_size), np.diag([1e-4]*u_size)).reshape(observed_u_ref.shape)
            observed_u_ref += noise
            ego_u_ref = []
            for k in range(main.T):
                # send the control of the opponent to bayesian, we include ego agent's u_ref here to keep dimension, it's unused
                main.update(observed_u_ref[k,:,:], k)
                # maximum likelihood NE:w
                max_likelihood_index = np.argsort(main.belief_weight)[-1:]
                this_u_ref = main.belief_support[max_likelihood_index].reshape((main.T,main.N,main.m)).copy()
                ego_u_ref.append(this_u_ref[k,0,:])
            ego_u_ref = np.array(ego_u_ref).reshape((main.T,main.m))
            observed_u_ref[:,0,:] = ego_u_ref
            case_3_collisions = checkCollisions(observed_u_ref)

            print(car_count, case_1_collisions, case_2_collisions, case_3_collisions)
            records.append((car_count, case_1_collisions, case_2_collisions, case_3_collisions))
        data.append(records)
        displayResults(data)
        with open('benchmark.p', 'wb') as f:
            pickle.dump(data,f)

