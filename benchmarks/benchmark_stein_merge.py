""" benchmark SteinMerge, simulate and find collision rate with and without stein"""
import pickle
import numpy as np
from examples.stein_merge import SteinMerge

if __name__ == '__main__':

    def check_collisions(u_ref):
        """ find total collisions with u_ref"""
        x_ref = main.rollout(main.x0, u_ref)
        h_plus_mask = main.getHplusMask(x_ref)
        return np.sum(h_plus_mask)

    def display_results(data):
        """ format and print results"""
        #  no.cars, case 1 - collision % , case 1 - avg collision, case 2...
        #print('  no.cars, case 1 - collision % , case 1 - avg collision, case 2...')
        print(
            '  no.cars, case 4 - collision % , case 5 - avg collision, case 5_alt, 6...'
        )
        for records in data:
            repeats = len(records)
            records = np.array(records)
            _car_count = records[0, 0]
            case1_col_ratio = np.sum(records[:, 1] > 0) / repeats
            case1_avg_col = np.mean(records[:, 1])
            case2_col_ratio = np.sum(records[:, 2] > 0) / repeats
            case2_avg_col = np.mean(records[:, 2])
            case3_col_ratio = np.sum(records[:, 3] > 0) / repeats
            case3_avg_col = np.mean(records[:, 3])
            print(f'{_car_count}\t'
                  f'{case1_col_ratio*100:.0f}%\t,'
                  f'{case1_avg_col:.2f}\t,'
                  f'{case2_col_ratio*100:.0f}%\t,'
                  f'{case2_avg_col:.2f}\t,'
                  f'{case3_col_ratio*100:.0f}%\t,'
                  f'{case3_avg_col:.2f}\t')

    # solve random games with stein
    data = []
    for car_count in range(2, 8):
        # one record for each experiment
        # (car_count, case_1_collision, case_2_col, case_3)
        records = []
        for i in range(100):
            main = SteinMerge(car_count=car_count)
            main.silent_mode_enable()
            main.setup()
            u_ref, full_x_ref, has_converged = main.solve()
            #main.final()
            # Case 1: what if just randomly pick two solutions, no coordination
            # ego agent
            chosen_ego = np.random.randint(0, len(main.belief_support))
            # non-ego, N-1 agents will share this NE
            chosen_nonego = np.random.randint(0, len(main.belief_support))
            ego_u_ref = main.belief_support[chosen_ego].reshape(
                (main.T, main.N, main.m)).copy()
            nonego_u_ref = main.belief_support[chosen_nonego].reshape(
                (main.T, main.N, main.m)).copy()
            # assemble joint-control, we assume agent 0 is ego
            nonego_u_ref[:, 0, :] = ego_u_ref[:, 0, :]
            case_1_collisions = check_collisions(nonego_u_ref)

            # Case 2: what if ego agent use "best" NE from SVG,
            # while non-ego randomly pick a solution
            # ego agent
            chosen_ego = 0
            # non-ego, N-1 agents will share this NE
            chosen_nonego = np.random.randint(0, len(main.belief_support))
            ego_u_ref = main.belief_support[chosen_ego].reshape(
                (main.T, main.N, main.m)).copy()
            nonego_u_ref = main.belief_support[chosen_nonego].reshape(
                (main.T, main.N, main.m)).copy()
            # assemble joint-control, we assume agent 0 is ego
            nonego_u_ref[:, 0, :] = ego_u_ref[:, 0, :]
            case_2_collisions = check_collisions(nonego_u_ref)

            # Case 3: What if ego learn what non-egos are doing (full alg)
            chosen_nonego = np.random.randint(0, len(main.belief_support) // 3)
            observed_u_ref = main.belief_support[chosen_nonego].reshape(
                (main.T, main.N, main.m))

            # adding noise to non-ego's action so learning more challenging
            u_size = observed_u_ref.flatten().shape[0]
            noise = np.random.multivariate_normal(
                np.zeros(u_size),
                np.diag([1e-4] * u_size)).reshape(observed_u_ref.shape)
            observed_u_ref += noise
            ego_u_ref = []
            for k in range(main.T):
                # send the control of the opponent to bayesian,
                # we include ego agent's u_ref here to keep dimension, it's unused
                main.updateBelief(observed_u_ref[k, :, :], k)  # pylint: disable=no-member
                # maximum likelihood NE:w
                max_likelihood_index = np.argsort(main.belief_weight)[-1:]
                this_u_ref = main.belief_support[max_likelihood_index].reshape(
                    (main.T, main.N, main.m)).copy()
                ego_u_ref.append(this_u_ref[k, 0, :])
            ego_u_ref = np.array(ego_u_ref).reshape((main.T, main.m))
            observed_u_ref[:, 0, :] = ego_u_ref
            case_3_collisions = check_collisions(observed_u_ref)

            # Case 4: What if all agents pick their NE at random
            # randomly pick N NE, with replacement
            indices = np.random.randint(0,
                                        len(main.belief_support) // 3,
                                        size=car_count)
            u_ref_4 = np.zeros_like(u_ref)
            # assemble joint-control, we assume agent 0 is ego
            for i in range(car_count):
                u_ref_4[:, i, :] = main.belief_support[indices[i]].reshape(
                    (main.T, main.N, main.m)).copy()[:, i, :]
            case_4_collisions = check_collisions(u_ref_4)

            # Case 5: What if all agents use SVG and use the "Best" NE (N instances)
            # Case 5_alt: what if ego agent and the non-ago agents each use SVG and use best NE
            # (2 instances)
            agent_vec = []
            agent_u_ref = []
            u_ref_5 = np.zeros_like(u_ref)
            for i in range(car_count):
                this_agent = SteinMerge(car_count=car_count)
                this_agent.silent_mode_enable()
                this_agent.setup()
                this_u_ref, full_x_ref, has_converged = this_agent.solve(
                    save_gif=False, visualize=False, animate=False)
                agent_vec.append(this_agent)
                agent_u_ref.append(this_u_ref)
                u_ref_5[:, i, :] = this_u_ref.copy()[:, i, :]
            case_5_collisions = check_collisions(u_ref_5)
            u_ref_5_alt = agent_u_ref[1].copy()
            u_ref_5_alt[:, 0, :] = agent_u_ref[0][:, 0, :]
            case_5_alt_collisions = check_collisions(u_ref_5_alt)

            # Case 6, co-learning, ego agent runs SVG and all
            # non-ago agents run a shared instance of SVG

            # reusing case 5's solution
            # adding noise to agents action so learning more challenging
            chosen_ego = agent_u_ref[0]
            chosen_nonego = agent_u_ref[1]

            # k=0 control
            u_ref_6 = chosen_nonego.copy()
            u_ref_6[0, 0, :] = chosen_ego[0, 0, :]
            for k in range(main.T - 1):
                # send the control of the opponent to bayesian,
                # we include ego agent's u_ref here to keep dimension, it's unused
                agent_vec[0].updateEgo(u_ref_6[k, :, :], k, ego_agent_index=0)
                agent_vec[1].updateNonego(u_ref_6[k, :, :],
                                          k,
                                          ego_agent_index=0)
                # maximum likelihood NE
                max_likelihood_index = np.argsort(
                    agent_vec[0].belief_weight)[-1:]
                ego_u_k = agent_vec[0].belief_support[
                    max_likelihood_index].reshape(
                        (main.T, main.N, main.m)).copy()[k + 1, 0]
                max_likelihood_index = np.argsort(
                    agent_vec[1].belief_weight)[-1:]
                nonego_u_k = agent_vec[1].belief_support[
                    max_likelihood_index].reshape(
                        (main.T, main.N, main.m)).copy()[k + 1, 1:]
                u_ref_6[k + 1, 0] = ego_u_k
                u_ref_6[k + 1, 1:] = nonego_u_k
            case_6_collisions = check_collisions(u_ref_6)

            #print(car_count, case_1_collisions, case_2_collisions, case_3_collisions)
            #records.append((car_count, case_1_collisions, case_2_collisions, case_3_collisions))
            print(car_count, case_4_collisions, case_5_collisions,
                  case_5_alt_collisions, case_6_collisions)
            records.append((car_count, case_4_collisions, case_5_collisions,
                            case_5_alt_collisions, case_6_collisions))
        data.append(records)
        display_results(data)
        with open('benchmark.p', 'wb') as f:
            pickle.dump(data, f)
