#pragma once

#include <iostream>
#include <pybind11/stl.h>
using std::endl;
using std::cout;
using std::min;

typedef Eigen::MatrixXd np_array;

inline float sqr(const float a){
    return a*a;
}

class ParticleGame {
    private:
        np_array A,B,J_Qr,J_Q,J_R,h_Qh,target_y;
        int N,T,n,m;
        float dt,rho,rho_b,bc_a,bc_b;


    public:
        ParticleGame(const int _N, const int _T, const int _n, const int _m,
                const float _dt, const float _rho, const float _rho_b, const float _bc_a, const float _bc_b,
                const np_array _J_Qr, const np_array _J_Q, const np_array _J_R, const np_array _A, const np_array _B, const np_array _h_Qh, const np_array _target_y):
            N(_N), T(_T), n(_n), m(_m),
            dt(_dt), rho(_rho), rho_b(_rho_b),bc_a(_bc_a), bc_b(_bc_b),
            J_Qr(_J_Qr), J_Q(_J_Q), J_R(_J_R), A(_A), B(_B), h_Qh(_h_Qh), target_y(_target_y) {
        }
        // TODO unnecessary copy
        // TODO fixed dimension arrays
        void set_A(const Eigen::MatrixXd &val){
            A = np_array(val);
        }
        void set_B(const Eigen::MatrixXd &val){
            B = np_array(val);
        }

        np_array f(const np_array x, const np_array u){
            return A * x + B * u;
        }
        np_array df_dx(const np_array x, const np_array u){
            return A;
        }
        np_array df_du(const np_array x, const np_array u){
            return B;
        }


        float h(const np_array x_i, const np_array x_j){
            return -sqr(x_i(0,0)-x_j(0,0)) - sqr(x_i(1,0)-x_j(1,0)) + sqr(1.2);
        }
        np_array dh_dxi(const np_array x_i, const np_array x_j){
            return 2*(x_i-x_j).transpose() * h_Qh;
        }
        np_array dh_dxj(const np_array x_i, const np_array x_j){
            return 2*(x_j-x_i).transpose() * h_Qh;
        }
        np_array dh_dxi_dxi(const np_array x_i, const np_array x_j){
            return 2*h_Qh.transpose();
        }
        np_array dh_dxj_dxi(const np_array x_i, const np_array x_j){
            return -2*h_Qh.transpose();
        }
        np_array dh_dxi_dxj(const np_array x_i, const np_array x_j){
            return -2*h_Qh.transpose();
        }
        np_array dh_dxj_dxj(const np_array x_i, const np_array x_j){
            return 2*h_Qh.transpose();
        }

        np_array J_x_ref_fun(int i){
            Eigen::MatrixXd mtx(n,1);
            (mtx << 0,target_y(i,0), 2.0, 0.0 ).finished();
            return mtx;
        }
        np_array J(const np_array x, const np_array u, int i){
            return (x-J_x_ref_fun(i)).transpose() * J_Qr * (x-J_x_ref_fun(i)) + x.transpose() * J_Q * x + u.transpose() * J_R * u;
        }
        np_array dJ_dx(const np_array x, const np_array u, int i){
            return  2* (x-J_x_ref_fun(i)).transpose() * J_Qr + 2*x.transpose() * J_Q;
        }
        np_array dJ_du(const np_array x, const np_array u, int i){
            return  2* u.transpose() * J_R;
        }
        np_array dJ_dxdx(const np_array x, const np_array u, int i){
            return  2*J_Qr + 2*J_Q;
        }


        // x_k: dim: N*n, u_k_i: dim:m*1, lambda_k:N*n, h_k_plus_mask: N*N, mu_k dim:N*N
        np_array dL_dx_ik(const np_array x_k,const np_array  u_k_i,const np_array  x_k1_i,const np_array h_k_plus_mask,const np_array lamda_k,const np_array mu_k,const int i){
            np_array val =  dJ_dx(x_k.row(i).transpose(),u_k_i,i) + lamda_k.row(i) * df_dx(x_k.row(i).transpose(),u_k_i);
            for (int j=0; j<N; j++){
                if (i==j){continue;}
                if (h_k_plus_mask(i,j)){
                    val +=  mu_k(i,j) * ( dh_dxi(x_k.row(i).transpose(), x_k.row(j).transpose()) );
                } else {
                    val += -1.0/rho*min(1.0/h(x_k.row(i).transpose(), x_k.row(j).transpose()),1e10) * dh_dxi(x_k.row(i).transpose(), x_k.row(j).transpose());
                }
            }
            return val;
        }

        np_array dL_du(const np_array x_k, const np_array u_k_i, const np_array x_k1_i, const np_array h_k_plus_mask,const np_array lamda_k, const np_array mu_k,const int i){
            return dJ_du(x_k.row(i).transpose(),u_k_i,i) + lamda_k.row(i) * df_du(x_k.row(i).transpose(), u_k_i);
        }

        // for 3d array, first dimension is list() -> std::vector
        //x_i_k: 1..T, T*N*n  NOTE starts from 1
        //u_i_k: 0..T-1, T*N*m
        //lamda_i_k: 0..T-1 T*N*n
        //mu_k_i_j: 1..T T*N*N NOTE starts from 1
        np_array dLLi_dx(const std::vector<np_array> x,const std::vector<np_array> u,const std::vector<np_array> h_plus_mask,const std::vector<np_array> lamda,const std::vector<np_array> mu,const int i){
            // TODO is this the best approach?
            Eigen::MatrixXd der(1,T*N*n);
            der.setZero();
            // dLLi_dxi
            for (int k=1; k<T; k++){
                der.block(0,(k-1)*N*n+i*n,1,n) = dL_dx_ik(x[k-1],u[k,i],x[k].row(i).transpose(),h_plus_mask[k-1],lamda[k],mu[k-1],i) -lamda[k-1].row(i);
            }
            // dLLi_dxi_T
            der.block(0,(T-1)*N*n+i*n,1,n) = -lamda[T-1].row(i) + dJ_dx(x[T-1].row(i).transpose(),Eigen::MatrixXd::Zero(m,1),i);
            for (int j=0; j<N; j++){
                if (i==j){continue;}
                if (h_plus_mask[T-1](i,j)){
                der.block(0,(T-1)*N*n+i*n,1,n) +=  mu[T-1](i,j) *  dh_dxi(x[T-1].row(i).transpose(), x[T-1].row(j).transpose());
                } else {
                der.block(0,(T-1)*N*n+i*n,1,n) += -1/rho*min(1.0/h(x[T-1].row(i).transpose(), x[T-1].row(j).transpose()),1e10)*dh_dxi(x[T-1].row(i).transpose(),x[T-1].row(j).transpose());
                }
            }
            // dLLi_dxj
            for (int j=0; j<N; j++){
                if (i==j){continue;}
                for (int k=1; k<T+1; k++){
                    if(h_plus_mask[k-1](i,j)){
                        der.block(0,(k-1)*N*n+j*n,1,n) = mu[k-1](i,j) * dh_dxj(x[k-1].row(i).transpose(), x[k-1].row(j).transpose());
                    } else {
                        der.block(0,(k-1)*N*n+j*n,1,n) = -1.0/rho*min(1.0/h(x[k-1].row(i).transpose(), x[k-1].row(j).transpose()),1e10)*dh_dxj(x[k-1].row(i).transpose(), x[k-1].row(j).transpose());
                    }
                }
            }
            return der;
        }

        // TODO remember to set x0
        np_array dLLi_du(const std::vector<np_array>& x, const std::vector<np_array>& u, const std::vector<np_array>& h_plus_mask, const std::vector<np_array>& lamda, const std::vector<np_array>& mu, const int i) {
            np_array der(1, T * N* m); // Initialize derivative vector as row vector
            der.setZero(); // Ensure all items are properly zero-initialized

            // dLLi_dui_0
            der.block(0, i * m, 1, m) = dJ_du(x0.row(i).transpose(), u[0].row(i).transpose(), i) + lamda[0].row(i) * df_du(x0.row(i).transpose(), u[0].row(i).transpose());

            // dLLi_dui_k
            for (int k = 1; k < T; ++k) {
                der.block(0, i * m + k * N * m, 1, m) = dJ_du(x[k - 1].row(i).transpose(), u[k].row(i).transpose(), i) + lamda[k].row(i) * df_du(x[k - 1].row(i).transpose(), u[k].row(i).transpose());
            }

            return der;
        }

np_array dBh_dxi(const np_array& x_i, const np_array& x_j) {
    // Calculate dBh/dxi
    return -1.0 / (rho * h(x_i, x_j)) * dh_dxi(x_i, x_j);
}

np_array dBh_dxj(const np_array& x_i, const np_array& x_j) {
    // Calculate dBh/dxj
    return -1.0 / (rho * h(x_i, x_j)) * dh_dxj(x_i, x_j);

}
np_array dBh_dxi_dxi(const np_array& x_i, const np_array& x_j) {
    // Calculate h, dh/dxi, and dhdxi_dxi
    np_array h = h(x_i, x_j);
    np_array dhdxi = dh_dxi(x_i, x_j);
    np_array dhdxi_dxi = dh_dxi_dxi(x_i, x_j);

    // Calculate dBh/dxi_dxi
    np_array val = 1.0 / (rho * h) * (-dhdxi_dxi + 1.0 / h * dhdxi.transpose() * dhdxi);
    return val;
}
np_array dBh_dxi_dxj(const np_array& x_i, const np_array& x_j) {
    // Calculate h, dh/dxi, and dhdxi_dxi
    np_array h = h(x_i, x_j);
    np_array dhdxi = dh_dxi(x_i, x_j);
    np_array dhdxi_dxj = dh_dxi_dxj(x_i, x_j);

    // Calculate dBh/dxi_dxi
    np_array val = 1.0 / (rho * h) * (-dhdxi_dxj + 1.0 / h * dhdxi.transpose() * dhdxi);
    return val;
}
np_array dBh_dxj_dxj(const np_array& x_i, const np_array& x_j) {
    // Calculate h, dh/dxi, and dhdxi_dxi
    np_array h = h(x_i, x_j);
    np_array dhdxj = dh_dxj(x_i, x_j);
    np_array dhdxj_dxj = dh_dxj_dxj(x_i, x_j);

    // Calculate dBh/dxi_dxi
    np_array val = 1.0 / (rho * h) * (-dhdxj_dxj + 1.0 / h * dhdxj.transpose() * dhdxi);
    return val;
}
np_array dF_dx(const std::vector<np_array>& x, const std::vector<np_array>& u, const int i, const int k) {
    // Initialize dF_dx matrix
    np_array dFdx = np_array::Zero(n, T * N * n);

    // Calculate indices for insertion
    int start_idx_1 = (k - 1) * N * n + i * n;
    int start_idx_2 = k * N * n + i * n;

    // Assign values to dF_dx
    dFdx.block(0, start_idx_1, n, n) = df_dx(x[k - 1].row(i).transpose(), u[k].row(i).transpose());
    dFdx.block(0, start_idx_2, n, n) = -np_array::Identity(n, n);

    return dFdx;
}

np_array dF0_dx(const std::vector<np_array>& x, const std::vector<np_array>& u, const int i) {
    // Initialize dF_dx matrix
    np_array dFdx = np_array::Zero(n, T * N * n);
    dFdx.block(0, i*n, n, n) = -np_array::Identity(n, n);

    return dFdx;
}

np_array dh_dx(const std::vector<np_array>& x, const int k, const int i, const int j) {
    // Initialize dh_dx matrix
    np_array dhdx = np_array::Zero(1, T * N * n);

    // Calculate indices for insertion
    int start_idx_1 = (k - 1) * N * n + i * n;
    int start_idx_2 = (k - 1) * N * n + j * n;

    // Assign values to dh_dx
    dhdx.block(0, start_idx_1, 1, n) = dh_dxi(x[k - 1].row(i).transpose(), x[k - 1].row(j).transpose());
    dhdx.block(0, start_idx_2, 1, n) = dh_dxj(x[k - 1].row(i).transpose(), x[k - 1].row(j).transpose());

    return dhdx;
}


np_array dLLi_dxdx(const std::vector<np_array>& x, const std::vector<np_array>& u, const std::vector<np_array>& h_plus_mask, const std::vector<np_array>& lamda, const std::vector<np_array>& mu, const int i) {
    np_array dLL_dxdx = np_array::Zero(T*N*n, T*N*n);

    auto submtx = [&](int k, int i, int j) {
        return dLL_dxdx.block((k - 1) * N * n + i * n, (k - 1) * N * n + j * n, n, n);
    };

    for (int k = 1; k < T; ++k) {
        auto mtx = submtx(k, i, i);
        mtx = dJ_dxdx(x[k - 1].row(i).transpose(), u[k].row(i).transpose(), i);
        for (int j=0; j<N; ++j){
            if (i==j){continue;}
            if (h_plus_mask[k-1](i,j)){
                mtx += mu[k - 1](i,j) * dh_dxi_dxi(x[k - 1].row(i).transpose(), x[k - 1].row(j).transpose());
            } else {
                mtx += dBh_dxi_dxi(x[k - 1].row(i).transpose(), x[k - 1].row(j).transpose());
            }
        }
    }

    auto mtx = submtx(T, i, i);
    mtx = dJ_dxdx(x[T - 1].row(i).transpose(), np_array::Zero(m,1), i);

    for (int j=0; j<N; ++j){
        if (i==j){continue;}
        if (h_plus_mask[T-1](i,j)){
            mtx += mu[T - 1](i,j) * dh_dxi_dxi(x[T - 1].row(i).transpose(), x[T - 1].row(j).transpose());
        } else {
            mtx += dBh_dxi_dxi(x[T - 1].row(i).transpose(), x[T - 1].row(j).transpose());
        }
    }

    for (int k = 1; k <= T; ++k) {
        for (int j = 0; j < N; ++j) {
            if (i == j) {
                continue;
            }
            auto val = np_array::Zero(n, n);
            if (h_plus_mask[k-1](i,j)){
                val = mu[k - 1](i,j) * dh_dxi_dxj(x[k - 1].row(i).transpose(), x[k - 1].row(j).transpose());
                submtx(k,j,j) = mu[k - 1](i,j) * dh_dxj_dxj(x[k - 1].row(i).transpose(), x[k - 1].row(j).transpose());
            } else {
                val = dBh_dxi_dxj(x[k - 1].row(i).transpose(), x[k - 1].row(j).transpose());
                submtx(k,j,j) = dBh_dxj_dxj(x[k - 1].row(i).transpose(), x[k - 1].row(j).transpose());
            }
            submtx(k,i,j) = val;
            submtx(k,j,i) = val.transpose();
        }
    }
    return dLL_dxdx;
}

np_array dr_dx(const std::vector<np_array>& x, const std::vector<np_array>& u, const std::vector<np_array>& lamda, const std::vector<np_array>& mu, const std::vector<np_array>& h_plus_mask) {
    // Calculate dimensions
    int dim_x = N * T * n;
    int dim_u = N * T * m;
    int h_plus_sum = 0;
    for (const auto& mask: h_plus_mask){
        h_plus_sum += mask.count();
    }
    int dim_r = N * (dim_x + dim_u + T * n) + h_plus_sum;

    // Initialize dr_dx matrix
    np_array drdx = np_array::Zero(dim_r, dim_x);

    // Initialize index
    int index = 0;

    for (int i = 0; i < N; ++i) {
        // Calculate dLL_dxdx
        np_array dLL_dxdx = dLLi_dxdx(x, u, h_plus_mask, lamda, mu, i);
        drdx.block(index, 0, dim_x, dim_x) = dLL_dxdx;
        index += dim_x + dim_u;

        np_array dF0dx = dF0_dx(x, u, i);
        drdx.block(index, 0, n, dim_x) = dF0dx;

        for (int k = 1; k < T; ++k) {
            np_array dFdx = dF_dx(x, u, i, k);
            drdx.block(index + n * k, 0, n, dim_x) = dFdx;
        }
        index += n*T;

        for (int k = 1; k <= T; ++k) {
            np_array dhdx = np_array::Zero(h_plus_mask[k-1].row(i).count(), dim_x);
            int dh_dx_idx = 0;
            for (int j = 0; j < N; ++j) {
                if (h_plus_mask[k-1](i,j)){
                    dhdx.row(dh_dx_idx) = dh_dx(x, k, i, j);
                    dh_dx_idx++;
                }
            }
            drdx.block(index, 0, dhdx.rows(), dim_x) = dhdx;
            index += dhdx.rows();
        }
    }
    return drdx;
}




/*
    np_array dr_du( x, u, lamda, mu, h_plus_mask){
        ''' return: dim(r)*dim(u) '''
        T = T; N = N; n = n; m = m
        dim_x = T*N*n; dim_u = T*N*m
        dim_r = N*(dim_x+dim_u+T*n)+np.sum(h_plus_mask)

        drdu = np.zeros((dim_r,dim_u))
        index = 0
        for i in range(N):
            index += dim_x
            for k in range(T):
                dLL_duik_duik = 2*J_R
                drdu[index+k*N*m+i*m:index+k*N*m+(i+1)*m, k*N*m+i*m:k*N*m+(i+1)*m] = dLL_duik_duik
            index += dim_u
            k = 0
            drdu[index+k*n:index+(k+1)*n, k*N*m+i*m:k*N*m+(i+1)*m] = df_du(x0[i],u[k,i])
            for k in range(1,T):
                drdu[index+k*n:index+(k+1)*n, k*N*m+i*m:k*N*m+(i+1)*m] = df_du(x[k-1,i],u[k,i])
            index += n*T + np.sum(h_plus_mask[:,i]) # skip  f(x,u)-x+,  h(x,x)

        return drdu

    }
    np_array dr_dlamda( x, u, lamda, mu, h_plus_mask){
        ''' return: dim(r)*dim(lamda) '''
        T = T; N = N; n = n; m = m
        dim_x = T*N*n; dim_u = T*N*m ; dim_lamda = T*N*n
        dim_r = N*(dim_x+dim_u+T*n)+np.sum(h_plus_mask)
        dr_dlamda = np.zeros((dim_r,dim_lamda))
        index = 0
        for i in range(N):
            for k in range(1,T):
                # dLLi_dxki_dlamda_ki
                dr_dlamda[index+(k-1)*N*n+i*n:index+(k-1)*N*n+(i+1)*n,k*N*n+i*n:k*N*n+(i+1)*n] = df_dx(x[k-1,i],u[k,i]).T
                # dLLi_dxki_dlamda_k-1,i
                dr_dlamda[index+(k-1)*N*n+i*n:index+(k-1)*N*n+(i+1)*n,(k-1)*N*n+i*n:(k-1)*N*n+(i+1)*n] = -np.eye(n)
            k = T
            dr_dlamda[index+(k-1)*N*n+i*n:index+(k-1)*N*n+(i+1)*n,(k-1)*N*n+i*n:(k-1)*N*n+(i+1)*n] = -np.eye(n)

            index += dim_x # skip dLL_dx, index now points at dLLi_du
            for k in range(0,T):
                dr_dlamda[index+k*N*m+i*m:index+k*N*m+(i+1)*m,k*N*n+i*n:k*N*n+(i+1)*n] = df_du(x[k-1,i],u[k,i]).T

            index += dim_u + n*T + np.sum(h_plus_mask[:,i]) # skip  dLL_du, f(x,u)-x+,  h(x,x)

        if (DEBUG):
            dr_dlamda_num = jacobianNumerical(lambda ll:r(x,u,ll.reshape(lamda.shape),mu,h_plus_mask), lamda.flatten(),dim=dim_r)
            ifprint(f'dr_dlamda err {np.linalg.norm(dr_dlamda-dr_dlamda_num)}')
            diff = dr_dlamda_num - dr_dlamda
            assert(np.linalg.norm(dr_dlamda-dr_dlamda_num)<1e-4)
        return dr_dlamda

    }
    np_array dLLi_dx_dmu(x,u,h_plus_mask,lamda,mu,i){
        ''' return: dim: dim_x*dim_mu '''
        T = T; N = N; n = n; m = m
        dim_x = T*N*n; dim_u = T*N*m
        dim_mu = T*N*N
        dLL_dx_dmu = np.zeros((dim_x,dim_mu))
        for k in range(1,T+1):
            for j in np.nonzero(h_plus_mask[k-1,i])[0]:
                dLLi_dxki_dmuijk = dh_dxi(x[k-1,i],x[k-1,j])
                dLLi_dxkj_dmuijk = dh_dxj(x[k-1,i],x[k-1,j])
                dLL_dx_dmu[(k-1)*N*n+i*n:(k-1)*N*n+(i+1)*n,(k-1)*N*N+i*N+j] = dLLi_dxki_dmuijk
                dLL_dx_dmu[(k-1)*N*n+j*n:(k-1)*N*n+(j+1)*n,(k-1)*N*N+i*N+j] = dLLi_dxkj_dmuijk
        return dLL_dx_dmu

    # TODO this is untested
    }
    np_array dr_dmu( x, u, lamda, mu, h_plus_mask){
        ''' return: dim(r)*dim(mu) '''
        T = T; N = N; n = n; m = m
        dim_x = T*N*n; dim_u = T*N*m
        dim_r = N*(dim_x+dim_u+T*n)+np.sum(h_plus_mask)
        dim_mu = T*N*N

        dr_dmu = np.zeros((dim_r,dim_mu))
        index = 0
        for i in range(N):
            # dmu i,j,k
            dLL_dx_dmu = dLLi_dx_dmu(x,u,h_plus_mask,lamda,mu,i)
            dr_dmu[index:index+dim_x,:] = dLL_dx_dmu
            if (DEBUG):
                dLL_dx_dmu_num = jacobianNumerical(lambda mm:dLLi_dx(x,u,h_plus_mask,lamda,mm.reshape(mu.shape),i), mu.flatten(),dim=dim_x)
                assert(np.linalg.norm(dLL_dx_dmu-dLL_dx_dmu_num)<1e-4)

            index += dim_x + dim_u + n*T + np.sum(h_plus_mask[:,i])

        if (DEBUG):
            dr_dmu_num = jacobianNumerical(lambda mm:r(x,u,lamda,mm.reshape(mu.shape),h_plus_mask), mu.flatten(),dim=dim_r)
            ifprint(f'drdx err {np.linalg.norm(dr_dmu-dr_dmu_num)}')
            assert(np.linalg.norm(dr_dmu-dr_dmu_num)<1e-4)
        return dr_dmu

    }
    np_array dr_dy( x, u, lamda, mu, h_plus_mask){
        t.s('drdx')
        drdx = dr_dx(x, u, lamda, mu, h_plus_mask)
        t.e('drdx')
        t.s('drdu')
        drdu = dr_du(x, u, lamda, mu, h_plus_mask)
        t.e('drdu')
        t.s('drdlamda')
        drdlamda = dr_dlamda(x, u, lamda, mu, h_plus_mask)
        t.e('drdlamda')
        t.s('drdmu')
        drdmu = dr_dmu(x, u, lamda, mu, h_plus_mask)
        t.e('drdmu')
        t.s('stack')
        Dr = np.hstack([drdx,drdu,drdlamda,drdmu])
        t.e('stack')
        return Dr

    }
    np_array getHplusMask(x){
        # h(i,i) should not be considered in either h_plus or h_minus
        # we check it in h_minux
        # for x 1-T, NOTE index start from 1
        h_plus_mask = np.zeros((T,N,N),dtype=bool)
        for k in range(1,T+1):
            for i in range(N):
                for j in range(i+1,N):
                    h_plus_mask[k-1,i,j] = h_plus_mask[k-1,j,i] = h(x[k-1,i],x[k-1,j]) >= 0
        return h_plus_mask
       */

        // TODO move to end
        void print_dim(const np_array val){
            std::cout << "rows " << val.rows() << "cols " << val.cols() << endl;
        }
        np_array test_bool_array(const np_array val, const np_array mask){
            Eigen::MatrixXd output(val);
            for (int i=0; i<val.rows(); i++){
                for (int j=0; j<val.cols(); j++){
                    if (!mask(i,j)){
                        output(i,j) = 0;
                    }
                }
            }
            return output;
        }
        np_array three_dim(const std::vector<np_array> mtx_vec){
            return mtx_vec[1];
        }
};
