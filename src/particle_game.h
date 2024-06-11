#pragma once
//#define EIGEN_RUNTIME_NO_MALLOC 
//Eigen::internal::set_is_malloc_allowed(false);

#include <iostream>
#include <pybind11/stl.h>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SparseCore>

// sparse solvers
#include <Eigen/OrderingMethods>
#include <Eigen/SparseQR>
using SolverClassName = Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>>;
// for LeastSquaresConjugateGradient
#include<Eigen/IterativeLinearSolvers>


using std::endl;
using std::cout;
using std::min;
// TODO fix h_plus_sum re-calculation
// TODO add fill-in style api
// TODO use template size for n,m, T,N
// TODO add step
// TODO add solve
// TODO make program self-independent
// TODO block

typedef Eigen::MatrixXd np_array;
using Eigen::MatrixBase;
using Eigen::SparseMatrix;
// TODO change all double 
using Scalar = double;

inline double sqr(const Scalar a){
    return a*a;
}

class ParticleGame {
    private:
        np_array A,B,J_Qr,J_Q,J_R,h_Qh,target_y,x0;
        int N,T,n,m;
        double dt,rho,rho_b,bc_a,bc_b;


    public:
        ParticleGame(const int _N, const int _T, const int _n, const int _m,
                const double _dt, const double _rho, const double _rho_b, const double _bc_a, const double _bc_b,
                const np_array _J_Qr, const np_array _J_Q, const np_array _J_R, const np_array _A, const np_array _B, const np_array _h_Qh, const np_array _target_y):
            N(_N), T(_T), n(_n), m(_m),
            dt(_dt), rho(_rho), rho_b(_rho_b),bc_a(_bc_a), bc_b(_bc_b),
            J_Qr(_J_Qr), J_Q(_J_Q), J_R(_J_R), A(_A), B(_B), h_Qh(_h_Qh), target_y(_target_y),x0() {
        }
        // TODO unnecessary copy
        // TODO fixed dimension arrays
        void set_A(const np_array &val){
            A = np_array(val);
        }
        void set_B(const np_array &val){
            B = np_array(val);
        }
        void set_x0(const np_array &val){
            x0 = np_array(val);
        }
        void post_step_update(){
            rho *= rho_b;
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


        double h(const np_array x_i, const np_array x_j){
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

        // NOTE this is dependent upon the car
        np_array J_x_ref_fun(int i){
            np_array mtx(n,1);
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
            np_array der(1,T*N*n);
            der.setZero();
            // dLLi_dxi
            for (int k=1; k<T; k++){
                der.block(0,(k-1)*N*n+i*n,1,n) = dL_dx_ik(x[k-1],u[k,i],x[k].row(i).transpose(),h_plus_mask[k-1],lamda[k],mu[k-1],i) -lamda[k-1].row(i);
            }
            // dLLi_dxi_T
            der.block(0,(T-1)*N*n+i*n,1,n) = -lamda[T-1].row(i) + dJ_dx(x[T-1].row(i).transpose(),np_array::Zero(m,1),i);
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
            double h_val = h(x_i, x_j);
            np_array dhdxi = dh_dxi(x_i, x_j);
            np_array dhdxi_dxi = dh_dxi_dxi(x_i, x_j);

            // Calculate dBh/dxi_dxi
            np_array val = 1.0 / (rho * h_val) * (-dhdxi_dxi + 1.0 / h_val * dhdxi.transpose() * dhdxi);
            return val;
        }
        np_array dBh_dxi_dxj(const np_array& x_i, const np_array& x_j) {
            // Calculate h, dh/dxi, and dhdxi_dxi
            double h_val = h(x_i, x_j);
            np_array dhdxi = dh_dxi(x_i, x_j);
            np_array dhdxj = dh_dxj(x_i, x_j);
            np_array dhdxi_dxj = dh_dxi_dxj(x_i, x_j);

            // Calculate dBh/dxi_dxi
            np_array val = 1.0 / (rho * h_val) * (-dhdxi_dxj + 1.0 / h_val * dhdxi.transpose() * dhdxj);
            return val;
        }
        np_array dBh_dxj_dxj(const np_array& x_i, const np_array& x_j) {
            // Calculate h, dh/dxi, and dhdxi_dxi
            double h_val = h(x_i, x_j);
            np_array dhdxj = dh_dxj(x_i, x_j);
            np_array dhdxj_dxj = dh_dxj_dxj(x_i, x_j);

            // Calculate dBh/dxi_dxi
            np_array val = 1.0 / (rho * h_val) * (-dhdxj_dxj + 1.0 / h_val * dhdxj.transpose() * dhdxj);
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
                    np_array val = np_array::Zero(n, n);
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

        np_array dLLi_dx_dmu(const std::vector<np_array>& x, const std::vector<np_array>& u, const std::vector<np_array>& h_plus_mask, const std::vector<np_array>& lamda, const std::vector<np_array>& mu, int i) {
            // Calculate dimensions
            const int dim_x = T * N * n;
            const int dim_u = T * N * m;
            const int dim_mu = T * N * N;

            // Initialize dLL_dx_dmu matrix
            np_array dLL_dx_dmu = np_array::Zero(dim_x, dim_mu);

            for (int k = 1; k < T+1; ++k) {
                for (int j=0; j<N; j++) {
                    if (h_plus_mask[k-1](i,j)){
                        np_array dLLi_dxki_dmuijk = dh_dxi(x[k - 1].row(i).transpose(), x[k - 1].row(j).transpose());
                        np_array dLLi_dxkj_dmuijk = dh_dxj(x[k - 1].row(i).transpose(), x[k - 1].row(j).transpose());

                        dLL_dx_dmu.block((k - 1) * N * n + i * n, (k - 1) * N * N + i * N + j, n, 1) = dLLi_dxki_dmuijk.transpose();
                        dLL_dx_dmu.block((k - 1) * N * n + j * n, (k - 1) * N * N + i * N + j, n, 1) = dLLi_dxkj_dmuijk.transpose();
                    }
                }
            }

            return dLL_dx_dmu;
        }

        template<typename Derived>
        void dr_dx(const std::vector<np_array>& x, const std::vector<np_array>& u, const std::vector<np_array>& lamda, const std::vector<np_array>& mu, const std::vector<np_array>& h_plus_mask, const MatrixBase<Derived>& mtx) {
            int dim_x = N * T * n;
            int dim_u = N * T * m;
            auto& drdx = const_cast<MatrixBase<Derived>&>(mtx);
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
                        //if (i==j){continue;}
                        if (h_plus_mask[k-1](i,j)){
                            dhdx.row(dh_dx_idx) = dh_dx(x, k, i, j);
                            dh_dx_idx++;
                        }
                    }
                    // TODO we could set the original matrix directly to avoid copying
                    drdx.block(index, 0, dhdx.rows(), dim_x) = dhdx;
                    index += dhdx.rows();
                }
            }
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
            dr_dx(x, u, lamda, mu, h_plus_mask, drdx);
            return drdx;
        }

        template<typename Derived>
        void dr_du(const std::vector<np_array>& x, const std::vector<np_array>& u, const std::vector<np_array>& lamda, const std::vector<np_array>& mu, const std::vector<np_array>& h_plus_mask, const MatrixBase<Derived>& mtx) {
            auto& drdu = const_cast<MatrixBase<Derived>&>(mtx);
            int dim_x = N * T * n;
            int dim_u = N * T * m;
            int index = 0;
            for (int i = 0; i < N; ++i) {
                index += dim_x;
                for (int k = 0; k < T; ++k) {
                    // dLL_duik_duik
                    np_array dLL_duik_duik = 2 * J_R;
                    drdu.block(index + k * N * m + i * m, k * N * m + i * m, m, m) = dLL_duik_duik;
                }
                index += dim_u;

                drdu.block(index + 0 * n, 0 * N * m + i * m, n, m) = df_du(x0.row(i).transpose(), u[0].row(i).transpose());
                for (int k = 1; k < T; ++k) {
                    drdu.block(index + k * n, k * N * m + i * m, n, m) = df_du(x[k - 1].row(i).transpose(), u[k].row(i).transpose());
                }
                // skip count for h_plus_mask[all k, i, all j]
                int skip_count = 0;
                for (int k = 1; k < T+1; ++k) {
                    skip_count +=h_plus_mask[k-1].row(i).count();
                }
                index += n * T + skip_count;
            }
        }

        np_array dr_du(const std::vector<np_array>& x, const std::vector<np_array>& u, const std::vector<np_array>& lamda, const std::vector<np_array>& mu, const std::vector<np_array>& h_plus_mask) {
            // Calculate dimensions
            int dim_x = N * T * n;
            int dim_u = T * N * m;
            int h_plus_sum = 0;
            for (const auto& mask : h_plus_mask) {
                h_plus_sum += mask.count();
            }
            int dim_r = N * (dim_x + dim_u + T * n) + h_plus_sum;
            np_array drdu = np_array::Zero(dim_r, dim_u);
            dr_du(x, u, lamda, mu, h_plus_mask, drdu);
            return drdu;
        }

        template<typename Derived>
        void dr_dlamda(const std::vector<np_array>& x, const std::vector<np_array>& u, const std::vector<np_array>& lamda, const std::vector<np_array>& mu, const std::vector<np_array>& h_plus_mask, const MatrixBase<Derived>& mtx) {
            auto& drdlamda = const_cast<MatrixBase<Derived>&>(mtx);
            int dim_x = T * N * n;
            int dim_u = T * N * m;
            int index = 0;
            for (int i = 0; i < N; ++i) {
                for (int k = 1; k < T; ++k) {
                    // dLLi_dxki_dlamda_ki
                    drdlamda.block(index + (k - 1) * N * n + i * n, k * N * n + i * n, n, n) = df_dx(x[k - 1].row(i).transpose(), u[k].row(i).transpose()).transpose();
                    // dLLi_dxki_dlamda_k-1,i
                    drdlamda.block(index + (k - 1) * N * n + i * n, (k - 1) * N * n + i * n, n, n) = -np_array::Identity(n, n);
                }

                const int k = T;
                drdlamda.block(index + (k - 1) * N * n + i * n, (k - 1) * N * n + i * n, n, n) = -np_array::Identity(n, n);
                // skip dLL_dx, index now points at dLLi_du
                index += dim_x;

                drdlamda.block(index + 0 * N * m + i * m, 0 * N * n + i * n, m, n) = df_du(x0.row(i).transpose(), u[0].row(i).transpose()).transpose();
                for (int k = 1; k < T; ++k) {
                    drdlamda.block(index + k * N * m + i * m, k * N * n + i * n, m, n) = df_du(x[k-1].row(i).transpose(), u[k].row(i).transpose()).transpose();
                }

                // skip count for h_plus_mask[all k, i, all j]
                int skip_count = 0;
                for (int k = 1; k < T+1; ++k) {
                    skip_count +=h_plus_mask[k-1].row(i).count();
                }
                index += dim_u + n * T + skip_count;
            }

        }
        np_array dr_dlamda(const std::vector<np_array>& x, const std::vector<np_array>& u, const std::vector<np_array>& lamda, const std::vector<np_array>& mu, const std::vector<np_array>& h_plus_mask) {
            // Calculate dimensions
            int dim_x = T * N * n;
            int dim_u = T * N * m;
            int dim_lamda = T * N * n;
            int h_plus_sum = 0;
            for (const auto& mask : h_plus_mask) {
                h_plus_sum += mask.count();
            }
            int dim_r = N * (dim_x + dim_u + T * n) + h_plus_sum;
            np_array drdlamda = np_array::Zero(dim_r, dim_lamda);
            dr_dlamda(x, u, lamda, mu, h_plus_mask, drdlamda);
            return drdlamda;
        }


        template<typename Derived>
        void dr_dmu(const std::vector<np_array>& x, const std::vector<np_array>& u, const std::vector<np_array>& lamda, const std::vector<np_array>& mu, const std::vector<np_array>& h_plus_mask, const MatrixBase<Derived>& mtx) {
            auto& drdmu = const_cast<MatrixBase<Derived>&>(mtx);
            const int dim_x = T * N * n;
            const int dim_u = T * N * m;
            const int dim_mu = T * N * N;
            int index = 0;
            for (int i = 0; i < N; ++i) {
                // Calculate dLL_dx_dmu
                np_array dLL_dx_dmu = dLLi_dx_dmu(x, u, h_plus_mask, lamda, mu, i);
                drdmu.block(index, 0, dim_x, dim_mu) = dLL_dx_dmu;
                // skip count for h_plus_mask[all k, i, all j]
                int skip_count = 0;
                for (int k = 1; k < T+1; ++k) {
                    skip_count +=h_plus_mask[k-1].row(i).count();
                }
                index += dim_x + dim_u + n * T + skip_count;
            }

        }
        np_array dr_dmu(const std::vector<np_array>& x, const std::vector<np_array>& u, const std::vector<np_array>& lamda, const std::vector<np_array>& mu, const std::vector<np_array>& h_plus_mask) {
            // Calculate dimensions
            const int dim_x = T * N * n;
            const int dim_u = T * N * m;
            const int dim_mu = T * N * N;
            int h_plus_sum = 0;
            for (const auto& mask : h_plus_mask) {
                h_plus_sum += mask.count();
            }
            int dim_r = N * (dim_x + dim_u + T * n) + h_plus_sum;

            // Initialize dr_dmu matrix
            np_array drdmu = np_array::Zero(dim_r, dim_mu);
            dr_dmu(x, u, lamda, mu, h_plus_mask, drdmu);
            return drdmu;
        }

        std::vector<np_array> getHplusMask(const std::vector<np_array>& x) {
             std::vector<np_array> h_plus_mask(T);
             for (int i=0; i<T; i++){
                 h_plus_mask[i]= np_array::Zero(N, N);
             }

            for (int k = 1; k < T+1; ++k) {
                for (int i = 0; i < N; ++i) {
                    for (int j = i + 1; j < N; ++j) {
                        h_plus_mask[k-1](i, j) = h_plus_mask[k-1](j, i) = h(x[k-1].row(i).transpose(), x[k-1].row(j).transpose()) >= 0;
                    }
                }
            }
            return h_plus_mask;
        }

        np_array r(const std::vector<np_array>& x, const std::vector<np_array>& u, const std::vector<np_array>& lamda, const std::vector<np_array>& mu, const std::vector<np_array>& h_plus_mask) {
            const int dim_x = N * T * n;
            const int dim_u = N * T * m;
            int h_plus_sum = 0;
            for (const auto& mask : h_plus_mask) {
                h_plus_sum += mask.count();
            }
            const int dim_r = N * (dim_x + dim_u + T * n) + h_plus_sum;

            np_array r = np_array::Zero(dim_r,1);
            int index = 0;
            for (int i = 0; i < N; ++i) {
                np_array dLL_dx = dLLi_dx(x, u, h_plus_mask, lamda, mu, i).transpose();
                np_array dLL_du = dLLi_du(x, u, h_plus_mask, lamda, mu, i).transpose();
                r.block(index, 0, dim_x, 1) = dLL_dx;
                index += dim_x;
                r.block(index, 0, dim_u, 1) = dLL_du;
                index += dim_u;

                // Dynamics for f(x0,u0) = x1
                r.block(index, 0, n, 1) = f(x0.row(i).transpose(), u[0].row(i).transpose()) - x[0].row(i).transpose();

                for (int k = 1; k < T; ++k) {
                    r.block(index+k*n,0,n,1) = f(x[k - 1].row(i).transpose(), u[k].row(i).transpose()) - x[k].row(i).transpose();
                }
                index += n * T;

                for (int k = 1; k < T+1; ++k) {
                    int h_idx = 0;
                    for (int j=0; j<N; ++j){
                        if (h_plus_mask[k-1](i,j)){
                            r(index,0) = h(x[k - 1].row(i).transpose(), x[k - 1].row(j).transpose());
                            h_idx++;
                        }
                    }
                    index += h_idx;
                }
            }
            return r;
        }

        np_array dr_dy(const std::vector<np_array>& x, const std::vector<np_array>& u, const std::vector<np_array>& lamda, const std::vector<np_array>& mu, const std::vector<np_array>& h_plus_mask) {
            int h_plus_sum = 0;
            for (const auto& mask : h_plus_mask) {
                h_plus_sum += mask.count();
            }
            const int dim_x = N * T * n;
            const int dim_u = N * T * m;
            const int dim_lamda = T * N * n;
            const int dim_mu = T * N * N;
            const int dim_r = N * (dim_x + dim_u + T * n) + h_plus_sum;

            const int dim_y = dim_x + dim_u + dim_lamda + dim_mu;
            np_array Dr(dim_r,dim_y);
            Dr.setZero();

            dr_dx(x, u, lamda, mu, h_plus_mask, Dr.block(0,0,dim_r,dim_x));
            dr_du(x, u, lamda, mu, h_plus_mask, Dr.block(0,dim_x,dim_r,dim_u));
            dr_dlamda(x, u, lamda, mu, h_plus_mask, Dr.block(0,dim_x+dim_u,dim_r,dim_lamda));
            dr_dmu(x, u, lamda, mu, h_plus_mask, Dr.block(0,dim_x+dim_u+dim_lamda,dim_r,dim_mu));
            return Dr;
        }

        std::vector<std::vector<np_array>> step(const std::vector<np_array>& x, const std::vector<np_array>& u, const std::vector<np_array>& lamda, const std::vector<np_array>& mu) {
            const auto h_plus_mask = getHplusMask(x);
            int h_plus_sum = 0;
            for (const auto& mask : h_plus_mask) {
                h_plus_sum += mask.count();
            }


            const int dim_x = N * T * n;
            const int dim_u = N * T * m;
            const int dim_lamda = T * N * n;
            const int dim_mu = T * N * N;
            const int dim_r = N * (dim_x + dim_u + T * n) + h_plus_sum;
            const int dim_y = dim_x + dim_u + dim_lamda + dim_mu;

            auto r0 = r(x, u, lamda, mu, h_plus_mask);
            const np_array Dr = dr_dy(x, u, lamda, mu, h_plus_mask);
            // print dimension of everything
            //cout << "Dr: " << Dr.rows() << " * " << Dr.cols() << endl;
            // get nonzero terms, reduce matrix dimension
            std::vector<int> nonzero_rows_idx;
            std::vector<int> nonzero_cols_idx;
            std::tie(nonzero_rows_idx, nonzero_cols_idx) = nonzeros(Dr);
            //cout << "nonzero" << endl;


            np_array Dr_reduced = Dr(nonzero_rows_idx, nonzero_cols_idx);
            auto Dr_reduced_sparse = Dr_reduced.sparseView();
            //cout << "sparseview" << endl;
            np_array r0_reduced = r0(nonzero_rows_idx,Eigen::all);
            //cout << "Dr " << Dr.rows() << " " << Dr.cols() << endl;
            //cout << "r0 " << r0.rows() << " " << r0.cols() << endl;
            //cout << "Dr_reduced " << Dr_reduced.rows() << " " << Dr_reduced.cols() << endl;
            //cout << "r0_reduced " << r0_reduced.rows() << " " << r0_reduced.cols() << endl;

            SolverClassName solver;
            // solve r0 + Dr* dy = 0 least square
            solver.compute(Dr_reduced_sparse);
            //solver.compute(Dr.sparseView());
            //cout << "compute" << endl;

            if (solver.info() != Eigen::Success){
                cout << " solver initialization failed" << endl;
                return std::vector<std::vector<np_array>>();
            }

            // FIXME this fails
            np_array dy_reduced = solver.solve(-r0_reduced);
            //np_array dy_reduced = solver.solve(-r0);
            //cout << "solve" << dy_reduced.maxCoeff() << endl;
            if (solver.info() != Eigen::Success){
                cout << " solver solve failed" << endl;
                return std::vector<std::vector<np_array>>();
            }
            // reconstruct dy from dy_reduced
            // FIXME debug printout
            //cout << "Dr (reduced): " << Dr_reduced_sparse.rows() << " * " << Dr_reduced_sparse.cols() << endl;
            //cout << "dim_y " << dim_y << endl;
            //cout << "max col idx " << *std::max_element(nonzero_cols_idx.begin(), nonzero_cols_idx.end()) << endl;

            np_array dy(dim_y,1);
            dy.setZero();
            dy(nonzero_cols_idx,Eigen::all) = dy_reduced;
            std::vector<np_array> dummy{dy};
            return std::vector<std::vector<np_array>>{dummy};

            // line search
            Scalar step = 1.0; // step size
            Scalar r0_norm = r0.norm();
            Scalar rt_norm = r0_norm;
            auto split_y = [&](const std::vector<np_array>& x, const std::vector<np_array>& u, const std::vector<np_array>& lamda, const std::vector<np_array>& mu, const np_array& dy, Scalar my_step){
                const auto x_size = x.size();
                std::vector<np_array> xx(x_size);
                for (int i=0; i<x_size; i++){
                    xx.at(i) = x.at(i) + my_step * dy.block(i*N*n,0,N*n,1).reshaped(N,n);
                }
                const auto u_size = u.size();
                std::vector<np_array> uu(x_size);
                for (int i=0; i<u_size; i++){
                    uu.at(i) = u.at(i) + my_step * dy.block(dim_x+i*N*m,0,N*m,1).reshaped(N,m);
                }
                const auto lamda_size = lamda.size();
                std::vector<np_array> ll(lamda_size);
                for (int i=0; i<lamda_size; i++){
                    ll.at(i) = lamda.at(i) + my_step * dy.block(dim_x+dim_u+i*N*n,0,N*n,1).reshaped(N,n);
                }
                const auto mu_size = mu.size();
                std::vector<np_array> mm(mu_size);
                for (int i=0; i<mu_size; i++){
                    mm.at(i) = mu.at(i) + my_step * dy.block(dim_x+dim_u+dim_lamda+i*N*N,0,N*N,1).reshaped(N,N);
                }
                return std::tuple<std::vector<np_array>,std::vector<np_array>,std::vector<np_array>,std::vector<np_array>> {xx, uu, ll, mm};
            };


            auto r_t_norm = [&](Scalar my_step){
                auto y_tuple = split_y(x, u, lamda, mu, dy, my_step);
                return r(std::get<0>(y_tuple), std::get<1>(y_tuple), std::get<2>(y_tuple),std::get<3>(y_tuple), h_plus_mask).norm();
            };

            for (int i=0; i<10; i++){
                rt_norm = r_t_norm(step);
                if (rt_norm > (1-bc_a*step)*r0_norm){
                    step *= bc_b;
                } else {
                    break;
                }
            }

            // stopping criteria
            auto y_tuple = split_y(x, u, lamda, mu, dy, step);
            if ( abs(rt_norm - r0_norm) < 5e-4 and h_plus_sum == 0){
                // stopping
                return std::vector<std::vector<np_array>>();
            } else {
                auto xx = std::get<0>(y_tuple);
                auto uu = std::get<1>(y_tuple);
                auto ll = std::get<2>(y_tuple);
                auto mm = std::get<3>(y_tuple);
                return std::vector<std::vector<np_array>>{xx,uu,ll,mm};
            }
        }


        // --- helper function ---
        // find nonzero submatrix
        // return: skimmed matrix (dense), nonzero row indices, nonzero col indices
        std::tuple<std::vector<int>, std::vector<int>>
        nonzeros(const Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>& mtx){
            // cols
            Eigen::Matrix<bool,1,Eigen::Dynamic> nonzero_cols_mask = mtx.cast<bool>().colwise().any();
            const int nonzero_cols_size = nonzero_cols_mask.cast<int>().sum();
            std::vector<int> nonzero_cols_idx;
            nonzero_cols_idx.reserve(nonzero_cols_size);
            const int mtx_cols = mtx.cols();
            for (int i=0; i<mtx_cols; i++){
                if (nonzero_cols_mask(0,i)){
                nonzero_cols_idx.push_back(i);
                }
            }
            // rows
            Eigen::Matrix<bool,1,Eigen::Dynamic> nonzero_rows_mask = mtx.cast<bool>().rowwise().any();
            const int nonzero_rows_size = nonzero_rows_mask.cast<int>().sum();
            std::vector<int> nonzero_rows_idx;
            nonzero_rows_idx.reserve(nonzero_rows_size);
            const int mtx_rows = mtx.rows();
            for (int i=0; i<mtx_rows; i++){
                if (nonzero_rows_mask(0,i)){
                nonzero_rows_idx.push_back(i);
                }
            }

            return {nonzero_rows_idx, nonzero_cols_idx};
        }

        // DEBUG functions

        // solve Ax=B
        np_array SparseQR(const np_array& A, const np_array& B){
            Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;
            solver.compute(A.sparseView());
            if (solver.info() != Eigen::Success){
                cout << " solver initialization failed" << endl;
                return np_array{};
            }

            np_array x = solver.solve(B);
            if (solver.info() != Eigen::Success){
                cout << " solver solve failed" << endl;
                return np_array{};
            }
            return x;
        }

        np_array LeastSquaresConjugateGradient(const np_array& A, const np_array& B){
            Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double>> solver;
            solver.compute(A.sparseView());
            if (solver.info() != Eigen::Success){
                cout << " solver initialization failed" << endl;
                return np_array{};
            }

            // solver.setMaxIterations();
            // solver.setTolerance
            np_array x = solver.solve(B);
            if (solver.info() != Eigen::Success){
                cout << " solver solve failed" << endl;
                return np_array{};
            }
            return x;
        }

        void print_dim(const np_array val){
            std::cout << "rows " << val.rows() << "cols " << val.cols() << endl;
        }
        np_array test_bool_array(const np_array val, const np_array mask){
            np_array output(val);
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
        // doesn't work unfortunately
        void pass_by_ref(std::vector<np_array>& array){
            // multiply the first array value by 2
            array[0] *= 2;
            // multiply the first array value by 0.5
            array[1] *= 0.5;
            return;
        }
};
