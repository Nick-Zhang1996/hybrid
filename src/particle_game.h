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
        /*
    def dLLi_du(self,x,u,h_plus_mask,lamda,mu,i):
    def dLLi_dxdx(self,x,u,h_plus_mask,lamda,mu,i):
    def dLLi_dudx(self,x,u,h_plus_mask,lamda,mu,i):

    def r(self, x, u, lamda, mu, h_plus_mask):

    def dBh_dxi(self,x_i,x_j):
    def dBh_dxj(self,x_i,x_j):
    def dBh_dxi_dxi(self,x_i,x_j):
    def dBh_dxi_dxj(self,x_i,x_j):
    def dBh_dxj_dxj(self,x_i,x_j):

    def dF_dx(self,x,u,i,k):
    def dF0_dx(self,x,u,i):

    def dh_dx(self,x,k,i,j):

    def dr_dx(self, x, u, lamda, mu, h_plus_mask):
    def dr_du(self, x, u, lamda, mu, h_plus_mask):
    def dr_dlamda(self, x, u, lamda, mu, h_plus_mask):

    def dLLi_dx_dmu(self,x,u,h_plus_mask,lamda,mu,i):
    def dr_dmu(self, x, u, lamda, mu, h_plus_mask):
    def dr_dy(self, x, u, lamda, mu, h_plus_mask):

    def getHplusMask(self,x):
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
