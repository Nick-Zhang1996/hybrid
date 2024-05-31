#pragma once

#include <iostream>
using std::endl;
using std::cout;

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
            return -sqr(x_i.coeff(0,0)-x_j.coeff(0,0)) - sqr(x_i.coeff(0,1)-x_j.coeff(0,1)) + sqr(1.2);
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
            (mtx << 0,target_y.coeff(i,0), 2.0, 0.0 ).finished();
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

        // TODO what's a 1d/3d array converted to?
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
        /*

        np_array dL_dx_ik(const np_array x_k,const np_array  u_k_i,const np_array  x_k1_i,const np_array h_k_plus_mask,const np_array lamda_k,const np_array mu_k,const int i):

        val =  dJ_dx(x_k[i],u_k_i,i) + lamda_k[i].T @ self.df_dx(x_k[i],u_k_i)
        val += np.sum( [ mu_k[i,j.item()] * ( self.dh_dxi(x_k[i], x_k[j.item()]) ) for j in np.nonzero(h_k_plus_mask[i])[0] ], axis=0)
        val += -1/self.rho*np.sum([min(1/self.h(x_k[i], x_k[j.item()]),1e10) * self.dh_dxi(x_k[i], x_k[j.item()]) * (j.item() != i) for j in np.nonzero(~h_k_plus_mask[i])[0] ],axis=0)
        return val

    def dL_dx_ik1(self,x_k, u_k_i, x_k1_i, h_k_plus_mask,lamda_k, mu_k,i):
        ''' dL/dx_i_k+1 '''
        return -lamda_k[i].T
    def dL_dx_jk(self,x_k, u_k_i, x_k1_i, h_k_plus_mask,lamda_k, mu_k,i,j):
        assert (i!=j)
        return  mu_k[i,j] * ( self.dh_dxj(x_k[i], x_k[j]) ) if h_k_plus_mask[i,j] else \
            -1/self.rho*min(1/self.h(x_k[i], x_k[j]),1e10) * self.dh_dxi(x_k[i], x_k[j])
    def dL_du(self,x_k, u_k_i, x_k1_i, h_k_plus_mask,lamda_k, mu_k,i):
        return self.dJ_du(x_k[i],u_k_i,i) + lamda_k[i].T @ self.df_du(x_k[i], u_k_i)
        */
        /*
    def f(self,x,u):
    def df_dx(self,x,u):
    def df_du(self,x,u):

    def h(self, x_i, x_j):
    def dh_dxi(self,x_i,x_j):
    def dh_dxj(self,x_i,x_j):
    def dh_dxi_dxi(self,x_i,x_j):
    def dh_dxj_dxi(self,x_i,x_j):
    def dh_dxi_dxj(self,x_i,x_j):
    def dh_dxj_dxj(self,x_i,x_j):

    def J(self,x,u,i):
    def dJ_dx(self,x,u,i):
    def dJ_du(self,x,u,i):
    def dJ_dxdx(self,x,u,i):


    def L(self,x_k, u_k_i, x_k1_i, h_k_plus_mask,lamda_k, mu_k,i):
    def dL_dx_ik(self,x_k, u_k_i, x_k1_i, h_k_plus_mask,lamda_k, mu_k,i):
    def dL_dx_ik1(self,x_k, u_k_i, x_k1_i, h_k_plus_mask,lamda_k, mu_k,i):
    def dL_dx_jk(self,x_k, u_k_i, x_k1_i, h_k_plus_mask,lamda_k, mu_k,i,j):
    def dL_du(self,x_k, u_k_i, x_k1_i, h_k_plus_mask,lamda_k, mu_k,i):

    def LLi(self,x,u,h_plus_mask,lamda,mu,i):
    def dLLi_dx(self,x,u,h_plus_mask,lamda,mu,i):
    def dLLi_du(self,x,u,h_plus_mask,lamda,mu,i):
    def dLLi_dxdx(self,x,u,h_plus_mask,lamda,mu,i):
        #dLLi_dxki_dxki, k=T, u_T is undefined, use 0 to penalize J(x) only
    def dLLi_dudx(self,x,u,h_plus_mask,lamda,mu,i):

    def r(self, x, u, lamda, mu, h_plus_mask):
    def r_fillin(self, x, u, lamda, mu, h_plus_mask):

    def Bh(self,x_i,x_j):
    def dBh_dxi(self,x_i,x_j):
    def dBh_dxj(self,x_i,x_j):
    def dBh_dxi_dxi(self,x_i,x_j):
    def dBh_dxi_dxj(self,x_i,x_j):
    def dBh_dxj_dxj(self,x_i,x_j):

    def dF_dx(self,x,u,i,k):
    def dF0_dx(self,x,u,i):

    def dh_dx(self,x,k,i,j):

    def dr_dx(self, x, u, lamda, mu, h_plus_mask):
    def dr_dx_old(self, x, u, lamda, mu, h_plus_mask):
    def dr_du(self, x, u, lamda, mu, h_plus_mask):
    def dr_dlamda(self, x, u, lamda, mu, h_plus_mask):

    def dLLi_dx_dmu(self,x,u,h_plus_mask,lamda,mu,i):
    def dr_dmu(self, x, u, lamda, mu, h_plus_mask):
    def dr_dy(self, x, u, lamda, mu, h_plus_mask):

    def getHplusMask(self,x):
       */

};
