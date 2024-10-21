#include "car_merge_kinematic_bicycle.h"

__global__
std::tuple<std::vector<Matrix>,std::vector<Matrix>,std::vector<Matrix>,std::vector<Matrix>,bool>
naive_particle_solve(const int _N, const int _T,
                const Scalar _dt, const Scalar _rho, const Scalar _rho_b, const Scalar _bc_a, const Scalar _bc_b, const Scalar _tolerance, const int _backtracking_max_iter, Matrix x0){
    // TODO generate random in_u
    std::vector<Matrix> u(T,Matrix(N,m));
    std::vector<Matrix> x(T,Matrix(N,n));
    std::vector<Matrix> lamda(T,Matrix::Zero(N,n));
    std::vector<Matrix> mu(T,Matrix::Zero(N,N));
    bool has_converged;

    ResidualGame game(_N, _T, _dt, _rho, _rho_b, _bc_a, _bc_b, _tolerance,_backtracking_max_iter);
    game.set_x0(x0);
    std::tie(x,u,lamda, mu) = game.solve(u);
}

int main(){
    int N = 2;
    int n = 4;
    Matrix x0 = (Matrix(N,n) << 0.0, 1.0, 2.0, 0, 
            1.0, 0.0, 2.0, 0).finished();
    naive_particle_solve(N,20,0.02,100,1.0, 0.1, 0.5,1e-3, 10, x0);


}
