#include "sparse_residual_game.h"
#include <cmath>

constexpr int n = 4;
constexpr int m = 2;
using std::cos, std::sin, std::pow, std::max, std::pow;

class Unicycle : public ResidualGame<n, m> {
private:
  Matrix A, B, J_Qr, J_Q, J_R, x_ref;
  Scalar collision_diameter;
  Scalar J_Q_col;

public:
  Unicycle(const int _N, const int _T, const Scalar _dt, const Scalar _rho,
           const Scalar _rho_b, const Scalar _bc_a, const Scalar _bc_b,
           const Scalar _tolerance, const int _backtracking_max_iter,
           const Matrix _J_Qr, const Matrix _J_Q, const Matrix _J_R,
           const Scalar _J_Q_col, const Matrix _x_ref,
           const Scalar _collision_diameter, const int _max_iter,
           const bool _verbose)
      : ResidualGame(_N, _T, _dt, _rho, _rho_b, _bc_a, _bc_b, _tolerance,
                     _backtracking_max_iter, _max_iter, _verbose),
        J_Qr(_J_Qr), J_Q(_J_Q), J_R(_J_R), J_Q_col(_J_Q_col), x_ref(_x_ref),
        collision_diameter(_collision_diameter) {
    /*
    cout << "N = " << N;
    cout << "T = " << T;
    cout << "n = " << n;
    cout << "m = " << m;
    */
  }

  Matrix f(const Matrix x, const Matrix u, const int i) const {
    Matrix dx = (Matrix(n, 1) << x(3, 0) * cos(x(2, 0)), x(3, 0) * sin(x(2, 0)),
                 u(1, 0), u(0, 0))
                    .finished();
    return x + dx * dt;
  }
  Matrix df_dx(const Matrix x, const Matrix u, const int i) const {
    Matrix A = (Matrix(n, n) << 0.0, 0.0, -x(3, 0) * sin(x(2, 0)), cos(x(2, 0)),
                0.0, 0.0, x(3, 0) * cos(x(2, 0)), sin(x(2, 0)), 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0)
                   .finished();

    return Matrix::Identity(n, n) + A * dt;
  }
  Matrix df_du(const Matrix x, const Matrix u, const int i) const {
    Matrix B =
        (Matrix(n, m) << 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0).finished();
    return B * dt;
  }

  // collision constraint function
  Scalar h(const Matrix x_i, const Matrix x_j) const { return -1; }
  Matrix dh_dxi(const Matrix x_i, const Matrix x_j) const {
    return Matrix::Zero(1, n);
  }
  Matrix dh_dxj(const Matrix x_i, const Matrix x_j) const {
    return Matrix::Zero(1, n);
  }
  Matrix dh_dxi_dxi(const Matrix x_i, const Matrix x_j) const {
    return Matrix::Zero(n, n);
  }
  Matrix dh_dxj_dxi(const Matrix x_i, const Matrix x_j) const {
    return Matrix::Zero(n, n);
  }
  Matrix dh_dxi_dxj(const Matrix x_i, const Matrix x_j) const {
    return Matrix::Zero(n, n);
  }
  Matrix dh_dxj_dxj(const Matrix x_i, const Matrix x_j) const {
    return Matrix::Zero(n, n);
  }

  // Objective function (J)
  Matrix J_x_ref_fun(int i) const { return x_ref.row(i).transpose(); }
  Matrix J(const Matrix x_k, const Matrix u_k_i, int i) const {
    Matrix dx = x_k.row(i).transpose() - J_x_ref_fun(i);
    Matrix cost = dx.transpose() * J_Qr * dx +
                  x_k.row(i) * J_Q * x_k.row(i).transpose() +
                  u_k_i.transpose() * J_R * u_k_i;
    // Collision cost
    for (int j = 0; j < N; j++) {
      if (i == j)
        continue;
      Scalar dx = x_k(i, 0) - x_k(j, 0);
      Scalar dy = x_k(i, 1) - x_k(j, 1);
      cost(0, 0) +=
          J_Q_col * max(0.0, pow(collision_diameter, 2) - dx * dx - dy * dy);
    }
    return cost;
  }
  Matrix dJi_dxi(const Matrix x_k, const Matrix u, int i) const {
    Matrix dx = x_k.row(i).transpose() - J_x_ref_fun(i);
    Matrix grad = 2 * dx.transpose() * J_Qr + 2 * x_k.row(i) * J_Q;
    //  Collision cost gradient
    for (int j = 0; j < N; j++) {
      if (i == j)
        continue;
      Scalar dx = x_k(i, 0) - x_k(j, 0);
      Scalar dy = x_k(i, 1) - x_k(j, 1);
      Scalar cost = pow(collision_diameter, 2) - dx * dx - dy * dy;
      if (cost > 0) {
        grad +=
            J_Q_col * (Matrix(1, n) << -2 * dx, -2 * dy, 0.0, 0.0).finished();
      }
    }
    return grad;
  }
  Matrix dJi_dxj(const Matrix x_k, const Matrix u, int i, int j) const {
    Matrix dx = x_k.row(i).transpose() - J_x_ref_fun(i);
    Matrix grad = 2 * dx.transpose() * J_Qr + 2 * x_k.row(i) * J_Q;
    // Collision cost gradient
    for (int j = 0; j < N; j++) {
      if (i == j)
        continue;
      Scalar dx = x_k(i, 0) - x_k(j, 0);
      Scalar dy = x_k(i, 1) - x_k(j, 1);
      Scalar cost = pow(collision_diameter, 2) - dx * dx - dy * dy;
      if (cost > 0) {
        grad += J_Q_col * (Matrix(1, n) << 2 * dx, 2 * dy, 0.0, 0.0).finished();
      }
    }
    return grad;
  }
  Matrix dJi_du(const Matrix x_k, const Matrix u, int i) const {
    return 2 * u.transpose() * J_R;
  }
  Matrix dJi_dxi_dxi(const Matrix x_k, const Matrix u, int i) const {
    Matrix grad = 2 * J_Qr + 2 * J_Q;
    // Collision cost gradient
    for (int j = 0; j < N; j++) {
      if (i == j)
        continue;
      Scalar dx = x_k(i, 0) - x_k(j, 0);
      Scalar dy = x_k(i, 1) - x_k(j, 1);
      Scalar cost = pow(collision_diameter, 2) - dx * dx - dy * dy;
      if (cost > 0) {
        const Eigen::Vector4d diag(2.0, 2.0, 0.0, 0.0);
        grad += diag.asDiagonal();
      }
    }
    return grad;
  }
  Matrix dJi_dxi_dxj(const Matrix x_k, const Matrix u, int i, int j) const {
    return Matrix::Zero(n, n);
  }
  Matrix dJi_dxj_dxj(const Matrix x_k, const Matrix u, int i, int j) const {
    return Matrix::Zero(n, n);
  }
  Matrix dJi_dudu(const Matrix x_k, const Matrix u, int i) const {
    return 2 * J_R;
  }
};
