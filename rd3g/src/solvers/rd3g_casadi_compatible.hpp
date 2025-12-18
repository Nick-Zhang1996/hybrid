// RD3G solver compatible with casadi codegen J(), Jfi(), f(), h() etc.
#pragma once
// #define EIGEN_RUNTIME_NO_MALLOC
// Eigen::internal::set_is_malloc_allowed(false);
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SparseCore>
#include <fstream>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string.h>
#include <string>

namespace py = pybind11;

// get virtual memory currently used by this process
// #include "stdlib.h"
// #include "stdio.h"
// #include "string.h"

// sparse solvers
#include <Eigen/OrderingMethods>
#include <Eigen/SparseQR>
// for LeastSquaresConjugateGradient
#include <Eigen/IterativeLinearSolvers>

#include "profiler.h"

using Scalar = double;
using Eigen::MatrixBase;
using std::cout;
using std::endl;
using std::max;
using std::min;
// NOTE has to be RowMajor
using Matrix =
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
// SpMatrix was taken
using SpMatrix = Eigen::SparseMatrix<double, Eigen::ColMajor>;

template <typename Derived> void checksum(const MatrixBase<Derived> &mtx) {
  Eigen::Index maxIndex;
  float maxNorm = mtx.rowwise().sum().maxCoeff(&maxIndex);
  std::cout << "Maximum sum at position " << maxIndex << std::endl;
  std::cout << "its sum is is: " << maxNorm << std::endl;

  Eigen::Index minIndex;
  float minNorm = mtx.rowwise().sum().minCoeff(&minIndex);
  std::cout << "Minimum sum at position " << minIndex << std::endl;
  std::cout << "its sum is is: " << minNorm << std::endl;
}

inline Scalar sqr(const Scalar a) { return a * a; }

int getCurrentMemoryUsageInKB() { // Note: this value is in KB!
  std::ifstream file("/proc/self/status");
  std::string line;
  int memory_usage = 0;

  while (std::getline(file, line)) {
    if (line.rfind("VmSize", 0) == 0) {
      std::size_t startPos = line.find_first_of("0123456789");
      std::size_t endPos = line.find(" kB");
      memory_usage = std::stoi(line.substr(startPos, endPos - startPos));
      break;
    }
  }
  return memory_usage;
}

// assign a dense matrix to a sub-block of a sparse matrix
// this function assumes there's no existing entries in the sparse matrix, it
// uses SpMatrix.insert() for addition, use sp_add()
template <typename Derived>
void sp_assign(const MatrixBase<Derived> &in_mtx, SpMatrix &out_mtx,
               const int row_offset, const int col_offset, const int row_size,
               const int col_size) {
  // maybe we can avoid creating this variable?
  const SpMatrix sp_in_mtx = in_mtx.sparseView();
  for (int k = 0; k < sp_in_mtx.outerSize(); ++k) {
    for (SpMatrix::InnerIterator it(sp_in_mtx, k); it; ++it) {
      out_mtx.insert(row_offset + it.row(), col_offset + it.col()) = it.value();
      // we should check that we are not inserting outside the target block,
      // creating a memory leak, but only for debug build
    }
  }
}

template <typename Derived>
void sp_add(const MatrixBase<Derived> &in_mtx, SpMatrix &out_mtx,
            const int row_offset, const int col_offset, const int row_size,
            const int col_size) {
  // maybe we can avoid creating this variable?
  const SpMatrix sp_in_mtx = in_mtx.sparseView();
  for (int k = 0; k < sp_in_mtx.outerSize(); ++k) {
    for (SpMatrix::InnerIterator it(sp_in_mtx, k); it; ++it) {
      out_mtx.coeffRef(row_offset + it.row(), col_offset + it.col()) +=
          it.value();
      // we should check that we are not inserting outside the target block,
      // creating a memory leak, but only for debug build
    }
  }
}

std::tuple<SpMatrix, std::vector<int>>
remove_empty_cols(SpMatrix &matrix, const int reserve_size) {
  //  Identify non-empty columns
  std::vector<int> nonEmptyCols;
  nonEmptyCols.reserve(reserve_size);
  for (int j = 0; j < matrix.cols(); ++j) {
    if (matrix.col(j).nonZeros() > 0) {
      nonEmptyCols.push_back(j);
    }
  }

  //  Create a new temporary matrix with non-empty columns
  SpMatrix tempMatrix(matrix.rows(), nonEmptyCols.size());
  for (int newColIdx = 0; newColIdx < nonEmptyCols.size(); ++newColIdx) {
    int oldColIdx = nonEmptyCols[newColIdx];
    tempMatrix.col(newColIdx) = matrix.col(oldColIdx);
  }

  return {tempMatrix, nonEmptyCols};
}

template <int n, int m> class ResidualGame {

protected:
  int N, T;
  Scalar dt, rho, rho_b, bc_a, bc_b;
  Scalar tolerance;
  int backtracking_max_iter;
  int max_iterations;
  // dim: N*n
  Matrix x0;
  mutable Profiler<false> profiler;
  int current_memory_usage_kb;
  bool verbose;

public:
  ResidualGame(const int _N, const int _T, const Scalar _dt, const Scalar _rho,
               const Scalar _rho_b, const Scalar _bc_a, const Scalar _bc_b,
               const Scalar _tolerance, const int _backtracking_max_iter,
               const int _max_iter, const bool _verbose)
      : N(_N), T(_T), dt(_dt), rho(_rho), rho_b(_rho_b), bc_a(_bc_a),
        bc_b(_bc_b), tolerance(_tolerance),
        backtracking_max_iter(_backtracking_max_iter), x0(), profiler(),
        current_memory_usage_kb(0), max_iterations(_max_iter),
        verbose(_verbose) {
    // current_memory_usage_kb = getCurrentMemoryUsageInKB();
    // std::cout << "existing memory usage " << current_memory_usage_kb << "KB"
    // << std::endl;
  }

  void set_x0(const Matrix &val) { x0 = Matrix(val); }
  void post_step_update() { rho *= rho_b; }

  // x_k: dim: N*n, u_k_i: dim:m*1, lambda_k:N*n, h_k_plus_mask: N*N, mu_k
  // dim:N*N
  Matrix dL_dx_ik(const Matrix x_k, const Matrix u_k_i, const Matrix x_k1_i,
                  const Matrix h_k_plus_mask, const Matrix lamda_k,
                  const Matrix mu_k, const int i) const {
    Matrix val = dJi_dxi(x_k, u_k_i, i) +
                 lamda_k.row(i) * df_dx(x_k.row(i).transpose(), u_k_i, i);
    for (int j = 0; j < N; j++) {
      if (i == j) {
        continue;
      }
      if (h_k_plus_mask(i, j)) {
        val += mu_k(i, j) *
               (dh_dxi(x_k.row(i).transpose(), x_k.row(j).transpose()));
      } else {
        val +=
            -1.0 / rho *
            min(1.0 / h(x_k.row(i).transpose(), x_k.row(j).transpose()), 1e10) *
            dh_dxi(x_k.row(i).transpose(), x_k.row(j).transpose());
      }
    }
    return val;
  }


  Matrix r(const std::vector<Matrix> &x, const std::vector<Matrix> &u,
           const std::vector<Matrix> &lamda, const std::vector<Matrix> &mu,
           const std::vector<Matrix> &h_plus_mask) const {
    int h_plus_sum = 0;
    for (const auto &mask : h_plus_mask) {
      h_plus_sum += mask.count();
    }
    const int dim_r = N * (T * n + T * m + T * n) + h_plus_sum;

    Matrix r = Matrix::Zero(dim_r, 1);
    int index = 0;
    for (int i = 0; i < N; ++i) {
      Matrix dLL_dxi = dLLi_dxi(x, u, h_plus_mask, lamda, mu, i).transpose();
      Matrix dLL_dui = dLLi_dui(x, u, h_plus_mask, lamda, mu, i).transpose();
      r.block(index, 0, T * n, 1) = dLL_dxi;
      index += T * n;
      r.block(index, 0, T * m, 1) = dLL_dui;
      index += T * m;

      // Dynamics for f(x0,u0) = x1
      r.template block<n, 1>(index, 0) =
          f(x0.row(i).transpose(), u[0].row(i).transpose(), i) -
          x[0].row(i).transpose();

      for (int k = 1; k < T; ++k) {
        r.template block<n, 1>(index + k * n, 0) =
            f(x[k - 1].row(i).transpose(), u[k].row(i).transpose(), i) -
            x[k].row(i).transpose();
      }
      index += n * T;

      for (int k = 1; k < T + 1; ++k) {
        int h_idx = 0;
        for (int j = 0; j < N; ++j) {
          if (h_plus_mask[k - 1](i, j)) {
            r(index, 0) =
                h(x[k - 1].row(i).transpose(), x[k - 1].row(j).transpose());
            h_idx++;
          }
        }
        index += h_idx;
      }
    }
    return r;
  }

  Matrix dr_dy(const std::vector<Matrix> &x, const std::vector<Matrix> &u,
               const std::vector<Matrix> &lamda, const std::vector<Matrix> &mu,
               const std::vector<Matrix> &h_plus_mask) const {
    int h_plus_sum = 0;
    for (const auto &mask : h_plus_mask) {
      h_plus_sum += mask.count();
    }
    const int dim_x = N * T * n;
    const int dim_u = N * T * m;
    const int dim_lamda = T * N * n;
    const int dim_mu = T * N * N;
    const int dim_r = N * (T * n + T * m + T * n) + h_plus_sum;

    const int dim_y = dim_x + dim_u + dim_lamda + dim_mu;
    Matrix Dr(dim_r, dim_y);
    Dr.setZero();

    // cout << "drdx: " << endl;
    dr_dx(x, u, lamda, mu, h_plus_mask, Dr.block(0, 0, dim_r, dim_x));
    // cout << "drdu: " << endl;
    dr_du(x, u, lamda, mu, h_plus_mask, Dr.block(0, dim_x, dim_r, dim_u));
    // cout << "drdlamda: " << endl;
    dr_dlamda(x, u, lamda, mu, h_plus_mask,
              Dr.block(0, dim_x + dim_u, dim_r, dim_lamda));
    // cout << "drdmu: " << endl;
    dr_dmu(x, u, lamda, mu, h_plus_mask,
           Dr.block(0, dim_x + dim_u + dim_lamda, dim_r, dim_mu));
    return Dr;
  }

  SpMatrix dr_dy_sparse(const std::vector<Matrix> &x,
                        const std::vector<Matrix> &u,
                        const std::vector<Matrix> &lamda,
                        const std::vector<Matrix> &mu,
                        const std::vector<Matrix> &h_plus_mask) const {
    int h_plus_sum = 0;
    for (const auto &mask : h_plus_mask) {
      h_plus_sum += mask.count();
    }
    const int dim_x = N * T * n;
    const int dim_u = N * T * m;
    const int dim_lamda = T * N * n;
    const int dim_mu = T * N * N;
    const int dim_r = N * (T * n + T * m + T * n) + h_plus_sum;

    const int dim_y = dim_x + dim_u + dim_lamda + dim_mu;
    SpMatrix Dr(dim_r, dim_y);
    // Dr.reserve(int(dim_y*dim_y*0.01));
    Dr.reserve(Eigen::VectorXi::Constant(dim_y, int(dim_r * 0.01)));
    // Dr.setZero();

    // cout << "drdx: " << endl;
    dr_dx_fill_block(x, u, lamda, mu, h_plus_mask, Dr, 0, 0, dim_r, dim_x);
    // cout << "drdu: " << endl;
    dr_du_fill_block(x, u, lamda, mu, h_plus_mask, Dr, 0, dim_x, dim_r, dim_u);
    // cout << "drdlamda: " << endl;
    dr_dlamda_fill_block(x, u, lamda, mu, h_plus_mask, Dr, 0, dim_x + dim_u,
                         dim_r, dim_lamda);
    // cout << "drdmu: " << endl;
    dr_dmu_fill_block(x, u, lamda, mu, h_plus_mask, Dr, 0,
                      dim_x + dim_u + dim_lamda, dim_r, dim_mu);
    return Dr;
  }

  void rollout_in_place(const Matrix &x0, const std::vector<Matrix> &u,
                        std::vector<Matrix> &x) const {
    // NOTE that x[0] = x_1, x = [x_1..x_T]
    // x1 = f(x0,u0)
    for (int i = 0; i < N; i++) {
      x[0].row(i).transpose() =
          f(x0.row(i).transpose(), u[0].row(i).transpose(), i);
    }
    // x_k = f(x_k-1,u_k-1)
    for (int k = 2; k < T + 1; k++) {
      for (int i = 0; i < N; i++) {
        x[k - 1].row(i).transpose() =
            f(x[k - 2].row(i).transpose(), u[k - 1].row(i).transpose(), i);
      }
    }
  }

  // solve the game using step(), return: x,u,lambda, mu, has_converged
  std::tuple<std::vector<Matrix>, std::vector<Matrix>, std::vector<Matrix>,
             std::vector<Matrix>, bool>
  solve(const std::vector<Matrix> &in_u) {
    // initialize x (from x0,u) ,lamda, mu
    std::vector<Matrix> lamda(T, Matrix::Zero(N, n));
    std::vector<Matrix> mu(T, Matrix::Zero(N, N));
    std::vector<Matrix> x(T, Matrix(N, n));
    std::vector<Matrix> u(in_u);
    rollout_in_place(x0, u, x);

    // iteratively solve
    bool is_stop_condition_met = false;
    bool has_converged = false;
    bool has_runtime_err = false;

    for (int iter = 0; iter < max_iterations; iter++) {
      try {
        // std::cout << "iter " << iter << std::endl;
        auto retval = step(x, u, lamda, mu);
        x = retval[0];
        u = retval[1];
        lamda = retval[2];
        mu = retval[3];
        rollout_in_place(x0, u, x);
        post_step_update();
      } catch (const pybind11::stop_iteration &e) {
        if (verbose) {
          std::cout << "iter " << iter << " " << e.what() << std::endl;
        }
        is_stop_condition_met = true;
        if (!strcmp(e.what(), "stopping criteria met")) {
          has_converged = true;
          if (verbose) {
            std::cout << "algorithm converged after " << iter << " iterations "
                      << std::endl;
          }
        }
        break;
      } catch (const std::runtime_error &e) {
        std::cout << e.what() << std::endl;
        has_runtime_err = true;
        break; // NOTE no point in continuing
      }
    }
    if (verbose) {
      if (!is_stop_condition_met && !has_converged && !has_runtime_err) {
        std::cout << "no convergence after max iter has reached" << std::endl;
      }
    }
    // return x,u, lamda, mu, status, residual
    return std::tuple<std::vector<Matrix>, std::vector<Matrix>,
                      std::vector<Matrix>, std::vector<Matrix>, bool>{
        x, u, lamda, mu, has_converged};
  }

  std::vector<std::vector<Matrix>> step(const std::vector<Matrix> &x,
                                        const std::vector<Matrix> &u,
                                        const std::vector<Matrix> &lamda,
                                        const std::vector<Matrix> &mu) const {
    // int additional_memory_usage_kb = getCurrentMemoryUsageInKB() -
    // current_memory_usage_kb; std::cout << "step entry memory: " <<
    // additional_memory_usage_kb << "KB" << std::endl;

    // cout << "step()" << endl;
    py::gil_scoped_release release;
    const auto h_plus_mask = getHplusMask(x);
    int h_plus_sum = 0;
    for (const auto &mask : h_plus_mask) {
      h_plus_sum += mask.count();
    }
    // cout << "getHplusMask()" << endl;
    profiler.s();

    const int dim_x = T * N * n;
    const int dim_u = T * N * m;
    const int dim_lamda = T * N * n;
    const int dim_mu = T * N * N;
    const int dim_r = N * (T * n + T * m + T * n) + h_plus_sum;
    const int dim_y = dim_x + dim_u + dim_lamda + dim_mu;

    // cout << "r()" << endl;
    auto r0 = r(x, u, lamda, mu, h_plus_mask);
    /*
    profiler.s("build dense");
    const Matrix Dr_dense = dr_dy(x, u, lamda, mu, h_plus_mask);
    profiler.e("build dense");
    // remove zero rows/cols
    profiler.s("dense nonzeros");
    std::vector<int> nonzero_cols_idx_dense;
    nonzero_cols_idx_dense = nonzero_cols(Dr_dense);
    Matrix Dr_reduced_dense = Dr_dense(Eigen::all, nonzero_cols_idx_dense);
    SpMatrix Dr_reduced_sparse = Dr_reduced_dense.sparseView();
    profiler.e("dense nonzeros");
    */

    profiler.s("build sparse");
    SpMatrix Dr_sparse = dr_dy_sparse(x, u, lamda, mu, h_plus_mask);
    profiler.e("build sparse");
    // NOTE here the memory occupied by Dr is not released
    profiler.s("sparse nonzeros");
    std::vector<int> nonzero_cols_idx;
    SpMatrix Dr_reduced;
    std::tie(Dr_reduced, nonzero_cols_idx) =
        remove_empty_cols(Dr_sparse, dim_r);
    profiler.e("sparse nonzeros");

    profiler.s("solve");
    Eigen::LeastSquaresConjugateGradient<SpMatrix> solver;
    solver.setTolerance(1e-5);
    // solve r0 + Dr* dy = 0 least square
    solver.compute(Dr_reduced);
    // solver.compute(Dr.sparseView());
    // cout << "compute" << endl;

    if (solver.info() != Eigen::Success) {
      profiler.reject();
      throw std::runtime_error(" solver initialization failed");
      // return std::vector<std::vector<Matrix>>();
    }

    Matrix dy_reduced = solver.solve(-r0);
    if (solver.info() != Eigen::Success) {
      std::cout << "LSCG solver failed, trying SparseQR" << std::endl;
      try {
        dy_reduced = sp_SparseQR(Dr_reduced, -r0);
      } catch (const std::runtime_error &e) {
        profiler.reject();
        throw e;
      }
    }
    profiler.e("solve");

    // reconstruct dy from dy_reduced
    profiler.s("reconstruct dy");
    Matrix dy(dim_y, 1);
    dy.setZero();
    dy(nonzero_cols_idx, Eigen::all) = dy_reduced;
    profiler.e("reconstruct dy");

    // line search
    profiler.s("line search");
    Scalar step = 1.0; // step size
    Scalar r0_norm = r0.norm();
    Scalar rt_norm = r0_norm;
    Scalar apriori_h_res = getCollisionResidual(x);

    auto split_y = [&](const std::vector<Matrix> &x,
                       const std::vector<Matrix> &u,
                       const std::vector<Matrix> &lamda,
                       const std::vector<Matrix> &mu, const Matrix &dy,
                       Scalar my_step) {
      const auto x_size = x.size();
      std::vector<Matrix> xx(x_size);
      for (int i = 0; i < x_size; i++) {
        xx.at(i) = x.at(i) + my_step * dy.block(i * N * n, 0, N * n, 1)
                                           .reshaped<Eigen::RowMajor>(N, n);
      }
      const auto u_size = u.size();
      std::vector<Matrix> uu(x_size);
      for (int i = 0; i < u_size; i++) {
        uu.at(i) = u.at(i) + my_step * dy.block(dim_x + i * N * m, 0, N * m, 1)
                                           .reshaped<Eigen::RowMajor>(N, m);
      }
      const auto lamda_size = lamda.size();
      std::vector<Matrix> ll(lamda_size);
      for (int i = 0; i < lamda_size; i++) {
        ll.at(i) = lamda.at(i) +
                   my_step * dy.block(dim_x + dim_u + i * N * n, 0, N * n, 1)
                                 .reshaped<Eigen::RowMajor>(N, n);
      }
      const auto mu_size = mu.size();
      std::vector<Matrix> mm(mu_size);
      for (int i = 0; i < mu_size; i++) {
        mm.at(i) =
            mu.at(i) + my_step * dy.block(dim_x + dim_u + dim_lamda + i * N * N,
                                          0, N * N, 1)
                                     .reshaped<Eigen::RowMajor>(N, N);
      }
      return std::tuple<std::vector<Matrix>, std::vector<Matrix>,
                        std::vector<Matrix>, std::vector<Matrix>>{xx, uu, ll,
                                                                  mm};
    };

    auto r_t_norm = [&](Scalar my_step) {
      auto y_tuple = split_y(x, u, lamda, mu, dy, my_step);
      return r(std::get<0>(y_tuple), std::get<1>(y_tuple), std::get<2>(y_tuple),
               std::get<3>(y_tuple), h_plus_mask)
          .norm();
    };

    bool flag_no_step = true;
    for (int i = 0; i < backtracking_max_iter; i++) {
      auto y_tuple = split_y(x, u, lamda, mu, dy, step);
      rt_norm = r(std::get<0>(y_tuple), std::get<1>(y_tuple),
                  std::get<2>(y_tuple), std::get<3>(y_tuple), h_plus_mask)
                    .norm();
      if (rt_norm > (1 - bc_a * step) * r0_norm) {
        step *= bc_b;
      } else {
        Scalar search_h_res = getCollisionResidual(std::get<0>(y_tuple));
        if (search_h_res > apriori_h_res) {
          step *= bc_b;
        } else {
          flag_no_step = false;
          break;
        }
      }
    }
    profiler.e("line search");
    profiler.e();

    // additional_memory_usage_kb = getCurrentMemoryUsageInKB() -
    // current_memory_usage_kb; std::cout << "step exit memory: " <<
    // additional_memory_usage_kb << "KB" << std::endl;

    // stopping criteria
    auto y_tuple = split_y(x, u, lamda, mu, dy, step);
    if (abs(rt_norm) < tolerance and h_plus_sum == 0) {
      // stopping
      throw pybind11::stop_iteration("stopping criteria met");
      // return std::vector<std::vector<Matrix>>();
    } else if (flag_no_step) {
      throw pybind11::stop_iteration("iteration not making progress");
    } else {
      auto xx = std::get<0>(y_tuple);
      auto uu = std::get<1>(y_tuple);
      auto ll = std::get<2>(y_tuple);
      auto mm = std::get<3>(y_tuple);
      return std::vector<std::vector<Matrix>>{xx, uu, ll, mm};
    }
  }

  // --- helper function ---
  // find nonzero submatrix
  // return: skimmed matrix (dense), nonzero row indices, nonzero col indices
  std::vector<int> nonzero_cols(const Matrix &mtx) const {
    // cols
    Eigen::Matrix<bool, 1, Eigen::Dynamic, Eigen::RowMajor> nonzero_cols_mask =
        mtx.cast<bool>().colwise().any();
    const int nonzero_cols_size = nonzero_cols_mask.cast<int>().sum();
    std::vector<int> nonzero_cols_idx;
    nonzero_cols_idx.reserve(nonzero_cols_size);
    const int mtx_cols = mtx.cols();
    for (int i = 0; i < mtx_cols; i++) {
      if (nonzero_cols_mask(0, i)) {
        nonzero_cols_idx.push_back(i);
      }
    }

    return nonzero_cols_idx;
  }

  void summary() { profiler.summary(); }

  // DEBUG functions

  // solve Ax=B
  Matrix SparseQR(const Matrix &A, const Matrix &B) const {
    Eigen::SparseQR<SpMatrix, Eigen::COLAMDOrdering<int>> solver;
    solver.compute(A.sparseView());
    if (solver.info() != Eigen::Success) {
      throw std::runtime_error(" SparseQR solver initialization failed");
      return Matrix{};
    }

    Matrix x = solver.solve(B);
    if (solver.info() != Eigen::Success) {
      throw std::runtime_error(" SparseQR solver solve() failed");
      return Matrix{};
    }
    return x;
  }

  Matrix sp_SparseQR(const SpMatrix &sp_A, const Matrix &B) const {
    Eigen::SparseQR<SpMatrix, Eigen::COLAMDOrdering<int>> solver;
    solver.compute(sp_A);
    if (solver.info() != Eigen::Success) {
      throw std::runtime_error(" SparseQR solver initialization failed");
      return Matrix{};
    }

    Matrix x = solver.solve(B);
    if (solver.info() != Eigen::Success) {
      throw std::runtime_error(" SparseQR solver solve() failed");
      return Matrix{};
    }
    return x;
  }

  Matrix LeastSquaresConjugateGradient(const Matrix &A, const Matrix &B) const {
    Eigen::LeastSquaresConjugateGradient<SpMatrix> solver;
    solver.compute(A.sparseView());
    if (solver.info() != Eigen::Success) {
      throw std::runtime_error(" LSCG solver initialization failed");
      return Matrix{};
    }

    // solver.setMaxIterations();
    // solver.setTolerance
    Matrix x = solver.solve(B);
    if (solver.info() != Eigen::Success) {
      throw std::runtime_error(" LSCG solver solve() failed");
      return Matrix{};
    }
    return x;
  }

  void print_dim(const Matrix val) const {
    std::cout << "rows " << val.rows() << "cols " << val.cols() << endl;
  }
  Matrix test_bool_array(const Matrix val, const Matrix mask) const {
    Matrix output(val);
    for (int i = 0; i < val.rows(); i++) {
      for (int j = 0; j < val.cols(); j++) {
        if (!mask(i, j)) {
          output(i, j) = 0;
        }
      }
    }
    return output;
  }
  Matrix three_dim(const std::vector<Matrix> mtx_vec) const {
    return mtx_vec[1];
  }
