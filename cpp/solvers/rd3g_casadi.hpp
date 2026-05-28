// RD3G solver compatible with casadi codegen
#pragma once

#include <dlfcn.h>
#include <limits.h>
#include <string.h>
#include <unistd.h>

#include <filesystem>  // c++ 17
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#define SPDLOG_HEADER_ONLY
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/fmt/ranges.h>  // <--- Critical for printing containers
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

// #define EIGEN_RUNTIME_NO_MALLOC
// Eigen::internal::set_is_malloc_allowed(false);
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

// sparse solvers
#include <Eigen/OrderingMethods>
#include <Eigen/SparseQR>
// for LeastSquaresConjugateGradient
#include <Eigen/IterativeLinearSolvers>
#include <casadi/casadi.hpp>

namespace py = pybind11;
namespace cas = casadi;

#include "profiler.h"

using Scalar = double;
using Eigen::MatrixXd;
using std::cout;
using std::endl;
using std::max;
using std::min;
// SpMatrix was taken
using Vector = Eigen::VectorXd;
using SpMatrix = Eigen::SparseMatrix<double, Eigen::ColMajor, casadi_int>;
using MappedSparseMatrix = Eigen::Map<SpMatrix>;

namespace fs = std::filesystem;

// Work buffer for casadi functions
struct FuncWorkBuffer {
  std::vector<const double *> args;
  std::vector<double *> res;
  std::vector<casadi_int> iw;
  std::vector<double> w;
};

// Compressed Colume Storage matrix, for passing to Python
struct SparseMatrixResult {
  std::pair<int, int> shape;  // (rows, cols)
  std::vector<casadi_int> row;
  std::vector<casadi_int> colind;
  std::vector<double> data;
};

// Given a vector of CasADi functions, allocate sufficiently sized work buffers
// NOTE [args] and [res] are vector of pointers, this fun only allocate the
// pointers, not the actual content
FuncWorkBuffer get_max_buffer(std::vector<cas::Function> fun_vec) {
  size_t sz_arg = 0, sz_res = 0, sz_iw = 0, sz_w = 0;
  for (const auto &fun : fun_vec) {
    size_t sz_arg_, sz_res_, sz_iw_, sz_w_;
    fun.sz_work(sz_arg_, sz_res_, sz_iw_, sz_w_);
    sz_arg = sz_arg > sz_arg_ ? sz_arg : sz_arg_;
    sz_res = sz_res > sz_res_ ? sz_res : sz_res_;
    sz_iw = sz_iw > sz_iw_ ? sz_iw : sz_iw_;
    sz_w = sz_w > sz_w_ ? sz_w : sz_w_;
  }
  std::vector<const double *> args(sz_arg);
  std::vector<double *> res(sz_res);
  std::vector<casadi_int> iw(sz_iw);
  std::vector<double> w(sz_w);
  return {args, res, iw, w};
}

cas::Function safe_load_fun(std::string fun_name, fs::path lib_path) {
  cas::Function fun = cas::external(fun_name, lib_path.string());

  if (fun.is_null()) {
    throw std::runtime_error("Failed to load " + fun_name + " from " + lib_path.string());
  }
  return fun;
}

// Create MappedSparseMatrix sparsity pattern and data
inline MappedSparseMatrix get_mapped_spmatrix(casadi::Sparsity sp, double *data) {
  MappedSparseMatrix retval(sp.size1(), sp.size2(), sp.nnz(), const_cast<casadi_int *>(sp.colind()),
                            const_cast<casadi_int *>(sp.row()), data);
  return retval;
}

class Rd3gCasadi {
 protected:
  int n_, m_, N_, T_, n_h_;
  Scalar dt_, rho_, rho_b_, bc_a_, bc_b_, reg0_, reg_;
  Scalar tolerance_;
  int line_search_max_iter_;
  int max_iterations_;
  int max_in_reg_iter_;
  Scalar max_in_reg_val_;
  // dim: N*n
  MatrixXd x0_;
  mutable Profiler<false> profiler_;
  int verbose_;  // 0:error, 1:warning, 2:info, 3:debug
  const bool inertia_correction_;
  const bool keep_only_active_constraints_in_ki_;
  const bool reduce_kkt_system_;
  const bool rollout_each_step_;
  int line_search_no_progress_counter_;

  std::shared_ptr<spdlog::logger> logger_;

  // CasADi work buffer
  FuncWorkBuffer wb_;

  cas::Function r_;
  cas::Function dr_dy_;
  cas::Function rollout_;
  cas::Function h_;
  cas::Function collision_h_;
  cas::Function get_full_context_;
  std::vector<cas::Function> Ki_vec_;
  // std::vector<cas::Function> ri_vec_;

  bool debug_got_vals_;  // We only need the first vals
  SpMatrix debug_full_KKT_;
  SpMatrix debug_reduced_KKT_;
  SpMatrix debug_full_r0_;
  std::vector<double> debug_context;
  MatrixXd debug_full_dy;
  MatrixXd debug_x;
  MatrixXd debug_u;

 public:
  // NOTE n,m may need to be template variables for performance
  Rd3gCasadi(const int N, const int T, const int n_h, const Scalar dt, const Scalar rho,
             const Scalar rho_b, const Scalar bc_a, const Scalar bc_b, const Scalar reg,
             const bool inertia_correction, const bool keep_only_active_constraints_in_ki,
             const bool reduce_kkt_system,
             const bool rollout_each_step, const Scalar tolerance, const int backtracking_max_iter,
             const int max_iter, const int max_in_reg_iter, const Scalar max_in_reg_val,
             const int verbose, const std::string base_dir, const std::string casadi_module_name)
      : N_{N},
        T_{T},
        n_h_{n_h},
        dt_{dt},
        rho_{rho},
        rho_b_{rho_b},
        bc_a_{bc_a},
        bc_b_{bc_b},
        reg0_{reg},
        reg_{reg},
        inertia_correction_{inertia_correction},
        keep_only_active_constraints_in_ki_{keep_only_active_constraints_in_ki},
        reduce_kkt_system_{reduce_kkt_system},
        rollout_each_step_{rollout_each_step},
        tolerance_{tolerance},
        line_search_max_iter_{backtracking_max_iter},
        x0_{},
        profiler_{},
        max_iterations_{max_iter},
        max_in_reg_iter_{max_in_reg_iter},
        max_in_reg_val_{max_in_reg_val},
        verbose_{verbose},
        line_search_no_progress_counter_{0},
        debug_got_vals_{false} {
    const std::string logger_name{"rd3g_casadi_cpp"};
    logger_ = spdlog::get(logger_name);
    if (!logger_) {
      logger_ = spdlog::stdout_color_mt(logger_name);
    }

    // 0:error, 1:warning, 2:info, 3:debug
    switch (verbose_) {
      case 0:
        spdlog::set_level(spdlog::level::err);
        break;
      case 1:
        spdlog::set_level(spdlog::level::warn);
        break;
      case 2:
        spdlog::set_level(spdlog::level::info);
        break;
      case 3:
        spdlog::set_level(spdlog::level::debug);
        break;
    }
    // Set custom format: [Time] [Logger] [Level] Message
    spdlog::set_pattern("[%H:%M:%S.%e] [%n] [%^%l%$] %v");

    // Load CasADi dll for given game following naming convention,
    // game (module name), N, T -> this determines a unique dll name.
    std::stringstream ss;
    ss << "lib" << casadi_module_name << "_rd3g_N" << N << "_T" << T << ".so";
    fs::path lib_path = fs::path(base_dir) / "build" / "lib" / ss.str();

    r_ = safe_load_fun("r", lib_path);
    dr_dy_ = safe_load_fun("dr_dy", lib_path);
    rollout_ = safe_load_fun("rollout", lib_path);
    h_ = safe_load_fun("h", lib_path);
    collision_h_ = safe_load_fun("collision_h", lib_path);
    get_full_context_ = safe_load_fun("get_full_context", lib_path);

    cas::Function get_n = safe_load_fun("get_n", lib_path);
    cas::Function get_m = safe_load_fun("get_m", lib_path);

    Ki_vec_.reserve(N_);
    // ri_vec_.reserve(N_);
    for (int i = 0; i < N_; i++) {
      Ki_vec_.push_back(safe_load_fun("K_" + std::to_string(i), lib_path));
      // ri_vec_.push_back(safe_load_fun("r_"+std::to_string(i), lib_path));
    }

    wb_ = get_max_buffer({r_, dr_dy_, rollout_, h_, collision_h_, get_full_context_, get_n, get_m});

    std::vector<double> n_buffer(get_n.sparsity_out(0).nnz());
    wb_.res[0] = n_buffer.data();
    get_n(wb_.args.data(), wb_.res.data(), wb_.iw.data(), wb_.w.data(), 0);
    wb_.res[0] = nullptr;
    assert(get_n.n_out() == 1);
    n_ = n_buffer[0];

    std::vector<double> m_buffer(get_m.sparsity_out(0).nnz());
    wb_.res[0] = m_buffer.data();
    get_m(wb_.args.data(), wb_.res.data(), wb_.iw.data(), wb_.w.data(), 0);
    wb_.res[0] = nullptr;
    assert(get_m.n_out() == 1);
    m_ = m_buffer[0];

    logger_->debug("RD3G CasADi initialized, n={}, m={}", n_, m_);
  }

  void set_x0(const MatrixXd &val) { x0_ = MatrixXd(val); }
  void post_step_update() { rho_ *= rho_b_; }

  // Evaluate dr_dy function from casadi using python arguments.
  // Demonstrating data representation conversion and call procedure
  SparseMatrixResult casadi_dr_dy(py::array_t<double> x, py::array_t<double> u,
                                  py::array_t<double> lamda, py::array_t<double> mu,
                                  py::array_t<double> context, py::array_t<double> int_param,
                                  py::array_t<double> double_param) {
    // Set input args
    auto x_val = x.request();  // py::buffer_info
    auto u_val = u.request();
    auto lamda_val = lamda.request();
    auto mu_val = mu.request();
    auto context_val = context.request();
    auto int_param_val = int_param.request();
    auto double_param_val = double_param.request();

    wb_.args[0] = static_cast<double *>(x_val.ptr);
    wb_.args[1] = static_cast<double *>(u_val.ptr);
    wb_.args[2] = static_cast<double *>(lamda_val.ptr);
    wb_.args[3] = static_cast<double *>(mu_val.ptr);
    wb_.args[4] = static_cast<double *>(context_val.ptr);
    wb_.args[5] = static_cast<double *>(int_param_val.ptr);
    wb_.args[6] = static_cast<double *>(double_param_val.ptr);
    assert(dr_dy_.n_in() == 7);

    const casadi::Sparsity &res_sp = dr_dy_.sparsity_out(0);  // 0th output sparsity
    // Allocate output buffer
    std::vector<double> res_buffer(res_sp.nnz());
    wb_.res[0] = res_buffer.data();
    assert(dr_dy_.n_out() == 1);

    // Call work function
    dr_dy_(wb_.args.data(), wb_.res.data(), wb_.iw.data(), wb_.w.data(), 0);

    SparseMatrixResult res;
    res.shape = res_sp.size();
    res.data = res_buffer;
    res.row.assign(res_sp.row(), res_sp.row() + res_sp.nnz());
    res.colind.assign(res_sp.colind(), res_sp.colind() + res_sp.size2() + 1);
    return res;
  }

  // Given [h_val],
  // remove rows corresponding to h<0 (inactive constraints) full_r0
  // Remove rows for h<0, and cols for corresponding mu (multiplier) from
  // full_KKT. Returns:
  //      reduced_KKT:
  //      reduced_r0:
  //      active_h_indices:
  std::tuple<SpMatrix, MatrixXd, std::vector<int>> reduce_KKT_system(const SpMatrix &full_KKT,
                                                                     const SpMatrix &full_r0,
                                                                     const MatrixXd &h_val) {
    // Starting index of first h() in residual
    // Skipping through dLLi_dx, dLLi_du, dynamics constraint
    // Also starting index of mu, multiplier for h(), in y
    const int h_in_r_offset = n_ * N_ * T_ + m_ * N_ * T_ + n_ * N_ * T_;
    assert(h_val.rows() == n_h_ * N_);
    assert(h_val.cols() == 1);

    // Maps rows in full_r0 to reduced_r0, -1 means delete
    std::vector<int> old_to_new_idx(full_r0.rows(), -1);
    std::vector<int> active_h_indices;
    active_h_indices.reserve(n_h_ * N_);
    int reduced_r_dim = 0;

    // Always keep dL/dx, dL/du, dynamics constraints
    for (int i = 0; i < h_in_r_offset; i++) {
      old_to_new_idx[i] = reduced_r_dim++;
    }

    assert(h_val.rows() == full_r0.rows() - h_in_r_offset);
    for (int i = h_in_r_offset; i < full_r0.rows(); i++) {
      if (h_val(i - h_in_r_offset, 0) >= 0.0) {
        old_to_new_idx[i] = reduced_r_dim++;
        active_h_indices.push_back(i - h_in_r_offset);
      }
    }

    // Construct reduced_r0
    // TODO should this be sparse?
    MatrixXd reduced_r0(reduced_r_dim, 1);

    for (int i = 0; i < full_r0.rows(); i++) {
      if (old_to_new_idx[i] != -1) {
        reduced_r0(old_to_new_idx[i], 0) = full_r0.coeff(i, 0);
      }
    }

    // Construct reduced_KKT
    std::vector<Eigen::Triplet<Scalar>> triplets;
    triplets.reserve(full_KKT.rows());
    for (int j = 0; j < full_KKT.outerSize(); j++) {
      for (SpMatrix::InnerIterator it(full_KKT, j); it; ++it) {
        // Removing h and its corresponding mu (multiplier) together means
        // removing same row and col
        int new_row = old_to_new_idx[it.row()];
        int new_col = old_to_new_idx[it.col()];
        if (new_row != -1 && new_col != -1) {
          triplets.emplace_back(new_row, new_col, it.value());
        }
      }
    }

    // KKT is a square matrix, same dim as r
    SpMatrix reduced_KKT(reduced_r_dim, reduced_r_dim);
    reduced_KKT.setFromTriplets(triplets.begin(), triplets.end());

    return {reduced_KKT, reduced_r0, active_h_indices};
  }

  std::tuple<SpMatrix, std::vector<int>> reduce_Ki_system(const SpMatrix &Ki,
                                                          const MatrixXd &h_i_val) {
    // K_i uses the same ordering for residual rows and primal/dual variables,
    // so removing an inactive h row and its corresponding mu column means
    // keeping the same active index set on both axes.
    const int h_in_r_offset = (n_ + m_) * T_ + n_ * T_;
    assert(h_i_val.rows() * h_i_val.cols() == n_h_);

    std::vector<int> old_to_new_idx(Ki.rows(), -1);
    std::vector<int> active_h_indices;
    active_h_indices.reserve(n_h_);
    int reduced_dim = 0;

    for (int i = 0; i < h_in_r_offset; ++i) {
      old_to_new_idx[i] = reduced_dim++;
    }

    for (int i = 0; i < n_h_; ++i) {
      if (h_i_val(i, 0) >= 0.0) {
        old_to_new_idx[h_in_r_offset + i] = reduced_dim++;
        active_h_indices.push_back(i);
      }
    }

    std::vector<Eigen::Triplet<Scalar>> triplets;
    triplets.reserve(Ki.nonZeros());
    for (int j = 0; j < Ki.outerSize(); ++j) {
      for (SpMatrix::InnerIterator it(Ki, j); it; ++it) {
        const int new_row = old_to_new_idx[it.row()];
        const int new_col = old_to_new_idx[it.col()];
        if (new_row != -1 && new_col != -1) {
          triplets.emplace_back(new_row, new_col, it.value());
        }
      }
    }

    SpMatrix reduced_Ki(reduced_dim, reduced_dim);
    reduced_Ki.setFromTriplets(triplets.begin(), triplets.end());
    return {reduced_Ki, active_h_indices};
  }

  // Solve linear system of the form Ax = b
  // Returns:
  //   x: solution
  //   res: residual, norm(Ax-b)
  std::tuple<SpMatrix, Scalar> solve_linear_system(const SpMatrix &A, const MatrixXd &b,
                                                   std::string method) {
    if (method == "lscg") {
      Eigen::LeastSquaresConjugateGradient<SpMatrix> solver;
      solver.setTolerance(1e-5);
      // solve r0 + Dr* dy = 0 least square
      solver.compute(A);
      // solver.compute(Dr.sparseView());

      if (solver.info() != Eigen::Success) {
        throw std::runtime_error(" solver initialization failed");
        // return std::vector<std::vector<Matrix>>();
      }

      MatrixXd x = solver.solve(b);
      if (solver.info() != Eigen::Success) {
        // TODO use logging
        logger_->info("LSCG solver failed ");
      }
      // NOTE this is relative error |Ax-b|/|Ax|, make sure it's consistent
      // elsewhere
      // TODO should we use dense matrix for x?
      return {x.sparseView(), static_cast<Scalar>(solver.error())};
    } else if (method == "ldl") {
      // SimplicialLDLT is a direct sparse solver for P*A*P' = L*D*L'
      // Note: Unlike qdldl Eigen defaults to reading the LOWER triangular part.
      // If somehow using upper,
      // change this to: Eigen::SimplicialLDLT<SpMatrix, Eigen::Upper> solver;
      Eigen::SimplicialLDLT<SpMatrix> solver;

      // A must be symmetric, symbolic LDL decomposition
      // Consider reusing this, if A's structure doesn't change
      // logger_->debug("Symbolic LDL decomp...");
      assert((A - SpMatrix(A.transpose())).norm() < 1e-6);
      solver.compute(A);

      if (solver.info() != Eigen::Success) {
        logger_->error("LDL decomposition failed");
        throw std::runtime_error("LDL decomposition failed");
      }

      // Solve System (A * x = b)
      // This automatically handles the permutations (P) internally
      // logger_->debug("Solving...");
      Eigen::VectorXd x = solver.solve(b);

      // Get diagonal of matrix D from L*D*L'
      // TODO does Eigen give 2 by 2 blocks?
      Eigen::VectorXd D = solver.vectorD();

      int pos = 0;
      int neg = 0;
      int zero = 0;

      // Iterate through D to count signs
      // logger_->debug("Checking inertia...");
      /*
      const double epsilon = 1e-8;
      for (int i = 0; i < D.size(); ++i) {
          if (D[i] > epsilon) pos++;
          else if (D[i] < -epsilon) neg++;
          else zero++;
      }
      auto inertia = std::make_tuple(pos, neg, zero);
      */

      const Scalar residual = (A * x - b).norm();

      return {x.sparseView(), residual};
    } else if (method == "lsqr") {
      Eigen::SparseQR<SpMatrix, Eigen::COLAMDOrdering<SpMatrix::StorageIndex>> solver;

      // Compute the QR factorization
      solver.compute(A);

      if (solver.info() != Eigen::Success) {
        logger_->error("QR decomposition failed");
        throw std::runtime_error("QR decomposition failed");
      }

      // Solves the least squares problem min ||Ax - b||
      MatrixXd x = solver.solve(b);

      if (solver.info() != Eigen::Success) {
        logger_->error("QR solver failed during solve phase");
      }

      const Scalar residual = (A * x - b).norm();
      return {x.sparseView(), residual};
    }
    throw std::runtime_error("Unknown method type: " + method);
  }

  // Solve dynamic game
  // Returns:
  // x
  // u
  // lamda
  // mu
  // residual
  // has_converged
  // is_optimal
  // iterations
  // msg
  std::tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd, Scalar, bool, bool, int, std::string> solve(
      py::array_t<double> x0, py::array_t<double> u_guess, py::array_t<double> int_param,
      py::array_t<double> double_param) {
    // Set input args
    auto x0_val = x0.request();  // py::buffer_info
    auto u_guess_val = u_guess.request();
    auto int_param_val = int_param.request();
    auto double_param_val = double_param.request();

    // Call rollout(x0, u_guess, int_param, double_param) -> x
    wb_.args[0] = static_cast<double *>(x0_val.ptr);
    wb_.args[1] = static_cast<double *>(u_guess_val.ptr);
    wb_.args[2] = static_cast<double *>(int_param_val.ptr);
    wb_.args[3] = static_cast<double *>(double_param_val.ptr);
    assert(rollout_.n_in() == 4);

    casadi::Sparsity x_sp = rollout_.sparsity_out(0);
    std::vector<double> x_buffer(x_sp.nnz());
    wb_.res[0] = x_buffer.data();
    assert(rollout_.n_out() == 1);

    assert(x_sp.is_dense());
    rollout_(wb_.args.data(), wb_.res.data(), wb_.iw.data(), wb_.w.data(), 0);
    wb_.res[0] = nullptr;  // Avoid accidentally overwriting the buffer
    Eigen::Map<MatrixXd> x(x_buffer.data(), x_sp.size1(), x_sp.size2());

    // u_guess, dense
    auto u = u_guess.cast<MatrixXd>();

    MatrixXd lamda = MatrixXd::Zero(n_ * N_, T_);
    MatrixXd mu = MatrixXd::Zero(n_h_ * N_, 1);
    bool has_converged = false;
    bool is_optimal = false;
    bool stop = false;
    Scalar residual = 1e10;

    int iter;
    for (iter = 0; iter < max_iterations_; iter++) {
      if (rollout_each_step_) {
        // Call rollout(x0, u_guess, int_param, double_param) -> x
        wb_.args[0] = static_cast<double *>(x0_val.ptr);
        wb_.args[1] = static_cast<double *>(u.data());
        wb_.args[2] = static_cast<double *>(int_param_val.ptr);
        wb_.args[3] = static_cast<double *>(double_param_val.ptr);
        assert(rollout_.n_in() == 4);
        wb_.res[0] = x_buffer.data();
        assert(rollout_.n_out() == 1);
        rollout_(wb_.args.data(), wb_.res.data(), wb_.iw.data(), wb_.w.data(), 0);
        wb_.res[0] = nullptr;  // Avoid accidentally overwriting the buffer
      }
      std::tie(stop, has_converged, is_optimal, residual) =
          step(x, u, lamda, mu, int_param, double_param);
      logger_->info("converged={}, optimal={}, residual={:.5f}", has_converged, is_optimal,
                    residual);
      if (has_converged || stop) {
        break;
      }
    }
    // Return: x,  u,  lamda,  mu,  residual, has_converged,  is_optimal,
    // iterations,  msg
    std::string msg{"no info"};
    return {x, u, lamda, mu, residual, has_converged, is_optimal, iter, msg};
  }

  // Take one Newton step, modify x,u,lamda,mu IN PLACE
  // x_ref: n*N,T
  // u_ref: m*N,T
  // lamda: n*N,T
  // mu: n_h*N, 1
  // Returns (stop, has_converged, is_optimal, residual)
  std::tuple<bool, bool, bool, Scalar> step(Eigen::Ref<MatrixXd> x, Eigen::Ref<MatrixXd> u,
                                            Eigen::Ref<MatrixXd> lamda, Eigen::Ref<MatrixXd> mu,
                                            py::array_t<double> &int_param,
                                            py::array_t<double> &double_param) {
    // Solve r0 + H @ dy = 0
    // i.e. full_r0 + full_KKT @ <dx, du, dlambda, dmu> = 0
    // identify inactive constraints (mu)
    // skim down H, dy, remove inactive constraints, dual variables
    // Solve for dy

    // Call h_val(), r(), dr_dy() to get full_r0 and full_KKT
    auto int_param_val = int_param.request();
    auto double_param_val = double_param.request();
    // Get game context
    // Call get_full_context_(x)
    wb_.args[0] = static_cast<double *>(x.data());
    assert(get_full_context_.n_in() == 1);
    assert(get_full_context_.sparsity_in(0).is_dense());
    casadi::Sparsity full_context_sp = get_full_context_.sparsity_out(0);
    assert(full_context_sp.is_dense());
    std::vector<double> full_context_buffer(full_context_sp.nnz());
    wb_.res[0] = full_context_buffer.data();
    assert(get_full_context_.n_out() == 1);
    if (!debug_got_vals_) {
      debug_context = full_context_buffer;
      debug_x = x;
      debug_u = u;
    }

    get_full_context_(wb_.args.data(), wb_.res.data(), wb_.iw.data(), wb_.w.data(), 0);
    wb_.res[0] = nullptr;

    // logger_->debug("Getting r0 and KKT matrix...");
    // Get residual
    // full_r0 = r(x, u, lamda, mu, context, int_param, double_param)
    wb_.args[0] = static_cast<double *>(x.data());
    wb_.args[1] = static_cast<double *>(u.data());
    wb_.args[2] = static_cast<double *>(lamda.data());
    wb_.args[3] = static_cast<double *>(mu.data());
    wb_.args[4] = static_cast<double *>(full_context_buffer.data());
    wb_.args[5] = static_cast<double *>(int_param_val.ptr);
    wb_.args[6] = static_cast<double *>(double_param_val.ptr);
    assert(r_.n_in() == 7);
    assert(r_.sparsity_in(0).is_dense());
    assert(r_.sparsity_in(1).is_dense());
    assert(r_.sparsity_in(2).is_dense());
    assert(r_.sparsity_in(3).is_dense());
    assert(r_.sparsity_in(4).is_dense());
    assert(r_.sparsity_in(5).is_dense());
    assert(r_.sparsity_in(6).is_dense());

    // DEBUG, ensure no nan in input
    if (x.hasNaN()) logger_->warn("x has nan");
    if (u.hasNaN()) logger_->warn("u has nan");
    if (lamda.hasNaN()) logger_->warn("lamda has nan");
    if (mu.hasNaN()) logger_->warn("mu has nan");
    if (std::any_of(full_context_buffer.begin(), full_context_buffer.end(),
                    [](double x) { return std::isnan(x); })) {
      logger_->warn("full_context has nan");
    }

    casadi::Sparsity full_r0_sp = r_.sparsity_out(0);
    assert(full_r0_sp.is_dense());
    std::vector<double> full_r0_buffer(full_r0_sp.nnz());
    wb_.res[0] = full_r0_buffer.data();

    casadi::Sparsity h_val_sp = r_.sparsity_out(1);
    std::vector<double> h_val_buffer(h_val_sp.nnz());
    wb_.res[1] = h_val_buffer.data();
    assert(r_.n_out() == 2);

    r_(wb_.args.data(), wb_.res.data(), wb_.iw.data(), wb_.w.data(), 0);
    wb_.res[0] = nullptr;
    wb_.res[1] = nullptr;

    auto full_r0 = get_mapped_spmatrix(full_r0_sp, full_r0_buffer.data());
    Scalar r0_norm = full_r0.norm();
    assert(h_val_sp.is_dense());
    // auto h_val = get_mapped_spmatrix(h_val_sp, h_val_buffer.data());
    Eigen::Map<MatrixXd> h_val(h_val_buffer.data(), h_val_sp.size1(), h_val_sp.size2());

    // Get Game Jacobian
    // Call full_KKT = dr_dy(x, u, lamda, mu, context, int_param, double_param)
    // Same input as r() call
    assert(dr_dy_.n_in() == 7);
    assert(dr_dy_.sparsity_in(0).is_dense());
    assert(dr_dy_.sparsity_in(1).is_dense());
    assert(dr_dy_.sparsity_in(2).is_dense());
    assert(dr_dy_.sparsity_in(3).is_dense());
    assert(dr_dy_.sparsity_in(4).is_dense());
    assert(dr_dy_.sparsity_in(5).is_dense());
    assert(dr_dy_.sparsity_in(6).is_dense());

    casadi::Sparsity full_KKT_sp = dr_dy_.sparsity_out(0);
    std::vector<double> full_KKT_buffer(full_KKT_sp.nnz());
    wb_.res[0] = full_KKT_buffer.data();
    assert(dr_dy_.n_out() == 1);

    dr_dy_(wb_.args.data(), wb_.res.data(), wb_.iw.data(), wb_.w.data(), 0);
    wb_.res[0] = nullptr;

    // Copy here because mapped SpMatrix is read-only, but we need to modify it
    // later
    SpMatrix full_KKT = get_mapped_spmatrix(full_KKT_sp, full_KKT_buffer.data());
    check_spmatrix_has_nan(full_KKT, "full_KKT");

    // Check inertia for each agent KKT matrix Ki
    bool is_optimal = true;
    std::vector<int> saddle_agent_vec;
    saddle_agent_vec.reserve(N_);
    std::vector<Scalar> reg_vec(N_, 0);

    const int primal_n = (n_ + m_) * T_;

    for (int i = 0; i < N_; i++) {
      // Call K_i
      casadi::Sparsity Ki_sp = Ki_vec_[i].sparsity_out(0);
      std::vector<double> Ki_buffer(Ki_sp.nnz());
      wb_.res[0] = Ki_buffer.data();
      assert(Ki_vec_[i].n_out() == 1);
      auto Ki = get_mapped_spmatrix(Ki_sp, Ki_buffer.data());

      // The arguments are the same as r, dr_dy. No need to reset wb_.args
      Ki_vec_[i](wb_.args.data(), wb_.res.data(), wb_.iw.data(), wb_.w.data(), 0);
      wb_.res[0] = nullptr;
      SpMatrix Ki_for_inertia = Ki;
      int active_h_count = n_h_;
      if (keep_only_active_constraints_in_ki_) {
        MatrixXd h_i_val = h_val.col(i);
        std::vector<int> active_h_indices;
        std::tie(Ki_for_inertia, active_h_indices) = reduce_Ki_system(Ki_for_inertia, h_i_val);
        active_h_count = active_h_indices.size();
      }

      // Apply Levenberg-Marquardt Regularization
      // Without this AMD permutation will fail
      int rows = Ki_for_inertia.rows();
      int cols = Ki_for_inertia.cols();
      std::vector<Eigen::Triplet<double>> triplets;
      triplets.reserve(rows);
      for (int k = 0; k < rows; ++k) {
        // Primal (+reg), Dual/Constraints (-reg)
        double val = (k < primal_n) ? reg_ : -reg_;
        triplets.emplace_back(k, k, val);
      }
      SpMatrix reg_matrix(rows, cols);
      reg_matrix.setFromTriplets(triplets.begin(), triplets.end());
      SpMatrix Ki_reg = Ki_for_inertia + reg_matrix;

      std::vector<Eigen::Triplet<double>> I_H_triplets;
      I_H_triplets.reserve(primal_n);
      for (int k = 0; k < primal_n; ++k) {
        I_H_triplets.emplace_back(k, k, 1.0);
      }
      SpMatrix I_H(rows, cols);
      I_H.setFromTriplets(I_H_triplets.begin(), I_H_triplets.end());
      // LDL decomposition
      Eigen::SimplicialLDLT<SpMatrix> solver;
      solver.analyzePattern(Ki_reg);  // .compute() without .factorize()
      auto inertia = get_inertia(Ki_reg, solver);
      auto expected_inertia = std::make_tuple(primal_n, n_ * T_ + active_h_count, 0);
      if (inertia != expected_inertia) {
        is_optimal = false;
        saddle_agent_vec.push_back(i);
        if (inertia_correction_) {
          // Old strategy: Find missing positive eigenvalue count, use pivot
          // This is often not enough to correct inertia
          // int bad_in_count = std::get<0>(expected_inertia) -
          // std::get<0>(inertia);
          // // NOTE neg_pivot_vec is uninitialized
          // std::sort(neg_pivot_vec.begin(), neg_pivot_vec.end());
          // reg_vec[i] = - neg_pivot_vec[bad_in_count-1];

          Scalar reg_upper = 2.0;
          Scalar reg_lower = 1.0;
          auto reg_upper_in = get_inertia(Ki_reg + reg_upper * I_H, solver);
          auto reg_lower_in = get_inertia(Ki_reg + reg_lower * I_H, solver);
          for (int reg_iter = 0; reg_iter < max_in_reg_iter_; reg_iter++) {
            if (reg_upper_in != expected_inertia) {
              reg_lower = reg_upper;
              reg_lower_in = reg_upper_in;
              reg_upper *= 2;
              reg_upper_in = get_inertia(Ki_reg + reg_upper * I_H, solver);
            } else if (reg_lower_in == expected_inertia) {
              reg_upper = reg_lower;
              reg_upper_in = reg_lower_in;
              reg_lower /= 2;
              reg_lower_in = get_inertia(Ki_reg + reg_lower * I_H, solver);
            } else {
              Scalar reg_middle = (reg_upper + reg_lower) / 2;
              auto reg_middle_in = get_inertia(Ki_reg + reg_middle * I_H, solver);
              if (reg_middle_in == expected_inertia) {
                reg_upper = reg_middle;
                reg_upper_in = reg_middle_in;
              } else {
                reg_lower = reg_middle;
                reg_lower_in = reg_middle_in;
              }
            }
            if ((reg_upper - reg_lower) / reg_upper < 0.1) {
              break;
            }
            if (reg_upper > max_in_reg_val_) {
              reg_upper = max_in_reg_val_;
              break;
            }
          }
          if (reg_upper_in != expected_inertia) {
            logger_->warn("Failed to correct K_{} inertia", i);
          }
          reg_vec[i] = reg_upper;
        }
      }
    }
    logger_->info("Saddle agents: {}", saddle_agent_vec);

    // TODO still need to add LM regularization
    if (inertia_correction_) {
      full_KKT += make_full_KKT_reg(reg_vec);
    }

    MatrixXd full_dy;
    Scalar residual;
    if (reduce_kkt_system_) {
      // Apply active set method, skim down full_r0 and full_KKT
      // logger_->debug("Reduce KKT system...");
      SpMatrix reduced_KKT;
      MatrixXd reduced_r0;
      // TODO maybe there's a more useful index? like new_to_old
      std::vector<int> active_h_indices;
      // Can't use r0 directly, can't differentiate between an inactive h<0 vs a
      // tight h=0 since both are 0 in r
      std::tie(reduced_KKT, reduced_r0, active_h_indices) =
          reduce_KKT_system(full_KKT, full_r0, h_val.reshaped());
      assert(reduced_KKT.rows() == reduced_KKT.cols());
      // Solve reduced system reduced_r0 + reduced_KKT @ reduced_dy = 0

      // TODO maybe use dense matrix for solution
      Vector reduced_dy;

      // Apply Levenberg-Marquardt Regularization
      // logger_->debug("Apply Regularization...");
      std::vector<Eigen::Triplet<double>> reg_triplets;
      reg_triplets.reserve(reduced_KKT.rows());
      // Primal Variables: Add +reg to diagonal
      const int primal_var_count = (n_ + m_) * N_ * T_;
      for (int i = 0; i < primal_var_count; ++i) {
        reg_triplets.emplace_back(i, i, reg_);
      }
      // Dual Variables (Constraints): Add -reg to diagonal
      for (int i = primal_var_count; i < reduced_KKT.rows(); ++i) {
        reg_triplets.emplace_back(i, i, -reg_);
      }
      Eigen::SparseMatrix<double> reg_matrix(reduced_KKT.rows(), reduced_KKT.cols());
      reg_matrix.setFromTriplets(reg_triplets.begin(), reg_triplets.end());
      reduced_KKT += reg_matrix;

      // logger_->debug("Solve linear system ...");
      check_spmatrix_has_nan(reduced_KKT, "reduced_KKT");
      if (reduced_r0.hasNaN()) {
        logger_->warn("reduced_dy has nan");
      }
      std::tie(reduced_dy, residual) = solve_linear_system(reduced_KKT, -reduced_r0, "lscg");
      if (reduced_dy.hasNaN()) {
        logger_->warn("reduced_dy has nan");
      }
      const int mu_in_y_offset = n_ * N_ * T_ + m_ * N_ * T_ + n_ * N_ * T_;
      assert(reduced_dy.rows() == mu_in_y_offset + active_h_indices.size());

      // Reconstruct full_dy from reduced_dy
      full_dy = MatrixXd::Zero(full_KKT.cols(), 1);
      full_dy.block(0, 0, mu_in_y_offset, 1) = reduced_dy.block(0, 0, mu_in_y_offset, 1);
      for (int i = 0; i < active_h_indices.size(); i++) {
        full_dy(mu_in_y_offset + active_h_indices[i], 0) = reduced_dy(mu_in_y_offset + i, 0);
      }
      logger_->debug("reduced_dy sq_norm {:.5f}", pow(reduced_dy.norm(), 2));
      debug_reduced_KKT_ = reduced_KKT;
    } else {
      std::tie(full_dy, residual) = solve_linear_system(full_KKT, -full_r0, "lscg");
    }
    // FIXME why are they different???
    logger_->debug("full_dy sq_norm {:.5f}", pow(full_dy.norm(), 2));

    // Line Search, regularization bloating
    Scalar step = 1.0;
    MatrixXd new_x(n_ * N_, T_);
    MatrixXd new_u(m_ * N_, T_);
    MatrixXd new_lamda(n_ * N_, T_);
    MatrixXd new_mu(n_h_ * N_, 1);
    int ls_iter;
    Scalar new_r_norm = -1;
    for (ls_iter = 0; ls_iter < line_search_max_iter_; ls_iter++) {
      // check r(y+step_size*dy).norm()
      // Construct new x,u,lamda,mu
      int offset = 0;
      const int x_dim = n_ * N_ * T_;
      new_x = x + step * full_dy.block(0, 0, x_dim, 1).reshaped(n_ * N_, T_);
      offset += x_dim;
      const int y_dim = m_ * N_ * T_;
      new_u = u + step * full_dy.block(offset, 0, y_dim, 1).reshaped(m_ * N_, T_);
      offset += y_dim;
      const int lamda_dim = n_ * N_ * T_;
      new_lamda = lamda + step * full_dy.block(offset, 0, lamda_dim, 1).reshaped(n_ * N_, T_);
      offset += lamda_dim;
      const int mu_dim = n_h_ * N_;  // mu is column vector
      new_mu = mu + step * full_dy.block(offset, 0, mu_dim, 1).reshaped(mu_dim, 1);

      // Get game context
      // Call get_full_context_(x)
      wb_.args[0] = static_cast<double *>(new_x.data());
      assert(get_full_context_.n_in() == 1);
      assert(get_full_context_.sparsity_in(0).is_dense());
      wb_.res[0] = full_context_buffer.data();
      assert(get_full_context_.n_out() == 1);

      get_full_context_(wb_.args.data(), wb_.res.data(), wb_.iw.data(), wb_.w.data(), 0);
      wb_.res[0] = nullptr;

      wb_.args[0] = static_cast<double *>(new_x.data());
      wb_.args[1] = static_cast<double *>(new_u.data());
      wb_.args[2] = static_cast<double *>(new_lamda.data());
      wb_.args[3] = static_cast<double *>(new_mu.data());
      wb_.args[4] = static_cast<double *>(full_context_buffer.data());
      wb_.args[5] = static_cast<double *>(int_param_val.ptr);
      wb_.args[6] = static_cast<double *>(double_param_val.ptr);
      assert(r_.sparsity_in(0).is_dense());
      assert(r_.sparsity_in(1).is_dense());
      assert(r_.sparsity_in(2).is_dense());
      assert(r_.sparsity_in(3).is_dense());
      assert(r_.sparsity_in(4).is_dense());
      assert(r_.sparsity_in(5).is_dense());
      assert(r_.sparsity_in(6).is_dense());
      assert(r_.n_in() == 7);

      std::vector<double> new_r_buffer(full_r0_sp.nnz());
      wb_.res[0] = new_r_buffer.data();
      wb_.res[1] = nullptr;  // Discard h_val output
      assert(r_.n_out() == 2);

      r_(wb_.args.data(), wb_.res.data(), wb_.iw.data(), wb_.w.data(), 0);
      wb_.res[0] = nullptr;

      auto new_r = get_mapped_spmatrix(full_r0_sp, new_r_buffer.data());
      new_r_norm = new_r.norm();
      logger_->debug("Line Search step {:.5f}, new_r_norm {:.5f}", step, new_r_norm);
      if (new_r_norm > (1 - bc_a_ * step) * r0_norm) {
        step *= bc_b_;
      } else {
        break;
      }
    }
    if (ls_iter == line_search_max_iter_) {
      reg_ = min(reg_ * 10.0, 0.1);
      step = 0.0;
      line_search_no_progress_counter_++;
      logger_->debug("Line Search no progress, inflating reg to {:.5f}", reg_);
    } else {
      reg_ = reg0_;
      line_search_no_progress_counter_ = 0;
    }
    logger_->debug("Line search stopped after {}/{} iterations", ls_iter + 1,
                   line_search_max_iter_);
    logger_->debug("Step size = {:.5f}", step);
    bool stop = line_search_no_progress_counter_ >= 2;

    x = new_x;
    u = new_u;
    lamda = new_lamda;
    mu = new_mu;

    bool has_converged = new_r_norm < tolerance_;
    if (!debug_got_vals_) {
      debug_full_KKT_ = full_KKT;
      debug_full_r0_ = full_r0;
      debug_full_dy = full_dy;
      debug_got_vals_ = true;
    }

    // Returns (has_converged, is_optimal, residual)
    return {stop, has_converged, is_optimal, new_r_norm};
  }

  SparseMatrixResult debug_get_full_KKT() { return get_spr(debug_full_KKT_); }
  SparseMatrixResult debug_get_reduced_KKT() { return get_spr(debug_reduced_KKT_); }
  SparseMatrixResult debug_get_full_r0() { return get_spr(debug_full_r0_); }
  std::vector<double> debug_get_context() { return debug_context; }
  MatrixXd debug_get_x() { return debug_x; }
  MatrixXd debug_get_u() { return debug_u; }
  MatrixXd debug_get_full_dy() { return debug_full_dy; }

  SparseMatrixResult get_spr(const SpMatrix &mtx) {
    SparseMatrixResult res;
    res.shape = {mtx.rows(), mtx.cols()};
    res.data.assign(mtx.valuePtr(), mtx.valuePtr() + mtx.nonZeros());
    res.row.assign(mtx.innerIndexPtr(), mtx.innerIndexPtr() + mtx.nonZeros());
    res.colind.assign(mtx.outerIndexPtr(), mtx.outerIndexPtr() + mtx.outerSize() + 1);
    return res;
  }

  bool check_spmatrix_has_nan(const SpMatrix &mtx, std::string name) {
    bool has_nan = false;
    const double *values = mtx.valuePtr();
    for (int i = 0; i < mtx.nonZeros(); ++i) {
      if (std::isnan(values[i])) {
        has_nan = true;
        break;
      }
    }
    if (has_nan) {
      logger_->warn(name + " has nan");
    }
    return has_nan;
  }

  // Make regularization matrix given regularization coefficient from each agent
  SpMatrix make_full_KKT_reg(const std::vector<double> &reg_vec) {
    int l = n_ * N_ * T_ + m_ * N_ * T_ + n_ * N_ * T_ + n_h_ * N_;

    typedef Eigen::Triplet<Scalar> T;
    std::vector<T> triplet_list;
    triplet_list.reserve(N_ * T_ * (n_ + m_));

    for (int i = 0; i < N_; ++i) {
      Scalar val = reg_vec[i];

      for (int k = 0; k < T_; ++k) {
        // Block 1: dLLi/dxi
        int x_offset = k * (n_ * N_) + i * n_;
        for (int idx = 0; idx < n_; ++idx) {
          int row_idx = x_offset + idx;
          // Diagonal matrix, so col_idx == row_idx
          triplet_list.push_back(T(row_idx, row_idx, val));
        }

        // Block 1: dLLi/dxi
        int u_offset = n_ * N_ * T_ + k * (m_ * N_) + i * m_;
        for (int idx = 0; idx < m_; ++idx) {
          int row_idx = u_offset + idx;
          triplet_list.push_back(T(row_idx, row_idx, val));
        }
      }
    }

    SpMatrix reg_mtx(l, l);
    reg_mtx.setFromTriplets(triplet_list.begin(), triplet_list.end());
    return reg_mtx;
  }

  // Find the inertia of a matrix
  // Args:
  //  solver: solver instance with symbolic factorization done.
  //        Caller must ensure identical sparsity pattern
  // Return:
  //  inertia tuple
  std::tuple<int, int, int> get_inertia(const SpMatrix &mtx,
                                        Eigen::SimplicialLDLT<SpMatrix> &solver) {
    solver.factorize(mtx);
    // Eigen::SimplicialLDLT<SpMatrix> solver;
    // solver.compute(mtx);
    if (solver.info() != Eigen::Success) {
      logger_->error("LDL decomposition failed");
      throw std::runtime_error("LDL decomposition failed");
    }
    // Check Inertia
    const auto &D = solver.vectorD();
    int pos = 0;
    int neg = 0;
    int zero = 0;
    const double epsilon = 1e-8;
    for (int i = 0; i < D.size(); ++i) {
      if (D[i] > epsilon) {
        pos++;
      } else if (D[i] < -epsilon) {
        neg++;
      } else
        zero++;
    }
    return std::make_tuple(pos, neg, zero);
  }

  // Evaluate rollout function from casadi using python arguments.
  MatrixXd casadi_rollout(py::array_t<double> x0, py::array_t<double> u,
                          py::array_t<double> int_param, py::array_t<double> double_param) {
    // Set input args
    auto x0_val = x0.request();  // py::buffer_info
    auto u_val = u.request();
    auto int_param_val = int_param.request();
    auto double_param_val = double_param.request();

    wb_.args[0] = static_cast<double *>(x0_val.ptr);
    wb_.args[1] = static_cast<double *>(u_val.ptr);
    wb_.args[2] = static_cast<double *>(int_param_val.ptr);
    wb_.args[3] = static_cast<double *>(double_param_val.ptr);
    assert(rollout_.n_in() == 4);

    const casadi::Sparsity &res_sp = rollout_.sparsity_out(0);  // 0th output sparsity
    // Allocate output buffer
    std::vector<double> res_buffer(res_sp.nnz());
    wb_.res[0] = res_buffer.data();
    assert(rollout_.n_out() == 1);
    assert(res_sp.is_dense());

    // Call work function
    rollout_(wb_.args.data(), wb_.res.data(), wb_.iw.data(), wb_.w.data(), 0);

    // SparseMatrixResult res;
    // res.shape = res_sp.size();
    // res.data = res_buffer;
    // res.row.assign(res_sp.row(), res_sp.row() + res_sp.nnz());
    // res.colind.assign(res_sp.colind(), res_sp.colind() + res_sp.size2() + 1);
    Eigen::Map<MatrixXd> res(res_buffer.data(), res_sp.size1(), res_sp.size2());
    return res;
  }
};
