// RD3G solver compatible with casadi codegen
#pragma once
// #define EIGEN_RUNTIME_NO_MALLOC
// Eigen::internal::set_is_malloc_allowed(false);
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SparseCore>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string.h>
#include <string>
#include <unistd.h>
#include <limits.h>
#include <filesystem> // c++ 17
#include <dlfcn.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

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
using SpMatrix = Eigen::SparseMatrix<double, Eigen::ColMajor>;
using MappedSparseMatrix = Eigen::Map<Eigen::SparseMatrix<double, Eigen::ColMajor, casadi_int>>;

namespace fs = std::filesystem;

// Work buffer for casadi functions
struct FuncWorkBuffer
{
  std::vector<const double *> args;
  std::vector<double *> res;
  std::vector<casadi_int> iw;
  std::vector<double> w;
};

// Compressed Colume Storage matrix, for passing to Python
struct SparseMatrixResult {
    std::pair<int, int> shape;         // (rows, cols)
    std::vector<casadi_int> row;
    std::vector<casadi_int> colind;
    std::vector<double> data;
};


// Given a vector of CasADi functions, allocate sufficiently sized work buffers
// NOTE [args] and [res] are vector of pointers, this fun only allocate the pointers,
// not the actual content
FuncWorkBuffer get_max_buffer(std::vector<cas::Function> fun_vec)
{
  size_t sz_arg = 0, sz_res = 0, sz_iw = 0, sz_w = 0;
  for (const auto &fun : fun_vec)
  {
    size_t sz_arg_, sz_res_, sz_iw_, sz_w_;
    fun.sz_work(sz_arg_, sz_res_, sz_iw_, sz_w_);
    //std::cout << sz_arg_ << " " << sz_res_ << " " << sz_iw_ << " " << sz_w_ << std::endl;
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

cas::Function safe_load_fun(std::string fun_name, fs::path lib_path)
{
    cas::Function fun = cas::external(fun_name, lib_path.string());

    if (fun.is_null()) {
      throw std::runtime_error("Failed to load " + fun_name + " from " + lib_path.string());
    }
    return fun;
}

// Create MappedSparseMatrix sparsity pattern and data
inline MappedSparseMatrix get_mapped_spmatrix(casadi::Sparsity sp, double* data){
  MappedSparseMatrix retval(
    sp.size1(), sp.size2(), sp.nnz(),
    const_cast<casadi_int*>(sp.colind()),
    const_cast<casadi_int*>(sp.row()),
    data
  );
  return retval;
}

class Rd3gCasadi
{

protected:
  int n_, m_, N_, T_;
  Scalar dt_, rho_, rho_b_, bc_a_, bc_b_;
  Scalar tolerance_;
  int backtracking_max_iter_;
  int max_iterations_;
  // dim: N*n
  MatrixXd x0_;
  mutable Profiler<false> profiler_;
  bool verbose_;

  // CasADi work buffer
  FuncWorkBuffer wb_;

  cas::Function r_;
  cas::Function dr_dy_;
  cas::Function rollout_;
  cas::Function h_;
  cas::Function collision_h_;

public:
  // TODO cleanup constructor arguments, remove useless or redundant ones

  // NOTE n,m may need to be template variables for performance
  Rd3gCasadi(const int N, const int T, const Scalar dt, const Scalar rho,
               const Scalar rho_b, const Scalar bc_a, const Scalar bc_b,
               const Scalar tolerance, const int backtracking_max_iter,
               const int max_iter, const bool verbose,
               const std::string base_dir,
               const std::string casadi_module_name)
      : N_{N}, T_{T}, dt_{dt}, rho_{rho}, rho_b_{rho_b}, bc_a_{bc_a},
        bc_b_{bc_b}, tolerance_{tolerance},
        backtracking_max_iter_{backtracking_max_iter}, x0_{}, profiler_{},
        max_iterations_{max_iter}, verbose_(verbose)
  {

    // Load CasADi dll for given game following naming convention,
    // game (module name), N, T -> this determines a unique dll name.
    std::stringstream ss;
    ss << "lib" << casadi_module_name << "_N" << N << "_T" << T << ".so";
    fs::path lib_path = fs::path(base_dir) / "rd3g" / "src" / "build" / "lib" / ss.str();

    r_ = safe_load_fun("r", lib_path);
    dr_dy_ = safe_load_fun("dr_dy", lib_path);
    rollout_ = safe_load_fun("rollout", lib_path);
    h_ = safe_load_fun("h", lib_path);
    collision_h_ = safe_load_fun("collision_h", lib_path);

    cas::Function get_n = safe_load_fun("get_n", lib_path);
    cas::Function get_m = safe_load_fun("get_m", lib_path);

    wb_ = get_max_buffer({r_, dr_dy_, rollout_, h_, collision_h_, get_n, get_m});

    n_ = get_n(wb_.args.data(), wb_.res.data(), wb_.iw.data(), wb_.w.data(), 0);
    m_ = get_m(wb_.args.data(), wb_.res.data(), wb_.iw.data(), wb_.w.data(), 0);
    std::cout << "Rd3gCasadi Initialized" << std::endl;
  }

  void set_x0(const MatrixXd &val) { x0_ = MatrixXd(val); }
  void post_step_update() { rho_ *= rho_b_; }

  // Evaluate dr_dy function from casadi using python arguments.
  // Demonstrating data representation conversion and call procedure
  SparseMatrixResult casadi_dr_dy(py::array_t<double> x,
                      py::array_t<double> u,
                      py::array_t<double> lamda,
                      py::array_t<double> mu,
                      py::array_t<double> int_param,
                      py::array_t<double> double_param)
  {
    // Set input args
    auto x_val = x.request(); // py::buffer_info
    auto u_val = u.request();
    auto lamda_val = lamda.request();
    auto mu_val = mu.request();
    auto int_param_val = int_param.request();
    auto double_param_val = double_param.request();

    wb_.args[0] = static_cast<double*>(x_val.ptr);
    wb_.args[1] = static_cast<double*>(u_val.ptr);
    wb_.args[2] = static_cast<double*>(lamda_val.ptr);
    wb_.args[3] = static_cast<double*>(mu_val.ptr);
    wb_.args[4] = static_cast<double*>(int_param_val.ptr);
    wb_.args[5] = static_cast<double*>(double_param_val.ptr);
    assert (dr_dy_.n_in() == 6);

    const casadi::Sparsity& res_sp = dr_dy_.sparsity_out(0); // 0th output sparsity
    // Allocate output buffer
    std::vector<double> res_buffer(res_sp.nnz());
    wb_.res[0] = res_buffer.data();
    assert (dr_dy_.n_out() == 1);

    // Call work function
    dr_dy_(wb_.args.data(), wb_.res.data(), wb_.iw.data(), wb_.w.data(), 0);

    SparseMatrixResult res;
    res.shape = res_sp.size();
    res.data = res_buffer;
    res.row.assign(res_sp.row(), res_sp.row() + res_sp.nnz());
    res.colind.assign(res_sp.colind(), res_sp.colind() + res_sp.size2() + 1);
    return res;
  }

  void solve(py::array_t<double> x0,
                      py::array_t<double> u_guess,
                      py::array_t<double> int_param,
                      py::array_t<double> double_param)
  {
    // Set input args
    auto x0_val = x0.request(); // py::buffer_info
    auto u_guess_val = u_guess.request();
    auto int_param_val = int_param.request();
    auto double_param_val = double_param.request();
    //TODO ensure buffer is continuous


    // Call rollout(x0, u_guess, int_param, double_param) -> x
    wb_.args[0] = static_cast<double*>(x0_val.ptr);
    wb_.args[1] = static_cast<double*>(u_guess_val.ptr);
    wb_.args[2] = static_cast<double*>(int_param_val.ptr);
    wb_.args[3] = static_cast<double*>(double_param_val.ptr);
    assert (rollout_.n_in() == 4);

    casadi::Sparsity x_sp = rollout_.sparsity_out(0);
    std::vector<double> x_buffer(x_sp.nnz());
    wb_.res[0] = x_buffer.data();
    assert (rollout_.n_out() == 1);

    assert (x_sp.is_dense());
    rollout_(wb_.args.data(), wb_.res.data(), wb_.iw.data(), wb_.w.data(), 0);
    wb_.res[0] = nullptr; // Avoid accidentally overwriting the buffer
    Eigen::Map<MatrixXd> x(x_buffer.data(), x_sp.size1(), x_sp.size2());

    // u_guess, dense
    auto u = u_guess.cast<MatrixXd>();

    MatrixXd lamda = MatrixXd::Zero(N_*n_, T_);
    MatrixXd mu = MatrixXd::Zero(N_*N_, T_);

    for (int iter=0; iter<max_iterations_; iter++){
      step(x, u, lamda, mu, int_param, double_param);
    }

  }

  // Take one Newton step, modify x,u,lamda,mu in place
  // x_ref: n*N,T
  // u_ref: m*N,T
  // lamda: n*N,T
  // mu: n_hi*N, 1
  // Returns (has_converged, is_optimal, residual)
  std::tuple<bool, bool, Scalar> 
  step(Eigen::Ref<MatrixXd> x,
            Eigen::Ref<MatrixXd> u,
            Eigen::Ref<MatrixXd> lamda,
            Eigen::Ref<MatrixXd> mu,
            py::array_t<double>& int_param,
            py::array_t<double>& double_param)
  {
    // Solve r0 + H @ dy = 0
    // i.e. full_r0 + full_KKT @ <dx, du, dlambda, dmu> = 0
    // identify inactive constraints (mu)
    // skim down H, dy, remove inactive constraints, dual variables
    // Solve for dy

    // Call r(), dr_dy() to get full_r0 and full_KKT
    auto int_param_val = int_param.request();
    auto double_param_val = double_param.request();

    // full_r0 = r(x, u, lamda, mu, int_param, double_param)
    wb_.args[0] = static_cast<double*>(x.data());
    wb_.args[1] = static_cast<double*>(u.data());
    wb_.args[2] = static_cast<double*>(lamda.data());
    wb_.args[3] = static_cast<double*>(mu.data());
    wb_.args[4] = static_cast<double*>(int_param_val.ptr);
    wb_.args[5] = static_cast<double*>(double_param_val.ptr);
    assert (r_.n_in() == 6);
    assert (r_.sparsity_in(0).is_dense());
    assert (r_.sparsity_in(1).is_dense());
    assert (r_.sparsity_in(2).is_dense());
    assert (r_.sparsity_in(3).is_dense());
    assert (r_.sparsity_in(4).is_dense());
    assert (r_.sparsity_in(5).is_dense());

    casadi::Sparsity full_r0_sp = r_.sparsity_out(0);
    std::vector<double> full_r0_buffer(full_r0_sp.nnz());
    wb_.res[0] = full_r0_buffer.data();
    assert (r_.n_out() == 1);

    r_(wb_.args.data(), wb_.res.data(), wb_.iw.data(), wb_.w.data(), 0);
    wb_.res[0] = nullptr;

    Eigen::Map<MatrixXd> full_r0(full_r0_buffer.data(), full_r0_sp.size1(), full_r0_sp.size2());

    // full_KKT = dr_dy(x, u, lamda, mu, int_param, double_param)
    // Same input as r() call
    assert (dr_dy_.n_in() == 6);
    assert (dr_dy_.sparsity_in(0).is_dense());
    assert (dr_dy_.sparsity_in(1).is_dense());
    assert (dr_dy_.sparsity_in(2).is_dense());
    assert (dr_dy_.sparsity_in(3).is_dense());
    assert (dr_dy_.sparsity_in(4).is_dense());
    assert (dr_dy_.sparsity_in(5).is_dense());

    casadi::Sparsity full_KKT_sp = dr_dy_.sparsity_out(0);
    std::vector<double> full_KKT_buffer(full_KKT_sp.nnz());
    wb_.res[0] = full_KKT_buffer.data();
    assert (dr_dy_.n_out() == 1);

    dr_dy_(wb_.args.data(), wb_.res.data(), wb_.iw.data(), wb_.w.data(), 0);
    wb_.res[0] = nullptr;

    Eigen::Map<MatrixXd> full_KKT(full_r0_buffer.data(), full_r0_sp.size1(), full_r0_sp.size2());

    return {false, false, 1.0};

  }
};