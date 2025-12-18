// RD3G solver compatible with casadi codegen J(), Jfi(), f(), h() etc.
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

namespace fs = std::filesystem;
fs::path get_executable_dir()
{
  char result[PATH_MAX];
  ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
  if (count != -1)
  {
    return fs::path(std::string(result, count)).parent_path();
  }
  return fs::current_path(); // Fallback (rare)
}

struct FuncWorkBuffer
{
  std::vector<const double *> args;
  std::vector<double *> res;
  std::vector<casadi_int> iw;
  std::vector<double> w;
};

// Compressed Colume Storage matrix
struct SparseMatrixResult {
    std::pair<int, int> shape;         // (rows, cols)
    std::vector<casadi_int> row;
    std::vector<casadi_int> colind;
    std::vector<double> data;
};


// Given a vector of CasADi functions, allocate properly sized work buffers
// NOTE [args] and [res] are vector of pointers, this fun only allocate the pointers,
// not the actual content
FuncWorkBuffer get_max_buffer(std::vector<cas::Function> fun_vec)
{
  size_t sz_arg = 0, sz_res = 0, sz_iw = 0, sz_w = 0;
  for (const auto &fun : fun_vec)
  {
    size_t sz_arg_, sz_res_, sz_iw_, sz_w_;
    fun.sz_work(sz_arg_, sz_res_, sz_iw_, sz_w_);
    std::cout << sz_arg_ << " " << sz_res_ << " " << sz_iw_ << " " << sz_w_ << std::endl;
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

class Rd3gCasadi
{

protected:
  int n_, m_, N_, T_;
  Scalar dt_, rho_, rho_b_, bc_a_, bc_b_;
  Scalar tolerance_;
  int backtracking_max_iter_;
  int max_iterations_;
  // dim: N*n
  Matrix x0_;
  mutable Profiler<false> profiler_;
  bool verbose_;

  // CasADi related
  FuncWorkBuffer wb_;

  cas::Function dr_dy;
  cas::Function r;

public:
  Rd3gCasadi(const int N, const int T, const Scalar dt, const Scalar rho,
               const Scalar rho_b, const Scalar bc_a, const Scalar bc_b,
               const Scalar tolerance, const int backtracking_max_iter,
               const int max_iter, const bool verbose,
               const char *casadi_module_name)
      : N_{N}, T_{T}, dt_{dt}, rho_{rho}, rho_b_{rho_b}, bc_a_{bc_a},
        bc_b_{bc_b}, tolerance_{tolerance},
        backtracking_max_iter_{backtracking_max_iter}, x0_{}, profiler_{},
        max_iterations_{max_iter}, verbose_(verbose)
  {
    // Load CasADi dll for given game, each (N,T) pair corresponds to a unique dll.
    fs::path exe_dir = get_executable_dir();
    std::stringstream ss;
    ss << "lib" << casadi_module_name << "_N" << N << "_T" << T << ".so";
    // TODO more informative error msg
    // Load Functions
    fs::path lib_path = exe_dir / ".." / "lib" / ss.str();
    cas::Function r = cas::external("r", lib_path.string());
    cas::Function dr_dy = cas::external("dr_dy", lib_path.string());
    cas::Function get_n = cas::external("get_n", lib_path.string());
    cas::Function get_m = cas::external("get_m", lib_path.string());

    wb_ = get_max_buffer({r, dr_dy});

    // Solver call should have the following args:
    // game (module name), N, T -> this determines dll.
    // x0, u_guess, game config (in addition to N,T), these are config param variables
    // NOTE n,m may need to be template variables for performance
    n_ = get_n(wb_.args.data(), wb_.res.data(), wb_.iw.data(), wb_.w.data(), 0);
    m_ = get_m(wb_.args.data(), wb_.res.data(), wb_.iw.data(), wb_.w.data(), 0);
  }

  void set_x0(const Matrix &val) { x0_ = Matrix(val); }
  void post_step_update() { rho_ *= rho_b_; }

  SparseMatrixResult casadi_dr_dy(py::array_t<double> x,
                      py::array_t<double> u,
                      py::array_t<double> lamda,
                      py::array_t<double> mu,
                      py::array_t<double> int_param,
                      py::array_t<double> double_param)
  {
    //py::buffer_info x_buf = x.request();
    //double* x_ptr = static_cast<double*>(x_buf.ptr);
    // Set input args
    wb_.args[0] = static_cast<double*>(x.request().ptr);
    wb_.args[1] = static_cast<double*>(u.request().ptr);
    wb_.args[2] = static_cast<double*>(lamda.request().ptr);
    wb_.args[3] = static_cast<double*>(mu.request().ptr);
    wb_.args[4] = static_cast<double*>(int_param.request().ptr);
    wb_.args[5] = static_cast<double*>(double_param.request().ptr);
    assert (dr_dy.n_in() == 6);

    const casadi::Sparsity& res_sp = dr_dy.sparsity_out(0);
    // Prepare output buffer
    std::vector<double> res_buffer(res_sp.nnz());
    wb_.res[0] = res_buffer.data();
    assert (dr_dy.n_out() == 1);

    // Call work function
    dr_dy(wb_.args.data(), wb_.res.data(), wb_.iw.data(), wb_.w.data(), 0);

    SparseMatrixResult res;
    res.shape = res_sp.size();
    res.data = res_buffer;
    res.row.assign(res_sp.row(), res_sp.row() + res_sp.nnz());
    res.colind.assign(res_sp.colind(), res_sp.colind() + res_sp.size2() + 1);
    return res;
  }
};