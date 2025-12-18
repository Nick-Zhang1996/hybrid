// Test to use use eigen and casadi together
#include <iostream>
#include <unistd.h>
#include <limits.h>
#include <filesystem> // c++ 17
#include <dlfcn.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <casadi/casadi.hpp>
using namespace casadi;
namespace fs = std::filesystem;
fs::path get_executable_dir() {
    char result[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
    if (count != -1) {
        return fs::path(std::string(result, count)).parent_path();
    }
    return fs::current_path(); // Fallback (rare)
}

struct FuncWorkBuffer {
  std::vector<const double*> args;
  std::vector<double*> res;
  std::vector<casadi_int> iw;
  std::vector<double> w;
};

// Given a vector of CasADi functions, allocate properly sized work buffers
// NOTE [args] and [res] are vector of pointers, this fun only allocate the pointers,
// not the actual content
FuncWorkBuffer get_max_buffer(std::vector<Function> fun_vec){
  size_t sz_arg=0, sz_res=0, sz_iw=0, sz_w=0;
  for (const auto &fun : fun_vec){
    size_t sz_arg_, sz_res_, sz_iw_, sz_w_;
    fun.sz_work(sz_arg_, sz_res_, sz_iw_, sz_w_);
    std::cout << sz_arg_ << " " << sz_res_ << " " << sz_iw_ << " " << sz_w_ << std::endl;
    sz_arg = sz_arg > sz_arg_ ? sz_arg : sz_arg_;
    sz_res = sz_res > sz_res_ ? sz_res : sz_res_;
    sz_iw = sz_iw > sz_iw_ ? sz_iw : sz_iw_;
    sz_w = sz_w > sz_w_ ? sz_w : sz_w_;
  }
  std::vector<const double*> args(sz_arg);
  std::vector<double*> res(sz_res);
  std::vector<casadi_int> iw(sz_iw);
  std::vector<double> w(sz_w);
  return {args, res, iw, w};
}

void casadi_eigen(){
  fs::path exe_dir = get_executable_dir();
  // TODO dynamically construct game
  int N = 3;
  int T = 20;
  std::stringstream ss;
  ss << "lib" << "car_merge_kinematic_bicycle_casadi_N" << N << "_T" << T << ".so";
  fs::path lib_path = exe_dir / ".." / "lib" / ss.str();
  Function r = external("r",lib_path.string());
  Function dr_dy = external("dr_dy",lib_path.string());
  Function get_n = external("get_n",lib_path.string());
  Function get_m = external("get_m",lib_path.string());

  Sparsity dr_dy_res_sp = dr_dy.sparsity_out(0);
  Eigen::VectorXd res_buffer = Eigen::VectorXd::Zero(dr_dy_res_sp.nnz());

  FuncWorkBuffer wb = get_max_buffer({r, dr_dy});
  std::cout << "r.n_in() " << r.n_in()  << std::endl;


  // Solver call should have the following args:
  // game (module name), N, T -> this determines dll.
  // x0, u_guess, game config (in addition to N,T), these are config param variables
  int n = get_n(wb.args.data(), wb.res.data(), wb.iw.data(), wb.w.data(), 0);
  int m = get_m(wb.args.data(), wb.res.data(), wb.iw.data(), wb.w.data(), 0);

  // dr_dy(x, u, lamda, mu)
  // TODO: can we call?
  // TODO: are values correct?
  Eigen::MatrixXd x = Eigen::MatrixXd::Random(N*n, T);
  Eigen::MatrixXd u = Eigen::MatrixXd::Random(N*m, T);
  Eigen::MatrixXd lamda = Eigen::MatrixXd::Random(N*n, T);
  Eigen::MatrixXd mu = Eigen::MatrixXd::Random(N*N, T);

  wb.args[0] = x.data();
  wb.args[1] = u.data();
  wb.args[2] = lamda.data();
  wb.args[3] = mu.data();
  wb.res[0] = res_buffer.data();

  auto start = std::chrono::high_resolution_clock::now();

  // arguments: input_ptrs, output_ptrs, int_workspace, double_workspace, mem_id
  for (int i=0; i<5000; i++){
    dr_dy(wb.args.data(), wb.res.data(), wb.iw.data(), wb.w.data(), 0); // 0.034ms
  }

  Eigen::MappedSparseMatrix<double, Eigen::ColMajor, casadi_int> dr_dy_res(
    dr_dy_res_sp.size1(), dr_dy_res_sp.size2(), dr_dy_res_sp.nnz(),
    const_cast<casadi_int*>(dr_dy_res_sp.colind()),
    const_cast<casadi_int*>(dr_dy_res_sp.row()),
    res_buffer.data()
  );

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;

  std::cout << "raw() Time: " << elapsed.count() << " s" << std::endl;
  std::cout << "val: " << dr_dy_res.norm() << std::endl;

}


int main(){
    casadi_eigen();
}