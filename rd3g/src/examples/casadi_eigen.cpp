// Test to use use eigen and casadi together
#include <unistd.h>
#include <limits.h>
#include <filesystem> // c++ 17
#include <dlfcn.h>
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
void casadi_eigen(){
  fs::path exe_dir = get_executable_dir();
  // TODO dynamically construct game
  int N = 3;
  int T = 20;
  fs::path lib_path = exe_dir / ".." / "lib" / "libcar_merge_kinematic_bicycle_casadi_N3_T20.so";
  Function r = external("r",lib_path.string());
  Function dr_dy = external("dr_dy",lib_path.string());
  Function get_n = external("get_n",lib_path.string());
  Function get_m = external("get_m",lib_path.string());

  // Create working buffer
  size_t sz_arg, sz_res, sz_iw, sz_w;
  r.sz_work(sz_arg, sz_res, sz_iw, sz_w);

  // TODO better way to find necessary working buffer size
  size_t sz_arg_alt, sz_res_alt, sz_iw_alt, sz_w_alt;
  dr_dy.sz_work(sz_arg_alt, sz_res_alt, sz_iw_alt, sz_w_alt);
  sz_arg = std::max(sz_arg, sz_arg_alt);
  sz_res = std::max(sz_res, sz_res_alt);
  sz_iw = std::max(sz_iw, sz_iw_alt);
  sz_w = std::max(sz_w, sz_w_alt);
  
  std::vector<const double*> args(sz_arg);
  std::vector<double*> res(sz_res);
  std::vector<casadi_int> iw(sz_iw);
  std::vector<double> w(sz_w);

  // Solver call should have the following args:
  // game (module name), N, T -> this determines dll.
  // x0, u_guess, game config (in addition to N,T), these are config param variables
  int n = get_n(args.data(), res.data(), iw.data(), w.data(), 0);
  int m = get_m(args.data(), res.data(), iw.data(), w.data(), 0);

  double x[] = {0.1, 0.2};
  double lamda[1] = {0.1};
  double out_buffer[3];
  std::vector<double> accumulator(3,0.0);

  args[0] = x;
  args[1] = lamda;
  res[0] = out_buffer;

  auto start = std::chrono::high_resolution_clock::now();

  for (int i=0;i<1000;i++){
    // arguments: input_ptrs, output_ptrs, int_workspace, double_workspace, mem_id
    r(args.data(), res.data(), iw.data(), w.data(), 0);
    for (int j=0;j<3;j++){
      accumulator[j] += out_buffer[j];
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;

  std::cout << "raw() Time: " << elapsed.count() << " s" << std::endl;
  std::cout << "val: " << accumulator << std::endl;

}


int main(){
    casadi_eigen();
}