// Example: Dynamically load a .so from casadi codegen
// Run the function with CasADi and run directly, compare performance difference
#include <chrono>
#include <unistd.h>
#include <limits.h>
#include <filesystem> // c++ 17
#include <dlfcn.h>
#include <casadi/casadi.hpp>
#include "casadi_toy_example.h"
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

// Load function and run with CasADi, 13ms
void usage_external(){
  std::cout << "---" << std::endl;
  std::cout << "Usage from CasADi C++:" << std::endl;
  std::cout << std::endl;

  // Use CasADi's "external" to load the compiled function
  // CasADi external() needs an exact path, unlike dlopen() which can load from RUNPATH
  fs::path exe_dir = get_executable_dir();
  fs::path lib_path = exe_dir / ".." / "lib" / "libtoy_example.so";
  Function r = external("r",lib_path.string());

  DM val = DM::zeros(3);

  auto start = std::chrono::high_resolution_clock::now();
  for (int i=0; i<1000; i++){
    DM x = {0.1,0.2};
    DM lamda = {0.1};
    DM decision_vars = vertcat(x, lamda);
    std::vector<DM> args = {decision_vars};
    std::vector<DM> res = r(args);
    val += res.at(0);
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;

  std::cout << "external() Time: " << elapsed.count() << " s" << std::endl;
  std::cout << "val: " << val << std::endl;
}

// Load function and run function directly 0.17ms
void usage_raw(){
  fs::path exe_dir = get_executable_dir();
  fs::path lib_path = exe_dir / ".." / "lib" / "libtoy_example.so";
  Function r = external("r",lib_path.string());

  size_t sz_arg, sz_res, sz_iw, sz_w;
  r.sz_work(sz_arg, sz_res, sz_iw, sz_w);
  
  std::vector<casadi_int> iw(sz_iw);
  std::vector<double> w(sz_w);

  std::vector<const double*> args(sz_arg);
  std::vector<double*> res(sz_res);

  double x[2] = {0.1, 0.2};
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
    usage_external();
    usage_raw();
}