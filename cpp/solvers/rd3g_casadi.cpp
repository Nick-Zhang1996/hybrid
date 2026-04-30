#include "rd3g_casadi.hpp"

#include <Python.h>  // For testing include
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <eigen3/Eigen/LU>

namespace py = pybind11;
// using Scalar = double;
using ClassName = Rd3gCasadi;

void bind_sparse_struct(py::module &m) {
  py::class_<SparseMatrixResult>(m, "SparseMatrixResult")
      .def_readonly("data", &SparseMatrixResult::data)
      .def_readonly("row", &SparseMatrixResult::row)
      .def_readonly("colind", &SparseMatrixResult::colind)
      .def_readonly("shape", &SparseMatrixResult::shape);
}

PYBIND11_MODULE(rd3g_casadi, m) {
  bind_sparse_struct(m);
  m.doc() = "RD3G CasADi solver";
  py::class_<ClassName>(m, "Rd3gCasadi")
      .def(py::init<int,          // N
                    int,          // T
                    int,          // n_hi
                    Scalar,       // dt
                    Scalar,       // bc_a
                    Scalar,       // bc_b
                    Scalar,       // reg
                    Scalar,       // reg_inertia
                    bool,         // inertia_correction
                    bool,         // rollout_each_step
                    bool,         // precondition_with_potential
                    bool,         // variational_gne
                    Scalar,       // tolerance
                    Scalar,       // tau_decay
                    int,          // line_search_max_iter
                    int,          // max_failed_line_search
                    int,          // max_iter
                    int,          // max_in_reg_iter
                    Scalar,       // max_in_reg_val
                    std::string,  // linear_solver_method
                    int,          // verbose
                    std::string,  // base_dir
                    std::string   // casadi_module_name
                    >(),
           py::arg("N"), py::arg("T"), py::arg("n_hi"), py::arg("dt"), py::arg("bc_a"),
           py::arg("bc_b"), py::arg("reg"), py::arg("reg_inertia"),
           py::arg("inertia_correction"), py::arg("rollout_each_step"),
           py::arg("precondition_with_potential"), py::arg("variational_gne"),
           py::arg("tolerance"), py::arg("tau_decay"), py::arg("line_search_max_iter"),
           py::arg("max_failed_line_search"), py::arg("max_iter"),
           py::arg("max_in_reg_iter"), py::arg("max_in_reg_val"),
           py::arg("linear_solver_method"), py::arg("verbose"), py::arg("base_dir"),
           py::arg("casadi_module_name"))
      .def("dr_dy", &ClassName::casadi_dr_dy)
      .def("rollout", &ClassName::casadi_rollout)
      .def("get_full_context", &ClassName::get_full_context)
      .def("solve", &ClassName::solve)
      .def("step", &ClassName::step)
      .def("debug_get_full_KKT", &ClassName::debug_get_full_KKT)
      .def("debug_get_full_r0", &ClassName::debug_get_full_r0)
      .def("debug_get_context", &ClassName::debug_get_context)
      .def("debug_get_x", &ClassName::debug_get_x)
      .def("debug_get_u", &ClassName::debug_get_u)
      .def("debug_get_full_dy", &ClassName::debug_get_full_dy);
};
