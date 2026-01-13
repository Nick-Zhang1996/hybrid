#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "rd3g_casadi.hpp"
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
    .def(py::init<int, int, int, Scalar, Scalar, Scalar, Scalar, Scalar, Scalar, Scalar,
                int, int,bool, std::string, std::string>())
    .def("dr_dy", &ClassName::casadi_dr_dy)
    .def("solve", &ClassName::solve)
    .def("step", &ClassName::step)
    .def("debug_get_full_KKT", &ClassName::debug_get_full_KKT)
    .def("debug_get_reduced_KKT", &ClassName::debug_get_reduced_KKT)
    .def("debug_get_full_r0", &ClassName::debug_get_full_r0);
};