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
  m.doc() = "RD3G CasADi";
  py::class_<ClassName>(m, "RD3G CasADi solver")
    .def(py::init<int, int, Scalar, Scalar, Scalar, Scalar, Scalar, Scalar,
                int, int,bool, char*>())
    .def("casadi_dr_dy", &ClassName::casadi_dr_dy);
};