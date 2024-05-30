#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <Eigen/LU>
#include "particle_game.h"

// N.B. this would equally work with Eigen-types that are not predefined. For example replacing
// all occurrences of "Eigen::MatrixXd" with "MatD", with the following definition:
//
//  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatD;

// ----------------
// regular C++ code
// ----------------

// ----------------
// Python interface
// ----------------

namespace py = pybind11;

PYBIND11_MODULE(particle_game,m)
{
  m.doc() = "pybind11 interface for c++/Eigen particle_game";
  py::class_<ParticleGame>(m,"ParticleGame")
      .def(py::init())
      .def("set_A", &ParticleGame::set_A)
      .def("set_B", &ParticleGame::set_B)
      .def("f", &ParticleGame::f);
}
