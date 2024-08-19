#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <eigen3/Eigen/LU>
#include "residual_game.h"

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

// ResidualGame is not an abstract class
// this is only a template

/*
namespace py = pybind11;
using ClassName = ResidualGame;


PYBIND11_MODULE(residual_game,m)
{
  m.doc() = "pybind11 interface for c++/Eigen particle_game";
  py::class_<ResidualGame>(m,"ResidualGame")
      .def(py::init<int,int,int,int, float,float,float,float,float, np_array,np_array,np_array,np_array,np_array,np_array,np_array>())
      ;
}
*/
