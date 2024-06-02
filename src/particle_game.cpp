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
      .def(py::init<int,int,int,int, float,float,float,float,float, np_array,np_array,np_array,np_array,np_array,np_array,np_array>())
      .def("set_A", &ParticleGame::set_A)
      .def("set_B", &ParticleGame::set_B)
      .def("set_x0", &ParticleGame::set_x0)

      .def("f", &ParticleGame::f)
      .def("df_dx", &ParticleGame::df_dx)
      .def("df_du", &ParticleGame::df_du)
      .def("h", &ParticleGame::h)
      .def("dh_dxi", &ParticleGame::dh_dxi)
      .def("dh_dxj", &ParticleGame::dh_dxj)
      .def("dh_dxi_dxi", &ParticleGame::dh_dxi_dxi)
      .def("dh_dxi_dxj", &ParticleGame::dh_dxi_dxj)
      .def("dh_dxj_dxi", &ParticleGame::dh_dxj_dxi)
      .def("dh_dxj_dxj", &ParticleGame::dh_dxj_dxj)
      .def("J", &ParticleGame::J)
      .def("dJ_dx", &ParticleGame::dJ_dx)
      .def("dJ_du", &ParticleGame::dJ_du)
      .def("dJ_dxdx", &ParticleGame::dJ_dxdx)
      .def("dL_dx_ik", &ParticleGame::dL_dx_ik)
      .def("dLLi_dx", &ParticleGame::dLLi_dx)
      .def("dL_du", &ParticleGame::dL_du)
      .def("dLLi_du", &ParticleGame::dLLi_du)
      .def("dBh_dxi", &ParticleGame::dBh_dxi)
      .def("dBh_dxj", &ParticleGame::dBh_dxj)
      .def("dBh_dxi_dxi", &ParticleGame::dBh_dxi_dxi)
      .def("dBh_dxi_dxj", &ParticleGame::dBh_dxi_dxj)
      .def("dBh_dxj_dxj", &ParticleGame::dBh_dxj_dxj)
      .def("dF_dx", &ParticleGame::dF_dx)
      .def("dF0_dx", &ParticleGame::dF0_dx)
      .def("dh_dx", &ParticleGame::dh_dx)
      .def("dLLi_dxdx", &ParticleGame::dLLi_dxdx)
      .def("dr_dx", &ParticleGame::dr_dx)
      .def("dr_du", &ParticleGame::dr_du)
      .def("dr_dlamda", &ParticleGame::dr_dlamda)
      .def("dLLi_dx_dmu", &ParticleGame::dLLi_dx_dmu)
      .def("dr_dmu", &ParticleGame::dr_dmu)

      .def("print_dim", &ParticleGame::print_dim)
      .def("test_bool_array",&ParticleGame::test_bool_array)
      .def("three_dim", &ParticleGame::three_dim)
      ;
}
