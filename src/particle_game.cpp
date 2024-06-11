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
      .def("post_step_update", &ParticleGame::post_step_update)

      .def("f", &ParticleGame::f)
      .def("df_dx", &ParticleGame::df_dx)
      .def("df_du", &ParticleGame::df_du)
      .def("h", &ParticleGame::h)
      .def("dh_dx", &ParticleGame::dh_dx)
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
      .def("dLLi_dx_dmu", &ParticleGame::dLLi_dx_dmu)
      .def("dBh_dxi", &ParticleGame::dBh_dxi)
      .def("dBh_dxj", &ParticleGame::dBh_dxj)
      .def("dBh_dxi_dxi", &ParticleGame::dBh_dxi_dxi)
      .def("dBh_dxi_dxj", &ParticleGame::dBh_dxi_dxj)
      .def("dBh_dxj_dxj", &ParticleGame::dBh_dxj_dxj)
      .def("dF_dx", &ParticleGame::dF_dx)
      .def("dF0_dx", &ParticleGame::dF0_dx)
      .def("dh_dx", &ParticleGame::dh_dx)
      .def("dLLi_dxdx", &ParticleGame::dLLi_dxdx)
      .def("dr_dx", static_cast<np_array (ParticleGame::*)(const std::vector<np_array>& x, const std::vector<np_array>& u, const std::vector<np_array>& lamda, const std::vector<np_array>& mu, const std::vector<np_array>& h_plus_mask)>(&ParticleGame::dr_dx) )
      .def("dr_du", static_cast<np_array (ParticleGame::*)(const std::vector<np_array>& x, const std::vector<np_array>& u, const std::vector<np_array>& lamda, const std::vector<np_array>& mu, const std::vector<np_array>& h_plus_mask)>(&ParticleGame::dr_du) )
      .def("dr_dlamda", static_cast<np_array (ParticleGame::*)(const std::vector<np_array>& x, const std::vector<np_array>& u, const std::vector<np_array>& lamda, const std::vector<np_array>& mu, const std::vector<np_array>& h_plus_mask)>(&ParticleGame::dr_dlamda) )
      .def("dr_dmu", static_cast<np_array (ParticleGame::*)(const std::vector<np_array>& x, const std::vector<np_array>& u, const std::vector<np_array>& lamda, const std::vector<np_array>& mu, const std::vector<np_array>& h_plus_mask)>(&ParticleGame::dr_dmu) )
      .def("r", &ParticleGame::r)
      .def("dr_dy", &ParticleGame::dr_dy)
      .def("getHplusMask", &ParticleGame::getHplusMask)
      .def("step", &ParticleGame::step)
      

      .def("SparseQR", &ParticleGame::SparseQR)
      .def("LeastSquaresConjugateGradient", &ParticleGame::LeastSquaresConjugateGradient)
      .def("print_dim", &ParticleGame::print_dim)
      .def("test_bool_array",&ParticleGame::test_bool_array)
      .def("three_dim", &ParticleGame::three_dim)
      .def("pass_by_ref", &ParticleGame::pass_by_ref)
      ;
}
