#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <Eigen/LU>
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

namespace py = pybind11;
using ClassName = ResidualGame;

PYBIND11_MODULE(residual_game,m)
{
  m.doc() = "pybind11 interface for c++/Eigen particle_game";
  py::class_<ResidualGame>(m,"ResidualGame")
      .def(py::init<int,int,int,int, float,float,float,float,float, np_array,np_array,np_array,np_array,np_array,np_array,np_array>())
      .def("set_A", &ClassName::set_A)
      .def("set_B", &ClassName::set_B)
      .def("set_x0", &ClassName::set_x0)
      .def("post_step_update", &ClassName::post_step_update)

      .def("f", &ClassName::f)
      .def("df_dx", &ClassName::df_dx)
      .def("df_du", &ClassName::df_du)
      .def("h", &ClassName::h)
      .def("dh_dx", &ClassName::dh_dx)
      .def("dh_dxi", &ClassName::dh_dxi)
      .def("dh_dxj", &ClassName::dh_dxj)
      .def("dh_dxi_dxi", &ClassName::dh_dxi_dxi)
      .def("dh_dxi_dxj", &ClassName::dh_dxi_dxj)
      .def("dh_dxj_dxi", &ClassName::dh_dxj_dxi)
      .def("dh_dxj_dxj", &ClassName::dh_dxj_dxj)
      .def("J", &ClassName::J)
      .def("dJ_dx", &ClassName::dJ_dx)
      .def("dJ_du", &ClassName::dJ_du)
      .def("dJ_dxdx", &ClassName::dJ_dxdx)
      .def("dL_dx_ik", &ClassName::dL_dx_ik)
      .def("dLLi_dx", &ClassName::dLLi_dx)
      .def("dL_du", &ClassName::dL_du)
      .def("dLLi_du", &ClassName::dLLi_du)
      .def("dLLi_dx_dmu", &ClassName::dLLi_dx_dmu)
      .def("dBh_dxi", &ClassName::dBh_dxi)
      .def("dBh_dxj", &ClassName::dBh_dxj)
      .def("dBh_dxi_dxi", &ClassName::dBh_dxi_dxi)
      .def("dBh_dxi_dxj", &ClassName::dBh_dxi_dxj)
      .def("dBh_dxj_dxj", &ClassName::dBh_dxj_dxj)
      .def("dF_dx", &ClassName::dF_dx)
      .def("dF0_dx", &ClassName::dF0_dx)
      .def("dh_dx", &ClassName::dh_dx)
      .def("dLLi_dxdx", &ClassName::dLLi_dxdx)
      .def("dr_dx", static_cast<np_array (ClassName::*)(const std::vector<np_array>& x, const std::vector<np_array>& u, const std::vector<np_array>& lamda, const std::vector<np_array>& mu, const std::vector<np_array>& h_plus_mask)>(&ClassName::dr_dx) )
      .def("dr_du", static_cast<np_array (ClassName::*)(const std::vector<np_array>& x, const std::vector<np_array>& u, const std::vector<np_array>& lamda, const std::vector<np_array>& mu, const std::vector<np_array>& h_plus_mask)>(&ClassName::dr_du) )
      .def("dr_dlamda", static_cast<np_array (ClassName::*)(const std::vector<np_array>& x, const std::vector<np_array>& u, const std::vector<np_array>& lamda, const std::vector<np_array>& mu, const std::vector<np_array>& h_plus_mask)>(&ClassName::dr_dlamda) )
      .def("dr_dmu", static_cast<np_array (ClassName::*)(const std::vector<np_array>& x, const std::vector<np_array>& u, const std::vector<np_array>& lamda, const std::vector<np_array>& mu, const std::vector<np_array>& h_plus_mask)>(&ClassName::dr_dmu) )
      .def("r", &ClassName::r)
      .def("dr_dy", &ClassName::dr_dy)
      .def("getHplusMask", &ClassName::getHplusMask)
      .def("step", &ClassName::step)
      

      .def("summary", &ClassName::summary)
      .def("SparseQR", &ClassName::SparseQR)
      .def("LeastSquaresConjugateGradient", &ClassName::LeastSquaresConjugateGradient)
      .def("print_dim", &ClassName::print_dim)
      .def("test_bool_array",&ClassName::test_bool_array)
      .def("three_dim", &ClassName::three_dim)
      .def("pass_by_ref", &ClassName::pass_by_ref)
      ;
}
