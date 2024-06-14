cmake_minimum_required(VERSION 2.8.12)
project(residual_game)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG -O2")

find_package( PkgConfig )
pkg_check_modules( EIGEN3 REQUIRED eigen3 )
include_directories( ${EIGEN3_INCLUDE_DIRS} )

add_subdirectory(pybind11)
#pybind11_add_module(residual_game residual_game.cpp)
pybind11_add_module(unstructured_driving unstructured_driving.cpp)
