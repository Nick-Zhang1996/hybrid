cmake_minimum_required(VERSION 3.5)
cmake_policy(SET CMP0069 NEW)
set(CMAKE_CXX_STANDARD 11)

# use std::string instead of std::__cxx11::basic_string to be compatible with pre-built casadi bin
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

project(residual_game)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG -O2")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG -O3 -Ofast -fno-math-errno -ffast-math -march=native -mfma")
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
message("CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

find_package( PkgConfig )
pkg_check_modules( EIGEN3 REQUIRED eigen3=3.4.0 )
include_directories( ${EIGEN3_INCLUDE_DIRS} )
message("eigen3 include directory: ${EIGEN3_INCLUDE_DIRS}")

# Find CasADi manually
execute_process(
    COMMAND python3 -c "import casadi; print(casadi.__path__[0])"
    OUTPUT_VARIABLE CASADI_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "Found CasADi via Python at: ${CASADI_ROOT}")

# Headers are usually in site-packages/casadi/include
set(CASADI_INCLUDE_DIR "${CASADI_ROOT}/include")

# The library (.so) is usually directly in site-packages/casadi
# We search for it explicitly to be safe
find_library(CASADI_LIB 
    NAMES casadi 
    HINTS "${CASADI_ROOT}" 
    NO_DEFAULT_PATH
)

if(NOT CASADI_LIB)
    message(FATAL_ERROR "Could not find libcasadi.so in ${CASADI_ROOT}. The pip package might be missing shared libraries.")
endif()

# ---- pybind targets ----
add_subdirectory(pybind11)
pybind11_add_module(unstructured_driving unstructured_driving.cpp)
pybind11_add_module(car_merge_kinematic_bicycle car_merge_kinematic_bicycle.cpp)
pybind11_add_module(car_racing car_racing.cpp)
pybind11_add_module(unicycle unicycle.cpp)

# --- CasADi toy exampl ----
add_library(gen_c SHARED gen.c)

add_executable(main main.cpp)
target_include_directories(main PUBLIC ${CASADI_INCLUDE_DIR})
target_link_libraries(main ${CASADI_LIB})