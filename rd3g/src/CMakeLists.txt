cmake_minimum_required(VERSION 3.8)
cmake_policy(SET CMP0069 NEW) # suppress warning, also compiler optimization
# set(CMAKE_CXX_STANDARD 11)

# use std::string instead of std::__cxx11::basic_string to be compatible with pre-built casadi bin
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

project(residual_game)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
# Normal debug-friendly build
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -fPIC")
# For release when we want performance
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG -O3 -Ofast -fPIC -fno-math-errno -ffast-math -march=native -mfma")

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
message("CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

include_directories(${PROJECT_SOURCE_DIR}/solvers)
include_directories(${PROJECT_SOURCE_DIR})

find_package( PkgConfig )
pkg_check_modules( EIGEN3 REQUIRED eigen3=3.4.0 )
include_directories( ${EIGEN3_INCLUDE_DIRS} )
message("eigen3 include directory: ${EIGEN3_INCLUDE_DIRS}")

# Find CasADi manually, since casadi is installed with pip
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

# Get spdlog for logging
include(FetchContent)
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG        v1.12.0)
FetchContent_MakeAvailable(spdlog)
find_package(spdlog REQUIRED)


# ---- pybind targets, for games with implementation in c++ directly ----
add_subdirectory(pybind11)
pybind11_add_module(unstructured_driving games/unstructured_driving.cpp)
pybind11_add_module(car_merge_kinematic_bicycle games/car_merge_kinematic_bicycle.cpp)
pybind11_add_module(car_racing games/car_racing.cpp)
pybind11_add_module(unicycle games/unicycle.cpp)

# --- CasADi compatible games ---
# add_library(car_merge_kinematic_bicycle_casadi_N3_T20 SHARED
#            games/casadi_codegen/car_merge_kinematic_bicycle_casadi_N3_T20.cpp)
file(GLOB CASADI_SOURCES "games/casadi_codegen/*.cpp")

foreach(SOURCE_FILE ${CASADI_SOURCES})
    # Extract the filename without extension
    get_filename_component(LIB_NAME ${SOURCE_FILE} NAME_WE)
    add_library(${LIB_NAME} SHARED ${SOURCE_FILE})
endforeach()

# --- CasADi - Eigen example ----
add_executable(casadi_eigen examples/casadi_eigen.cpp)
target_include_directories(casadi_eigen PUBLIC ${CASADI_INCLUDE_DIR})
target_link_libraries(casadi_eigen ${CASADI_LIB})

# --- CasADi - RD3G ----
pybind11_add_module(rd3g_casadi solvers/rd3g_casadi.cpp)
target_include_directories(rd3g_casadi PUBLIC ${CASADI_INCLUDE_DIR})
target_link_libraries(rd3g_casadi PRIVATE ${CASADI_LIB})
target_link_libraries(rd3g_casadi PRIVATE spdlog::spdlog)