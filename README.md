## Getting Started
This project makes heavy use of casadi generated C source code for performance.
You need to install cmake, gcc, and Eigen to run the projec with the C++ extensions. 
However, pure python implementation is also available, though at significantly lower speed performance.

## Dependencies
# Ubuntu / Debian Installation
`sudo apt-get install libsuitesparse-dev libsuperlu-dev`

To build and run the project, first clone the repository, then do the following .

* `git submodule update --init ` in the root directory of the repository. this will download pybind11
* from the root, `mkdir gifs`. This is where the gifs will be stored. You have to manually create it because git does not manage the local gifs, therefore it does not create this directory
* This project is set up to use `uv` for dependency management, run `uv sync` to load the dependencies.
* The code can run with just python, To use the python version, set `USE_CPP = False`

To build the cpp code
* Generate code with casadi, run `uv run scripts/generate_casadi_code.py`, the generated code is under `/casadi_codegen`
* From the root directory `mkdir build`, `cd build`. This is where you will build the cpp code and create the cython library. 
* build the cpp libraries, from `/build/`, do `uv run cmake ../cpp`, then `uv run cmake --build`
* if everything goes well you will see a `cython-xxxx.so` in `/build/lib/`
* go back to root, and run `uv run scripts/solve_merge.py` You should see some plots and an animation.

* If running from WSL, make sure you have external window client running such as Xming to see the visualization. Then export the display to whatever server is running the window (e.g. export DISPLAY=0:0)

## LQGame

This repository also contains a car merging scenario under the kinematic bicycle dynamic model using the LQGame algorithm for optimization (see [this paper](https://arxiv.org/abs/1909.04694)). To run the merging scenario using LQGame, use `python cmkb_LQGame.py`. Settings such as car count and starting position can be changed in that file; most relevant settings are labeled. Most of the actual logic is contained in `LQGame.py`. The current implementation supports good car merging behavior for up to 4 cars. From 5+ there is no convergence at the moment. It's possible that higher car count will need a more favorable starting position in order to converge.
