## Getting Started
You need to install cmake, cpp, and Eigen before running this program. 
I recommend you try some simple example by yourself to ensure they are properly installed. 

You also need `matplotlilb > 3.9`, `distutils` and maybe some other stuffs. Read error message carefully. 

To build and run the project, first clone the repository, then do the following .

* `git submodule update --init ` in the root directory of the repository. this will download pybind11
* from the root, `mkdir gifs`. This is where the gifs will be stored. You have to manually create it because git does not manage the local gifs, therefore it does not create this directory
* The code can run with just python, but you may need to remove some import initiatives for the cpp modules that the code expects to find. What we did is write everything in python, then rewrite identical functions in cpp, and use that instead for performance. To use the cpp version, set **class** attribute `USE_CPP = True`
* `cd src`, then `mkdir build`, `cd build`. This is where you will build the cpp code and create the cython library. 
* build the cpp libraries, from `build/`, do `cmake ..`, then `cmake --build`
* if everything goes well you will see a `cython-xxxx.so` in `build/`
* go back to root, and run `python3 CarMergeKinematicBicycle.py` You should see some plots and an animation. 


