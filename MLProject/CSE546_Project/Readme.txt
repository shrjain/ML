Please find the following files in this archive:

runCCLorma.m
This file contains the our Matlab implementation of the algorithm Context based LLORMA. It assumes the parsed data files "ratings10M.mat", "ratings.mat", genres.mat and genres10M.mat are already present in the same folder. Since we were asked not to upload the data, we are not uploading these parsed data mat files.  

runScript.m
This script calls the "runCCLorma" function with various parameters to run all the experiments. 

iterate_LLORMA.cpp
This is our base algorithm implementation in C++ so that it runs faster. The function in this file is called by runCCLorma.m.