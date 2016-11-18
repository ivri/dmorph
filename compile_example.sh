PATH_TO_CNN=/path/to/dynet
PATH_TO_EIGEN=/path/to/cnn/eigen
PATH_TO_MKL=/usr/lib/intel/mkl
PATH_TO_CNN_GPU=/path/to/dynet_gpu/dynet

#g++ -g -o dmorph_dnet deriv.cc -I/$PATH_TO_CNN -I/$PATH_TO_EIGEN -std=c++11 -L/usr/lib -L/$PATH_TO_CNN/build/dynet -ldynet -lboost_program_options -lboost_serialization -lboost_system -lboost_filesystem -Wno-narrowing -w

#GPU
g++ -g -o dmorph_gpu_dynet deriv.cc  -I/$PATH_TO_CNN_GPU -I/usr/local/cuda-7.0/include -I/$PATH_TO_EIGEN -std=c++11 -L/usr/lib -L/$PATH_TO_CNN_GPU/build/dynet -lgdynet -ldynetcuda -lboost_program_options -lboost_serialization -lboost_system -lboost_filesystem -DHAVE_CUDA -L/usr/local/cuda-7.0/targets/x86_64-linux/lib -lcudart -lcublas -Wno-narrowing -w


