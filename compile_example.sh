PATH_TO_CNN=../../clab/cnn
PATH_TO_EIGEN=../../clab/cnn/eigen



g++ -g -o dmorph deriv.cc -I/$PATH_TO_CNN -I/$PATH_TO_EIGEN  -std=c++11 -L/usr/lib -lcnn -lboost_program_options -lboost_serialization -lboost_system -lboost_filesystem -L/$PATH_TO_CNN/build/cnn   -Wno-narrowing -w

#GPU
g++ -g -o dmorph_gpu deriv.cc  -I/$PATH_TO_CNN -I/$PATH_TO_EIGEN  -I/usr/local/cuda-7.0/include -std=c++11 -L/usr/lib -lboost_program_options -lboost_serialization -lboost_system -lboost_filesystem -L/$PATH_TO_CNN/build/cnn -lcnn -lcnncuda -DHAVE_CUDA -L/usr/local/cuda-7.0/targets/x86_64-linux/lib -lcudart -lcublas -Wno-narrowing -w

