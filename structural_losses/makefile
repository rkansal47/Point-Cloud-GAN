#nvcc = /usr/local/cuda-8.0/bin/nvcc
cudalib =  "/usr/bin/cuda/lib64"
tensorflow = "/usr/local/lib/python3.6/dist-packages/tensorflow/include"
nsync = "/usr/local/lib/python3.6/dist-packages/tensorflow/include/external/nsync/public"
TF_LIB = "/usr/local/lib/python3.6/dist-packages/tensorflow"

#orflowll: tf_approxmatch_so.so tf_approxmatch_g.cu.o tf_nndistance_so.so tf_nndistance_g.cu.o

all: tf_approxmatch_so.so tf_nndistance_so.so tf_nndistance_g.cu.o

tf_approxmatch_so.so: tf_approxmatch_g.cu.o tf_approxmatch.cpp
	g++ -std=c++11 -I$(tensorflow) -I$(nsync) tf_approxmatch.cpp tf_approxmatch_g.cu.o -o tf_approxmatch_so.so -shared -fPIC -lcudart -L$(TF_LIB) -ltensorflow_framework -L$(cudalib) -O2 -D_GLIBCXX_USE_CXX11_ABI=0


tf_approxmatch_g.cu.o: tf_approxmatch_g.cu
	nvcc -std=c++11 -I$(tensorflow) -I$(nsync) -c -o tf_approxmatch_g.cu.o tf_approxmatch_g.cu -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2 -D_GLIBCXX_USE_CXX11_ABI=0

tf_nndistance_so.so: tf_nndistance_g.cu.o tf_nndistance.cpp
	g++ -std=c++11 -I$(tensorflow) -I$(nsync)  tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -lcudart -L$(TF_LIB) -ltensorflow_framework -L $(cudalib) -O2 -D_GLIBCXX_USE_CXX11_ABI=0


tf_nndistance_g.cu.o:
	echo $(TF_LIB)
	nvcc -std=c++11 -I$(tensorflow) -I$(nsync) -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu -L$(TF_LIB) -ltensorflow_framework -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2 -D_GLIBCXX_USE_CXX11_ABI=0

clean:
	rm -rf tf_approxmatch_so.so
	rm -rf tf_nndistance_so.so
	rm -rf  *.cu.o
