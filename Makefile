build: conway.c conway.cu
	mpixlc -O3 conway.c -c -o conway-xlc.o
	nvcc -O3 -arch=sm_70 conway.cu -c -o conway-nvcc.o 
	mpixlc -O3 conway-xlc.o conway-nvcc.o -o conway -L/usr/local/cuda-11.2/lib64/ -lcudadevrt -lcudart -lstdc++ 

debug: conway.c conway.cu
	mpixlc -g -O0 conway.c -c -o conway-xlc.o
	nvcc -g -G -O0 -arch=sm_70 conway.cu -c -o conway-nvcc.o
	mpixlc -g -O0 conway-xlc.o conway-nvcc.o -o conway-debug -L/usr/local/cuda-11.2/lib64/ -lcudadevrt -lcudart -lstdc++