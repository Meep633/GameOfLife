all: build build-mpi build-basic

build: conway.c conway.cu
	mpixlc -O3 conway.c -c -o conway-xlc.o
	nvcc -O3 -arch=sm_70 conway.cu -c -o conway-nvcc.o
	mpixlc -O3 conway-xlc.o conway-nvcc.o -o conway -L/usr/local/cuda-11.2/lib64/ -lcudadevrt -lcudart -lstdc++

build-mpi: conway-mpi.c
	mpixlc -O3 conway-mpi.c -o conway-mpi -lm

build-basic: conway-basic.c
	xlc -O3 conway-basic.c -o conway-basic

debug: conway.c conway.cu
	mpixlc -g -O0 conway.c -c -o conway-xlc.o
	nvcc -g -G -O0 -arch=sm_70 conway.cu -c -o conway-nvcc.o
	mpixlc -g -O0 conway-xlc.o conway-nvcc.o -o conway-debug -L/usr/local/cuda-11.2/lib64/ -lcudadevrt -lcudart -lstdc++

clean:
	rm -f *.o conway conway-mpi conway-basic conway-debug