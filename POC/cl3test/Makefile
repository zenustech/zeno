x:
	clang++ -c -D__OPENCL__ -cl-std=clc++ -target spir -emit-llvm -Xclang -finclude-default-header main.cpp -o main.bc
	./bc_as_array.py main.bc > main.bc.inl
	clang++ main.cpp -lOpenCL -o main.exe
	./main.exe
