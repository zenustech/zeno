x:
	cmake -B build -DComputeCpp_DIR=/opt/ComputeCpp-CE
	cmake --build build --parallel
	build/sycltest

d:
	cmake -B build -DComputeCpp_DIR=/opt/ComputeCpp-CE
	cmake --build build --parallel
	gdb build/sycltest -ex r
