x: b
	build/sycltest

d: b
	gdb build/sycltest -ex r

b:
	clear
	cmake -B build -DComputeCpp_DIR=/opt/ComputeCpp-CE -DCOMPUTECPP_BITCODE=ptx64
	cmake --build build --parallel
