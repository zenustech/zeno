x:
	cmake -B build
	cmake --build build --parallel
	build/openfdb

d:
	cmake -B build -DCMAKE_BUILD_TYPE=Debug
	cmake --build build --parallel
	gdb build/openfdb -ex r
