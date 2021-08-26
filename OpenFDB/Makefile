x:
	cmake -B build
	cmake --build build --parallel
	-rm -rf /tmp/a.obj
	build/openfdb | tee /tmp/a.obj

d:
	cmake -B build
	cmake --build build --parallel
	gdb build/openfdb -ex r
