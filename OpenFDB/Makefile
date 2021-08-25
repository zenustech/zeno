x:
	cmake -B build
	cmake --build build --parallel
	build/openfdb | tee /tmp/a.txt

d:
	cmake -B build
	cmake --build build --parallel
	gdb build/openfdb -ex r
