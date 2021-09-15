x: b
	build/openfdb

d: b
	gdb build/openfdb -ex r

b:
	clear
	cmake -B build
	cmake --build build --parallel
