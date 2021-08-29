a:
	cmake -B build
	cmake --build build --parallel
	build/openfdb > a.txt
	cd .. && python a.py

x:
	cmake -B build
	cmake --build build --parallel
	build/openfdb

d:
	cmake -B build
	cmake --build build --parallel
	gdb build/openfdb -ex r
