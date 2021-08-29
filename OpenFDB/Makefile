x:
	cmake -B build
	cmake --build build --parallel
	build/openfdb > a.txt
	python a.py

d:
	cmake -B build
	cmake --build build --parallel
	gdb build/openfdb -ex r

