x:
	cmake -B build
	cmake --build build --parallel
	build/sycltest | tee /tmp/a.txt

d:
	cmake -B build
	cmake --build build --parallel
	gdb build/sycltest -ex r
