x: mg.exe
	./$<
	#gdb ./$< -ex r

%.exe: %.cpp
	#g++ -std=c++17 -ggdb -gstabs+ -O0 -o $@ $<
	g++ -std=c++17 -fopenmp -march=native -O3 -o $@ $<
	#clang++ -std=c++17 -march=native -O3 -o $@ $<
