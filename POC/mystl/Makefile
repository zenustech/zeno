x=mystl

x: $x.exe Makefile
	./$<
	
d: $x.exe Makefile
	gdb ./$< -ex r

%.exe: %.cpp Makefile
	g++ -std=c++17 -ggdb -gstabs+ -O0 -o $@ $<
	#g++ -std=c++17 -fopenmp -march=native -O3 -o $@ $<
	#clang++ -std=c++17 -march=native -O3 -o $@ $<
