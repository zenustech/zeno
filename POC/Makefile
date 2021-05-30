x: mg.exe
	@#./$<
	gdb ./$< -ex r

%.exe: %.cpp
	g++ -ggdb -gstabs+ -O0 -o $@ $<
	@#g++ -fopenmp -march=native -O3 -o $@ $<
