x=adl

x: $x.exe Makefile
	./$<
	
d: $x.exe Makefile
	gdb ./$< -ex r

%.exe: %.cpp Makefile
	g++ -I../include -I../zeno/include -std=c++17 -ggdb -gstabs+ -O0 -o $@ $<
	#g++ -I../include -I../zeno/include -std=c++17 -fopenmp -march=native -O3 -o $@ $<
