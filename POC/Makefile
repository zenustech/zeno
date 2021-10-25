x=vec

x: $x.exe Makefile
	./$<
	
d: $x.exe Makefile
	gdb ./$< -ex r

%.exe: %.cpp Makefile
	#g++ -std=c++20 -ggdb -gstabs+ -O0 -o $@ $<
	g++ -I../include -I../zeno/include -std=c++20 -ggdb -gstabs+ -O0 -o $@ $<
	#g++ -I../include -I../zeno/include -std=c++20 -fopenmp -march=native -O3 -o $@ $<
