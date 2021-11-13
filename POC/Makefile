x=rangeview
#L=valgrind --tool=memcheck --leak-check=full
#E=source /opt/intel/oneapi/setvars.sh &&

x: $x.exe Makefile
	$E $L ./$<

d: $x.exe Makefile
	$E gdb ./$< -ex r

%.exe: %.cpp Makefile
	g++ -std=c++20 -ggdb -gstabs+ -O0 -o $@ $<
	#$E dpcpp -I../include -I../zeno/include -std=c++20 -O0 -o $@ $<
	#g++ -I../include -I../zeno/include -std=c++20 -ggdb -gstabs+ -O0 -o $@ $<
	#g++ -I../include -I../zeno/include -std=c++20 -fopenmp -march=native -O3 -o $@ $<
