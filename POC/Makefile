x: prim.exe
	./$<
	@#gdb ./$< -ex r

%.exe: %.cpp
	@#g++ -ggdb -gstabs+ -O0 -o $@ $<
	g++ -O1 -o $@ $<
