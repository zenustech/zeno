x=main

o: y x

x: $x.exe Makefile
	./$<
	
d: $x.exe Makefile
	gdb ./$< -ex r

%.exe: %.cpp Makefile
	g++ -I../../../include -std=c++17 -ggdb -gstabs+ -O3 -o $@ $<

y:
	cat test.asm | ./a.sh
