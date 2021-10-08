run: all
	build/zeno.main

debug: all
	gdb build/zeno.main -ex r

test: all
	build/zeno.tests

all:
	cmake -Bbuild
	make -Cbuild -j12

config:
	cmake -Bbuild
	ccmake -Bbuild
