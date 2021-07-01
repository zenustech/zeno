all:
	cmake -Bbuild
	make -Cbuild

pack: all
	cd build && cpack

install: pack
	build/zensim-*.sh
