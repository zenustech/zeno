default: all debug

all: core
	make -C Projects

demo: core
	make -C demo_project

core:
	cmake -B build
	make -C build -j `python -c 'from multiprocessing import cpu_count; print(cpu_count() * 2)'`

run: core
	./run.sh

install:
	python/setup.py install

dist:
	python/setup.py bdist_wheel

debug: all
	USE_GDB=1 ZEN_SPROC=1 ZEN_OPEN=assets/rigid.zsg ./run.sh
