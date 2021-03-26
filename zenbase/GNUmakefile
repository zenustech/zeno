x:
	make -C ../zen
	cmake -B build
	make -j 4 -C build
	PYTHONPATH=../zen python test.py
