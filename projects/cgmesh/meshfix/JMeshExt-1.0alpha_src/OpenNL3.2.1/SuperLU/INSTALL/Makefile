include ../make.inc

all:  testsmach testdmach testtimer install.out #testdlamch testslamch

testdmach: dmach.o lsame.o dmachtst.o
	$(LOADER) $(LOADOPTS) -o testdmach dmach.o lsame.o dmachtst.o

testsmach: smach.o lsame.o smachtst.o
	$(LOADER) $(LOADOPTS) -o testsmach smach.o lsame.o smachtst.o

testdlamch: dlamch.o lsame.o dlamchtst.o
	$(LOADER) $(LOADOPTS) -o testdlamch dlamch.o lsame.o dlamchtst.o

testslamch: slamch.o lsame.o slamchtst.o
	$(LOADER) $(LOADOPTS) -o testslamch slamch.o lsame.o slamchtst.o

testtimer: superlu_timer.o timertst.o
	$(LOADER) $(LOADOPTS) -o testtimer superlu_timer.o timertst.o

install.out: install.csh
	@echo Testing machines parameters and timer 
	csh install.csh

slamch.o: slamch.c ; $(CC) -c $(NOOPTS) $<
dlamch.o: dlamch.c ; $(CC) -c $(NOOPTS) $<
superlu_timer.o: superlu_timer.c; $(CC) -c $(NOOPTS) $<

.c.o:
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f *.o test* *.out
