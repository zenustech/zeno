#! /bin/csh

set ofile = install.out			# output file

echo '---- SINGLE PRECISION' >! $ofile
./testsmach >> $ofile
echo '' >> $ofile
echo ---- DOUBLE PRECISION >> $ofile
./testdmach >> $ofile
echo '' >> $ofile
echo ---- TIMER >> $ofile
./testtimer >> $ofile


