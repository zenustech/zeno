#!/usr/bin/perl

foreach $f (0..200) {
	open A,"rad_test.pov";
	open B,">temp.pov";

	while(<A>) {
		s/FRAME/rad_test_out\/fr$f.pov/;
		print B;
	}
	
	close A;
	close B;
	system "povray +H600 +W800 +A0.01 -J +R5 +Ofr_$f.png temp.pov";
}
