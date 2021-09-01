#!/usr/bin/perl

# Loop over the frames
foreach $f (0..256) {

	# Create the filename
	$fn=sprintf "lloyd_output/fr_%04d.png",$f;

	# Create a temporary gnuplot file to render the frame 
	open A,"lloyd_movie.gnuplot";
	open B,">temp.gnuplot";
	while(<A>) {
		s/FILENAME/$fn/;
		s/N/$f/g;
		print B;
	}
	close A;
	close B;

	# Render the frame using Gnuplot
	`gnuplot temp.gnuplot`;
}

# Remove temporary file
unlink "temp.gnuplot";
