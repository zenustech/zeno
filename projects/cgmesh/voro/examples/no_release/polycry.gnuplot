set term pdf color solid
set pointsize 0.5
set output 'polycry.pdf'
e=1e-9
set key outside right
set xlabel 'x'
set ylabel 'y'
set zlabel 'z'
splot [-e:81+e] [-e:81+e] [-e:81+e] 'lammps_input' u 2:3:4 w p pt 7 t 'Particles', 'lammps_cells' lt 4 t 'Voronoi cells', 'lammps_generators' u 2:3:4 w p lt 3 pt 9 t 'Generators'
set output
