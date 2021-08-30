set term pdf
set xlabel 'x'
set ylabel 'y'
set zlabel 'z'
set view 60,60
set size square
set xtics offset -0.2,-0.2
set ytics offset 0.2,-0.2
set output 'cp_test.pdf'
splot 'cp_test_v.gnu' w l t 'Voronoi cells', 'cp_test_p.gnu' u 2:3:4 w p lt 4 t 'Particles', 'cp_test_d.gnu' w l t 'Domain'
set output
