#include <ImmersedBoundaryMethod.h>
#include <mshr.h>
#include <AlgebraSolver/NewtonSolver.h>
#include <AlgebraSolver/StdVector.h>

using VectorType = StdVector<double, double3>;

int main(){
    
    loguru::add_file("test_ibm_everything.log", loguru::Append, loguru::Verbosity_MAX);
	loguru::add_file("test_ibm_warning.log", loguru::Append, loguru::Verbosity_WARNING);
    loguru::g_stderr_verbosity = loguru::Verbosity_FATAL;

    std::cout << "Create solid mesh.\n";
    auto domain = std::make_shared<mshr::Sphere>(dolfin::Point(0.6,0.5,0.5), 0.2);
    auto solid_mesh = std::make_shared<ImmersedMesh>(mshr::generate_mesh(domain, 10));

    std::cout << "Create background mesh.\n";
    int3 dim = {16,16,16};
    auto fluid_mesh = std::make_shared<BackgroundMesh>(dim);

    // ImmersedBoundaryMethod ibm(solid_mesh, fluid_mesh);
    // ibm.forward_euler_scheme();


    auto ibm_problem = std::make_shared<ImmersedBoundaryMethod<VectorType>>(solid_mesh, fluid_mesh);

    // ibm_problem->forward_euler_scheme();

    auto bicgstab = std::make_shared<BiCGSTAB<VectorType>>(ibm_problem->unkown_size()/3);
    NewtonSolver<VectorType> ns(bicgstab);

    // TODO : param b is useless here.
    std::vector<double3> x0(ibm_problem->unkown_size()/3);
    std::vector<double3> b(ibm_problem->unkown_size()/3);

    VectorType xx0;
    VectorType bb;


    
    // set the initial value of x0;
    x0 = ibm_problem->get_solid_positions();
    auto xn_1 = x0; // x_{n-1}

    
    for (size_t i = 0; i < 100; i++) 
    {
        Timer timer("nonlinear test.\n");
        // x0 = 2*x0 - xn_1;
        // initial guess for x0;
        for (size_t i = 0; i < x0.size(); i++)
        {
            double3 temp = {2.0*x0[i].x - xn_1[i].x, 2.0*x0[i].y - xn_1[i].y,2.0*x0[i].z - xn_1[i].z};
            xn_1[i] = x0[i];
            x0[i] = temp;
        }
        xx0.set(x0);
        bb.set(b);
        auto nonlinear_result = ns.Solve(ibm_problem, xx0, bb);
        xx0.get(x0);
        ibm_problem->advance(x0);
        ibm_problem->record(i);
        LOG_F(WARNING, "xn: %.8lf, %.8lf", x0[0].x, x0[1].y);
        LOG_F(WARNING, "Noninear solver successful? %d", nonlinear_result.first);
        LOG_F(WARNING, "residual : %lf, iter : %d", nonlinear_result.second.first, nonlinear_result.second.second);
    }


    return 0;
}