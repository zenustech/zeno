/* Copyright (c) 2015-2018 The Khronos Group Inc.

   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and/or associated documentation files (the
   "Materials"), to deal in the Materials without restriction, including
   without limitation the rights to use, copy, modify, merge, publish,
   distribute, sublicense, and/or sell copies of the Materials, and to
   permit persons to whom the Materials are furnished to do so, subject to
   the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Materials.

   MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS
   KHRONOS STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS
   SPECIFICATIONS AND HEADER INFORMATION ARE LOCATED AT
    https://www.khronos.org/registry/

  THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
  CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
  MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.

*/

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>
#include <numeric>

#include <experimental/algorithm>
#include <sycl/execution_policy>

#include "benchmark.h"

using namespace sycl::helpers;

/** getRand
 * @brief This function returns a random float number
 */
float getRand() { return 1e18 * std::exp(-1.8) * (.5 - std::rand()); }

/* Body.
* The Body represents a particle in a three dimensinal space with a mass
*/
class Body {
  cl::sycl::cl_float3 pos;  // position components
  cl::sycl::cl_float3 vel;  // velocity components
  cl::sycl::cl_float3 acc;  // force components
  float mass;               // mass of the particle

 public:
  /** generateBody
   * @brief Function that fills the attributes of a Body
   */
  void generateBody() {
    pos.x() = getRand();
    pos.y() = getRand();
    pos.z() = getRand();
    vel.x() = getRand();
    vel.y() = getRand();
    vel.z() = getRand();
    acc.x() = getRand();
    acc.y() = getRand();
    acc.z() = getRand();
    mass = 1.0f;
  }

  /** update
   * @brief Function that updates the position and velocity of each body
   */
  void update() {
    float dthf = 0.25;
    float dtime = 0.01;

    float dvelx = this->acc.x() * dthf;
    float dvely = this->acc.y() * dthf;
    float dvelz = this->acc.z() * dthf;

    float velhx = this->vel.x() + dvelx;
    float velhy = this->vel.y() + dvely;
    float velhz = this->vel.z() + dvelz;

    this->pos.x() = this->pos.x() + velhx * dtime;
    this->pos.y() = this->pos.y() + velhy * dtime;
    this->pos.z() = this->pos.z() + velhz * dtime;

    this->vel.x() = velhx + dvelx;
    this->vel.y() = velhy + dvely;
    this->vel.z() = velhz + dvelz;
  }

  /** addForce
   * @brief Function that adds the force that b applies over the body
   * @param b : The acting body
   */
  void addForce(Body b) {
    float EPS = 0.025;  // softening parameter (just to avoid infinities)
    float drx = b.pos.x() - this->pos.x();
    float dry = b.pos.y() - this->pos.y();
    float drz = b.pos.y() - this->pos.y();

    float drsq = drx * drx + dry * dry + drz * drz + EPS;
    float idr = 1.0f / cl::sycl::sqrt(drsq);
    float nphi = b.mass * idr;
    float scale = nphi * idr * idr;

    this->acc.x() = this->acc.x() + drx * scale;
    this->acc.y() = this->acc.y() + dry * scale;
    this->acc.z() = this->acc.z() + drz * scale;
  }

  /** printToFile
   * @brief Function that prints the Body attributes to an ostream
   * @param os : The output stream
   */
  void printToFile(std::ostream& os) {
    os << pos.x() << " " << pos.y() << " " << pos.z() << " ";
    os << vel.x() << " " << vel.y() << " " << vel.z() << " ";
    os << acc.x() << " " << acc.y() << " " << acc.z() << " ";
    os << mass << std::endl;
  }
};

/** benchmark_nbody
 * @brief Body Function that executes the SYCL CG of NBODY
 */
benchmark<>::time_units_t benchmark_nbody(const unsigned numReps,
                                          const unsigned N,
                                          const cli_device_selector cds) {
  srand(time(NULL));
  std::vector<Body> bodies(N);

  // randomly generating N Particles
  for (size_t i = 0; i < N; i++) {
    auto& b = bodies[i];
    b.generateBody();
  }
  auto mainLoop = [&]() {
    auto d_bodies = sycl::helpers::make_buffer(begin(bodies), end(bodies));
    cl::sycl::queue q(cds);
    sycl::sycl_execution_policy<class UpdateAlgorithm> snp2(q);
    // Main loop
    auto vectorSize = d_bodies.get_count();
    const auto ndRange = snp2.calculateNdRange(vectorSize);
    auto f = [vectorSize, ndRange, &d_bodies](cl::sycl::handler& h) mutable {
      auto a_bodies =
          d_bodies.get_access<cl::sycl::access::mode::read_write>(h);
      h.parallel_for<class NBodyAlgorithm>(
          ndRange, [a_bodies, vectorSize](cl::sycl::nd_item<1> id) {
            if (id.get_global_id(0) < vectorSize) {
              for (size_t i = 0; i < vectorSize; i++) {
                a_bodies[id.get_global_id(0)].addForce(a_bodies[i]);
              }
            }
          });
    };
    q.submit(f);

    // Update loop
    std::experimental::parallel::for_each(snp2, begin(d_bodies), end(d_bodies),
                                          [=](Body& body) {
                                            body.update();
                                            return body;
                                          });  // main loop

    q.wait_and_throw();
  };

  auto time = benchmark<>::duration(numReps, mainLoop);

  std::fstream file;
  std::stringstream fnss;
  fnss << "bodies_output_" << N << ".txt";
  file.open(fnss.str(), std::fstream::out);
  for(auto b:bodies){
    b.printToFile(file);
  }
  file.close();

  return time;
}

BENCHMARK_MAIN("BENCH_NBODY", benchmark_nbody, 2, 65536, 1);
