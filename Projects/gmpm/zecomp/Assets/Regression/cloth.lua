-- mpm 3f
math.randomseed(123)

output = "output/cloth"
end_frame = 10
dx = 0.2
cloth_dx = 0.2
gravity = TV.create({0, -3,0})
max_dt = 0.001
apic_rpic_ratio = 0.2
cfl = 0.3
element_partitions = 16

quasistatic = false
symplectic = true
matrix_free = true
verbose = false
run_diff_test = false
write_meshes = false
write_partio = false

use_bending = false
bending_Youngs = 6e-4
Youngs = 400
nu = 0.3
rho = 2
cloth_k = 800
cloth_gamma = 0
cloth_friction_angle = 0
cloth_damage = 999999

function initialize(frame)

   create_cloth(TV.create({0, 0,0}))

   -- pinner
   local sphere_center = TV.create({-1, 0.5,-0.5})
   local sphere = Sphere.new(sphere_center, 0.1)
   local sphere_object = AnalyticCollisionObject.new(sphere, STICKY)
   mpm:addAnalyticCollisionObject(sphere_object)

   local sphere_center = TV.create({-1, 0.5,0.5})
   local sphere = Sphere.new(sphere_center, 0.1)
   local sphere_object = AnalyticCollisionObject.new(sphere, STICKY)
   mpm:addAnalyticCollisionObject(sphere_object)

   -- bullet
   local sphere_center = TV.create({-0.8, 0 ,0})
   local sphere = Sphere.new(sphere_center, 0.25)
   local sphere_object = AnalyticCollisionObject.new(sphere, SEPARATE)
   mpm:addAnalyticCollisionObject(sphere_object)

   -- bullet2
   local sphere_center = TV.create({2.5, 0 ,0})
   local sphere = Sphere.new(sphere_center, 0.25)
   local sphere_object = AnalyticCollisionObject.new(sphere, SEPARATE)
   mpm:addAnalyticCollisionObject(sphere_object)

   local ground_origin = TV.create({0,0,0})
   local ground_normal = TV.create({0,1,0})
   local ground_ls = HalfSpace.new(ground_origin, ground_normal)
   local ground_object = AnalyticCollisionObject.new(ground_ls, SEPARATE)
   ground_object:setFriction(0.1)
   mpm:addAnalyticCollisionObject(ground_object)
end

function create_cloth(t)
   local transform = t;
   function cloth(index, mass, X, V)
      X[1], X[2] = -X[2], X[1]
      X[0] = X[0]  - 1
      X[1] = X[1] + 0.5
      X[2] = X[2] - 0.5
      X:set(X + transform)
   end

   local mesh = scene:createTriMesh(constructUnitSquare)
   local deformable = scene:addDeformableTriMesh(mesh)
   deformable:transform(cloth)

   local meshed_particles_handle = mpm:makeParticleHandle(deformable, 1)
   local thickness = 0.001
   local quadrature_count = 1
   local quadrature_particles_handle = mpm:sampleTriParticles(deformable, thickness, rho, quadrature_count)
   local m = QRCloth3D.new(Youngs,nu, cloth_k, cloth_gamma)
   quadrature_particles_handle:addFBasedMpmForce(m)
   local py = ClothYield.new(cloth_friction_angle, cloth_damage)
   quadrature_particles_handle:addPlasticity(m,py,"cotangent")

   if(use_bending==true) then
      local bending_mesh = mpm:addBendingSpringsForTriangles(deformable)
      local bending_deformable = scene:addDeformableSegMesh(bending_mesh)
      local m_bending = CorotatedCodimensional1.new(bending_Youngs,0.3)
      bending_deformable:addFemHyperelasticForce(m_bending)
   end

end

function constructUnitSquare(mesh, X)
   local z = TV2.create({0, 0})
   local cells = IntV2.create({2/cloth_dx, 1/cloth_dx})
   local grid = Grid2d.new(cells, cloth_dx , z);
   construct2dMattressMesh(grid, mesh, X)
end
