output = "output/mattress"
end_frame = 10
E = 100
rho = 1
nu = 0.3
m = 10
gravity = TV.create({0, -9.8, 0})
max_dt = 0.004
newton_iterations = 100

function initialize(frame)
    local mesh = scene:createTetMesh(constructUnitBox)
    local deformable = scene:addDeformableTetMesh(mesh)
    deformable:setMassFromDensity(rho)
    local m = Corotated.new(E,nu)
    deformable:addFemHyperelasticForce(m)
    scene:setBoundaryConditions(left_side)
end

function constructUnitBox(mesh, X)
    local z = TV.create({0, 0, 0})

    local cells = IntV.create({m, m, m})

    local grid = Grid3d.new(cells, 1/m, z);
    construct3dMattressMesh(grid, mesh, X)
end

function left_side(dof, X)
    return X[0] < 1e-9
end
