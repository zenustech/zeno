
  velocity->setTree(std::make_shared<openvdb::Vec3fTree>(
      particles->tree(), openvdb::Vec3f{0}, openvdb::TopologyCopy()));
  openvdb::tools::dilateActiveValues(
      velocity->tree(), 1,
      openvdb::tools::NearestNeighbors::NN_FACE_EDGE_VERTEX);

  velocity_weights = velocity->deepCopy();

  auto voxel_center_transform =
      openvdb::math::Transform::createLinearTransform(dx);
  liquid_sdf->setTransform(voxel_center_transform);
  liquid_sdf->setTree(std::make_shared<openvdb::FloatTree>(
      velocity->tree(), 0.6f * dx * 1.01, openvdb::TopologyCopy()));

  auto collector_op{p2g_collector(liquid_sdf, velocity, velocity_weights,
                                  particles, particle_radius)};

  auto vleafman =
      openvdb::tree::LeafManager<openvdb::Vec3fTree>(velocity->tree());

  vleafman.foreach (collector_op, true);
