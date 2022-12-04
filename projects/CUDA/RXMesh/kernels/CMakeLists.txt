cmake_minimum_required(VERSION 3.18)

target_sources(zeno PRIVATE
  for_each.cuh
  loader.cuh
  query_dispatcher.cuh
  rxmesh_queries.cuh
)