cmake_minimum_required(VERSION 3.18)

target_sources(zeno PRIVATE
  patcher_kernel.cuh
  patcher.cu
  patcher.h
)