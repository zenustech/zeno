/*
 * MIT License
 *
 * Copyright (c) 2024 wuzhen
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * 1. The above copyright notice and this permission notice shall be included in
 *    all copies or substantial portions of the Software.
 *
 * 2. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *    SOFTWARE.
 */

#include "Context.h"

namespace nemo {
void DataStorage::cleanup() {
  if (!instance || !fnFree)
    return;
  fnFree(instance);
}

void DataStorage::init(MOD mod) {
  fnNew = get_fn<decltype(fnNew)>(mod, "dataNew");
  fnFree = get_fn<decltype(fnFree)>(mod, "dataFree");
  instance = fnNew();

  fnResize = get_fn<decltype(fnResize)>(mod, "dataResize");
  fnGlobalIdOffset = get_fn<decltype(fnGlobalIdOffset)>(mod, "dataGlobalIdOffset");
  fnTypeid = get_fn<decltype(fnTypeid)>(mod, "dataTypeid");

  fnGetBool = get_fn<decltype(fnGetBool)>(mod, "dataGetBool");
  fnGetFloat = get_fn<decltype(fnGetFloat)>(mod, "dataGetFloat");
  fnGetDouble = get_fn<decltype(fnGetDouble)>(mod, "dataGetDouble");
  fnGetInt = get_fn<decltype(fnGetInt)>(mod, "dataGetInt");
  fnGetVec3 = get_fn<decltype(fnGetVec3)>(mod, "dataGetVec3");
  fnGetDVec3 = get_fn<decltype(fnGetDVec3)>(mod, "dataGetDVec3");
  fnGetMat4 = get_fn<decltype(fnGetMat4)>(mod, "dataGetMat4");
  fnGetDMat4 = get_fn<decltype(fnGetDMat4)>(mod, "dataGetDMat4");
  fnGetMesh = get_fn<decltype(fnGetMesh)>(mod, "dataGetMesh");
  fnGetDMesh = get_fn<decltype(fnGetDMesh)>(mod, "dataGetDMesh");
  fnGetCuShape = get_fn<decltype(fnGetCuShape)>(mod, "dataGetCuShape");
  fnPullCuShape = get_fn<decltype(fnPullCuShape)>(mod, "dataPullCuShape");
  fnGetDCuShape = get_fn<decltype(fnGetDCuShape)>(mod, "dataGetDCuShape");
  fnPullDCuShape = get_fn<decltype(fnPullDCuShape)>(mod, "dataPullDCuShape");
  fnGetCurve = get_fn<decltype(fnGetCurve)>(mod, "dataGetCurve");
  fnGetDCurve = get_fn<decltype(fnGetDCurve)>(mod, "dataGetDCurve");
  fnGetSurface = get_fn<decltype(fnGetSurface)>(mod, "dataGetSurface");
  fnGetDSurface = get_fn<decltype(fnGetDSurface)>(mod, "dataGetDSurface");

  fnSetBool = get_fn<decltype(fnSetBool)>(mod, "dataSetBool");
  fnSetFloat = get_fn<decltype(fnSetFloat)>(mod, "dataSetFloat");
  fnSetDouble = get_fn<decltype(fnSetDouble)>(mod, "dataSetDouble");
  fnSetInt = get_fn<decltype(fnSetInt)>(mod, "dataSetInt");
  fnSetVec2 = get_fn<decltype(fnSetVec2)>(mod, "dataSetVec2");
  fnSetDVec2 = get_fn<decltype(fnSetDVec2)>(mod, "dataSetDVec2");
  fnSetVec3 = get_fn<decltype(fnSetVec3)>(mod, "dataSetVec3");
  fnSetDVec3 = get_fn<decltype(fnSetDVec3)>(mod, "dataSetDVec3");
  fnSetMat4 = get_fn<decltype(fnSetMat4)>(mod, "dataSetMat4");
  fnSetDMat4 = get_fn<decltype(fnSetDMat4)>(mod, "dataSetDMat4");
  fnSetMesh = get_fn<decltype(fnSetMesh)>(mod, "dataSetMesh");
  fnSetDMesh = get_fn<decltype(fnSetDMesh)>(mod, "dataSetDMesh");
  fnSetCurve = get_fn<decltype(fnSetCurve)>(mod, "dataSetCurve");
  fnSetDCurve = get_fn<decltype(fnSetDCurve)>(mod, "dataSetDCurve");
  fnSetSurface = get_fn<decltype(fnSetSurface)>(mod, "dataSetSurface");
  fnSetDSurface = get_fn<decltype(fnSetDSurface)>(mod, "dataSetDSurface");
}

void ResourcePool::cleanup() {
  if (!instance || !fnFree)
    return;
  fnFree(instance);
}

void ResourcePool::init(MOD mod, std::string path) {
  fnNew = get_fn<decltype(fnNew)>(mod, "resNew");
  fnFree = get_fn<decltype(fnFree)>(mod, "resFree");
  instance = fnNew();

  fnLoad = get_fn<decltype(fnLoad)>(mod, "resLoad");
  fnLoad(instance, path);

  fnGetTopo = get_fn<decltype(fnGetTopo)>(mod, "resGetTopo");
  fnGetUV = get_fn<decltype(fnGetUV)>(mod, "resGetUV");
  fnGetColor = get_fn<decltype(fnGetColor)>(mod, "resGetColor");
  fnGetNormal = get_fn<decltype(fnGetNormal)>(mod, "resGetNormal");
  fnGetUVector = get_fn<decltype(fnGetUVector)>(mod, "resGetUVector");
}
} // namespace nemo
