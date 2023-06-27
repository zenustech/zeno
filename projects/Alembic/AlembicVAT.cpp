#include "ABCCommon.h"
#include "ABCTree.h"
#include "Alembic/Abc/IObject.h"
#include "tinyexr.h"
#include "zeno/funcs/PrimitiveUtils.h"
#include "zeno/para/parallel_reduce.h"
#include <filesystem>
#include <zeno/logger.h>
#include <zeno/zeno.h>

namespace zeno {

bool SaveEXR(const float* rgb, size_t width, size_t height, const char* outfilename) {
    EXRHeader header;
    InitEXRHeader(&header);

    EXRImage image;
    InitEXRImage(&image);

    image.num_channels = 3;

    std::vector<float> images[3];
    images[0].resize(width * height);
    images[1].resize(width * height);
    images[2].resize(width * height);

    // Split RGBRGBRGB... into R, G and B layer
    for (int i = 0; i < width * height; i++) {
        images[0][i] = rgb[3*i+0];
        images[1][i] = rgb[3*i+1];
        images[2][i] = rgb[3*i+2];
    }

    float* image_ptr[3];
    image_ptr[0] = &(images[2].at(0)); // B
    image_ptr[1] = &(images[1].at(0)); // G
    image_ptr[2] = &(images[0].at(0)); // R

    image.images = (unsigned char**)image_ptr;
    image.width = width;
    image.height = height;

    header.num_channels = 3;
    header.channels = (EXRChannelInfo *)malloc(sizeof(EXRChannelInfo) * header.num_channels);
    // Must be (A)BGR order, since most of EXR viewers expect this channel order.
    strncpy(header.channels[0].name, "B", 255); header.channels[0].name[strlen("B")] = '\0';
    strncpy(header.channels[1].name, "G", 255); header.channels[1].name[strlen("G")] = '\0';
    strncpy(header.channels[2].name, "R", 255); header.channels[2].name[strlen("R")] = '\0';

    header.pixel_types = (int *)malloc(sizeof(int) * header.num_channels);
    header.requested_pixel_types = (int *)malloc(sizeof(int) * header.num_channels);
    for (int i = 0; i < header.num_channels; i++) {
        header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT; // pixel type of input image
        header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_HALF; // pixel type of output image to be stored in .EXR
    }

    const char* err = nullptr; // or nullptr in C++11 or later.
    int ret = SaveEXRImageToFile(&image, &header, outfilename, &err);
    if (ret != TINYEXR_SUCCESS) {
        fprintf(stderr, "Save EXR err: %s\n", err);
        FreeEXRErrorMessage(err); // free's buffer for an error message
        return ret;
    }
    printf("Saved exr file. [ %s ] \n", outfilename);

    free(header.channels);
    free(header.pixel_types);
    free(header.requested_pixel_types);
}

vec3f normalized_vec3f(vec3f vec, vec3f _min, vec3f _max) {
    vec = (vec - _min) / (_max - _min);
    return vec;
}

int align_to(int count, int align) {
    int remainder = count % align;
    if (remainder == 0) {
        return count;
    }
    else {
        return count + (align - remainder);
    }
}

void writeObjFile(
    const std::shared_ptr<zeno::PrimitiveObject>& primitive,
    const char *path,
    int32_t frameNum,
    const std::pair<zeno::vec3f, zeno::vec3f>& bbox
)
{
    FILE *fp = fopen(path, "w");
    if (!fp) {
        perror(path);
        abort();
    }

    fprintf(fp, "# Zeno generated from an alembic file.\n");

    auto& vertices = primitive->verts;
    auto& triangle = primitive->tris;

    size_t vatWidth = std::min(vertices.size(), (size_t)8192);
    auto rowsPerFrame = static_cast<int32_t>(std::ceil((float)vertices.size() / (float)vatWidth));
    size_t vatHeight = rowsPerFrame * frameNum;
    fprintf(fp, "# metadata VATWidth %d\n", vatWidth);
    fprintf(fp, "# metadata RowsPerFrame %d\n", rowsPerFrame);
    fprintf(fp, "# metadata FrameNum %d\n", frameNum);
    fprintf(fp, "# metadata VATHeight %d\n", vatHeight);
    fprintf(fp, "# metadata BMin %f %f %f\n", bbox.first[0], bbox.first[1], bbox.first[2]);
    fprintf(fp, "# metadata BMax %f %f %f\n", bbox.second[0], bbox.second[1], bbox.second[2]);

    const auto map_into_bbox = [&bbox](const zeno::vec3f& v) {
//        return normalized_vec3f(v, bbox.first, bbox.second);
        return v;
    };

    for (auto const &vert: vertices) {
        const auto v = map_into_bbox(vert);
        fprintf(fp, "v %f %f %f\n", v[0], v[1], v[2]);
    }

    std::vector<std::pair<float, float>> vatUvMap;
    for (size_t i = 0; i < vertices.size(); i++) {
        float u1 = (float(i % vatWidth)) / (float)vatWidth;
        float u2 = (float((i + 1) % vatWidth)) / (float)vatWidth;
        if (u1 > 1.0f) {
            u1 -= 1.0f;
            u2 -= 1.0f;
        }
        if (u1 > u2) {
            u2 = 1.0f;
        }
        float v1 = std::floor((float)i / (float)vatWidth) / (float)vatHeight;
        float v2 = std::floor(float(i + vatWidth) / (float)vatWidth) / (float)vatHeight;
        vatUvMap.emplace_back((u1 + u2) * 0.5f, std::min((v1 + v2) * 0.5f, 1.0f));
        fprintf(fp, "vn %.10f %.10f %.10f\n", vatUvMap[i].first, vatUvMap[i].second, 0.0f);
    }

    auto& uv0 = triangle.attr<zeno::vec3f>("uv0");
    auto& uv1 = triangle.attr<zeno::vec3f>("uv1");
    auto& uv2 = triangle.attr<zeno::vec3f>("uv2");

    int32_t count = 0;
    for (auto const &ind: triangle) {
        const int32_t v0 = ind[0], v1 = ind[1], v2 = ind[2];
        const int32_t ui0 = count * 3 + 1, ui1 = count * 3 + 2, ui2 = count * 3 + 3;

        fprintf(fp, "vt %.10f %.10f\n", uv0[count][0], uv0[count][1]);
        fprintf(fp, "vt %.10f %.10f\n", uv1[count][0], uv1[count][1]);
        fprintf(fp, "vt %.10f %.10f\n", uv2[count][0], uv2[count][1]);
        fprintf(fp, "f %d/%d/%d %d/%d/%d %d/%d/%d\n",
            v0 + 1, ui0, v0 + 1,
            v1 + 1, ui1, v1 + 1,
            v2 + 1, ui2, v2 + 1
        );
        count++;
    }
    fclose(fp);
}

struct AlembicToSoftBodyVAT: public INode {
    Alembic::Abc::v12::IArchive archive;
    bool read_done = false;

    virtual void apply() override {
        int frameStart = get_input2<int>("frameStart");
        int frameEnd = get_input2<int>("frameEnd");
        if (frameEnd < frameStart) {
            std::swap(frameEnd, frameStart);
        }
        int frameNum = frameEnd - frameStart;
        if (frameNum <= 0) {
            zeno::log_error("Invalid frame range: {} - {}", frameStart, frameEnd);
            return;
        }
        bool use_xform = get_input2<bool>("useXForm");
        std::string writePath = get_input2<std::string>("outputPath");
        std::string path = get_input2<std::string>("path");
        size_t vatWidth;
        int32_t rowsPerFrame;
        size_t spaceToAlign;
        std::vector<vec3f> temp_bboxs;
        size_t vatHeight;
        {
            if (!read_done) {
                archive = readABC(path);
            }
            double _start, _end;
            GetArchiveStartAndEndTime(archive, _start, _end);
            auto obj = archive.getTop();
            std::vector<float> pos_f32;
            std::vector<float> nrm_f32;
            std::shared_ptr<ListObject> frameList = std::make_shared<ListObject>();
            for (int32_t idx = frameStart; idx < frameEnd; ++idx) {
                const int32_t frameIndex = frameEnd - idx - 1;
                auto abctree = std::make_shared<ABCTree>();
                auto prims = std::make_shared<zeno::ListObject>();
                traverseABC(obj, *abctree, idx, read_done);
                if (use_xform) {
                    prims = get_xformed_prims(abctree);
                } else {
                    abctree->visitPrims([&] (auto const &p) {
                        auto np = std::static_pointer_cast<PrimitiveObject>(p->clone());
                        prims->arr.push_back(np);
                    });
                }
                auto mergedPrim = zeno::primMerge(prims->getRaw<PrimitiveObject>());
                if (get_input2<bool>("flipFrontBack")) {
                    flipFrontBack(mergedPrim);
                }
                zeno::primTriangulate(mergedPrim.get());
                frameList->arr.push_back(mergedPrim);
                auto bbox = parallel_reduce_minmax(mergedPrim->verts.begin(), mergedPrim->verts.end());
                temp_bboxs.push_back(bbox.first);
                temp_bboxs.push_back(bbox.second);
                set_output("primitive", mergedPrim);
            }
            // reduce bbox_temp to actual bbox
            auto bbox = parallel_reduce_minmax(temp_bboxs.begin(), temp_bboxs.end());
            read_done = true;
            for (int32_t idx = frameStart; idx < frameEnd; ++idx) {
                zeno::log_info("Processing frame {} / {} ...", idx + 1, frameEnd);
                const int32_t frameIndex = idx - frameStart;
                auto mergedPrim = safe_dynamic_cast<zeno::PrimitiveObject>(frameList->arr[frameIndex]);
                // Save first frame mesh to obj
                if (frameIndex == 0) {
                    vatWidth = std::min(mergedPrim->verts.size(), (size_t)8192);
                    rowsPerFrame = static_cast<int32_t>(std::ceil((float)mergedPrim->verts.size() / (float)vatWidth));
                    vatHeight = rowsPerFrame * frameNum;
                    spaceToAlign = vatWidth * rowsPerFrame - mergedPrim->verts.size();
                    std::string objPath = writePath + ".obj";
                    if (std::filesystem::exists(objPath)) {
                        std::filesystem::remove(objPath);
                    }
                    writeObjFile(mergedPrim, objPath.c_str(), frameNum, bbox);
                }
                // Save other frames to vat
                // Position
                for (auto& vert : mergedPrim->verts) {
                    auto vec = normalized_vec3f(vert, bbox.first, bbox.second);
                    pos_f32.push_back(vec[0]);
                    pos_f32.push_back(vec[1]);
                    pos_f32.push_back(vec[2]);
                }
                for (size_t idx = 0; idx < spaceToAlign; ++idx) {
                    pos_f32.push_back(0.0f);
                    pos_f32.push_back(0.0f);
                    pos_f32.push_back(0.0f);
                }
                zeno::primCalcNormal(mergedPrim.get());
                auto& nrm_ref = mergedPrim->verts.attr<vec3f>("nrm");
                for (auto& normal : nrm_ref) {
                    nrm_f32.push_back(normal[0]);
                    nrm_f32.push_back(normal[1]);
                    nrm_f32.push_back(normal[2]);
                }
                for (size_t idx = 0; idx < spaceToAlign; ++idx) {
                    nrm_f32.push_back(0.0f);
                    nrm_f32.push_back(0.0f);
                    nrm_f32.push_back(0.0f);
                }
            }
            std::string posPath = writePath + "-position-texture.exr";
            if (std::filesystem::exists(posPath)) {
                std::filesystem::remove(posPath);
            }
            std::string nrmPath = writePath + "-normal-texture.exr";
            if (std::filesystem::exists(nrmPath)) {
                std::filesystem::remove(nrmPath);
            }
            SaveEXR(pos_f32.data(), vatWidth, vatHeight, posPath.c_str());
            SaveEXR(nrm_f32.data(), vatWidth, vatHeight, nrmPath.c_str());
        }
    }
};

ZENDEFNODE(AlembicToSoftBodyVAT, {
    {
      {"readpath", "path"},
      {"bool", "useXForm", "1"},
      {"bool", "flipFrontBack", "1"},
      {"int", "frameEnd", "1"},
      {"int", "frameStart", "0"},
      {"writepath", "outputPath", ""},
    },
    { {"primitive"} },
    {
    },
    {"alembic", "primitive"},
});

void writeDynamicRemeshObjFile(
  const char *path,
  int32_t frameNum,
  const std::pair<zeno::vec3f, zeno::vec3f>& bbox,
  size_t triNum
) {

  FILE *fp = fopen(path, "w");
  if (!fp) {
    perror(path);
    abort();
  }

  fprintf(fp, "# Zeno generated from an alembic file.\n");

  const size_t vertNum = triNum * 3;

  size_t vatWidth = std::min(vertNum, (size_t)8192);
  auto rowsPerFrame = static_cast<int32_t>(std::ceil((float)vertNum / (float)vatWidth));
  size_t vatHeight = rowsPerFrame * frameNum;
  fprintf(fp, "# metadata VATWidth %d\n", vatWidth);
  fprintf(fp, "# metadata RowsPerFrame %d\n", rowsPerFrame);
  fprintf(fp, "# metadata FrameNum %d\n", frameNum);
  fprintf(fp, "# metadata VATHeight %d\n", vatHeight);
  fprintf(fp, "# metadata BMin %f %f %f\n", bbox.first[0], bbox.first[1], bbox.first[2]);
  fprintf(fp, "# metadata BMax %f %f %f\n", bbox.second[0], bbox.second[1], bbox.second[2]);

  constexpr float scale = 0.01f;

  const float center = float(triNum) * scale * 0.5f;

  for (size_t idx = 0; idx < triNum; ++idx) {
    float x = float(idx) * scale - center;
    size_t idxBase = idx * 3 + 1;


    auto outputUV = [&](size_t vertId) {
      float u1 = float(vertId % vatWidth) / float(vatWidth);
      float u2 = float((vertId + 1) % vatWidth) / float(vatWidth);
      if (u1 > 1.0f) {
        u1 -= 1.0f;
        u2 -= 1.0f;
      }
      if (u1 > u2) u2 = 1.0f;
      float u = (u1 + u2) * 0.5f;
      float v1 = std::floor((float)vertId / (float)vatWidth) / (float)vatHeight;
      float v2 = std::floor(float(vertId + vatWidth) / (float)vatWidth) / (float)vatHeight;
      float v = std::min((v1 + v2) * 0.5f, 1.0f);

      fprintf(fp, "vt %.10f %.10f\n", u, v);
    };

    outputUV(idxBase - 1);
    outputUV(idxBase);
    outputUV(idxBase + 1);

    fprintf(fp, "v %f %f %f\n", x, x, x);
    fprintf(fp, "v %f %f %f\n", x + 0.25, x, x + 0.25);
    fprintf(fp, "v %f %f %f\n", x - 0.25, x, x - 0.25);
    fprintf(fp, "f %d/%d %d/%d %d/%d\n", idxBase, idxBase, idxBase + 1, idxBase + 1, idxBase + 2, idxBase + 2);
  }
  fclose(fp);
}

struct AlembicToDynamicRemeshVAT : public INode {
    Alembic::Abc::v12::IArchive archive;
    bool read_done = false;

    void apply() override {
      int frameStart = get_input2<int>("frameStart");
      int frameEnd = get_input2<int>("frameEnd");
      if (frameEnd < frameStart) {
        std::swap(frameEnd, frameStart);
      }
      int frameNum = frameEnd - frameStart;
      if (frameNum <= 0) {
        zeno::log_error("Invalid frame range: {} - {}", frameStart, frameEnd);
        return;
      }
      bool use_xform = get_input2<bool>("useXForm");
      std::string writePath = get_input2<std::string>("outputPath");
      std::string path = get_input2<std::string>("path");
      bool shouldFlipFrontBack = get_input2<bool>("flipFrontBack");

      std::vector<float> pos_f32;
      std::vector<float> nrm_f32;

      std::vector<vec3f> temp_bboxs;

      if (!read_done) {
        archive = readABC(path);
      }

      auto obj = archive.getTop();
      std::shared_ptr<ListObject> frameList = std::make_shared<ListObject>();
      size_t maxTriNum = 0;
      for (int32_t idx = frameStart; idx < frameEnd; ++idx) {
        const int32_t frameIndex = frameEnd - idx - 1;
        auto abctree = std::make_shared<ABCTree>();
        auto prims = std::make_shared<zeno::ListObject>();
        traverseABC(obj, *abctree, idx, read_done);
        if (use_xform) {
          prims = get_xformed_prims(abctree);
        } else {
          abctree->visitPrims([&] (auto const &p) {
            auto np = std::static_pointer_cast<PrimitiveObject>(p->clone());
            prims->arr.push_back(np);
          });
        }
        auto mergedPrim = zeno::primMerge(prims->getRaw<PrimitiveObject>());
        if (shouldFlipFrontBack) {
          flipFrontBack(mergedPrim);
        }
        zeno::primTriangulate(mergedPrim.get());
        maxTriNum = zeno::max(mergedPrim->tris.size(), maxTriNum);
        frameList->arr.push_back(mergedPrim);
        auto bbox = parallel_reduce_minmax(mergedPrim->verts.begin(), mergedPrim->verts.end());
        temp_bboxs.push_back(bbox.first);
        temp_bboxs.push_back(bbox.second);
        set_output("primitive", mergedPrim);
      }
      auto bbox = parallel_reduce_minmax(temp_bboxs.begin(), temp_bboxs.end());
      read_done = true;

      writeDynamicRemeshObjFile(writePath.c_str(), frameNum, bbox, maxTriNum);

      const size_t vertNum = maxTriNum * 3;
      size_t vatWidth = std::min(vertNum, (size_t)8192);
      auto rowsPerFrame = static_cast<int32_t>(std::ceil((float)vertNum / (float)vatWidth));
      size_t vatHeight = rowsPerFrame * frameNum;
      size_t spaceToAlign = -1;

      for (int32_t idx = frameStart; idx < frameEnd; ++idx) {
        zeno::log_info("Processing frame {} / {} ...", idx + 1, frameEnd);
        const int32_t frameIndex = idx - frameStart;
        auto mergedPrim = safe_dynamic_cast<zeno::PrimitiveObject>(frameList->arr[frameIndex]);
        spaceToAlign = vatWidth * rowsPerFrame - mergedPrim->tris.size() * 3;
        zeno::primCalcNormal(mergedPrim.get());
        auto& nrm_ref = mergedPrim->verts.attr<vec3f>("nrm");
        for (auto& trig : mergedPrim->tris) {
          for (uint8_t i = 0; i < 3; ++i) {
            auto vec = normalized_vec3f(mergedPrim->verts[trig[i]], bbox.first, bbox.second);
            pos_f32.push_back(vec[0]);
            pos_f32.push_back(vec[1]);
            pos_f32.push_back(vec[2]);
            auto& nrm = nrm_ref[trig[i]];
            nrm_f32.push_back(nrm[0]);
            nrm_f32.push_back(nrm[1]);
            nrm_f32.push_back(nrm[2]);
          }
        }
        for (size_t idx = 0; idx < spaceToAlign; ++idx) {
          pos_f32.push_back(0.0f);
          pos_f32.push_back(0.0f);
          pos_f32.push_back(0.0f);
          nrm_f32.push_back(0.0f);
          nrm_f32.push_back(0.0f);
          nrm_f32.push_back(0.0f);
        }
      }

      std::string posPath = writePath + "-position-texture.exr";
      if (std::filesystem::exists(posPath)) {
        std::filesystem::remove(posPath);
      }
      std::string nrmPath = writePath + "-normal-texture.exr";
      if (std::filesystem::exists(nrmPath)) {
        std::filesystem::remove(nrmPath);
      }
      SaveEXR(pos_f32.data(), vatWidth, vatHeight, posPath.c_str());
      SaveEXR(nrm_f32.data(), vatWidth, vatHeight, nrmPath.c_str());
    }

};

ZENDEFNODE(AlembicToDynamicRemeshVAT, {
  {
    {"readpath", "path"},
    {"bool", "useXForm", "1"},
    {"bool", "flipFrontBack", "1"},
    {"int", "frameEnd", "1"},
    {"int", "frameStart", "0"},
    {"writepath", "outputPath", ""},
  },
  { {"primitive"} },
  {},
  {"alembic", "primitive"},
});

}
