#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tinygltf/stb_image_write.h>
#include <filesystem>
#include "zeno/zeno.h"
#include "zeno/types/TextureObject.h"
#include "zeno/types/HeatmapObject.h"
#include "zeno/types/StringObject.h"
#include "glm/common.hpp"

#include <string>
#include "magic_enum.hpp"

namespace zeno
{
	static constexpr char
		texWrapping[] = "REPEAT MIRRORED_REPEAT CLAMP_TO_EDGE CLAMP_TO_BORDER",
		texFiltering[] = "NEAREST LINEAR NEAREST_MIPMAP_NEAREST LINEAR_MIPMAP_NEAREST NEAREST_MIPMAP_LINEAR LINEAR_MIPMAP_LINEAR";

	struct MakeTexture2D
		: zeno::INode
	{
		virtual void apply() override
		{
			auto tex = std::make_shared<zeno::Texture2DObject>();

			tex->path = get_input2<std::string>("path");
            if (has_input("heatmap")) {
                if (tex->path.empty()) {
                    std::srand(std::time(0));
                    tex->path = std::filesystem::temp_directory_path().string() + '/' + "heatmap-" + std::to_string(std::rand()) + ".png";
                }
                auto heatmap = get_input<zeno::HeatmapObject>("heatmap");
                std::vector<uint8_t> col;
                int width = heatmap->colors.size();
                int height = width;
                col.reserve(width * height * 3);
                for (auto i = 0; i < height; i++) {
                    for (auto & color : heatmap->colors) {
                        col.push_back(glm::clamp(int(color[0] * 255.99), 0, 255));
                        col.push_back(glm::clamp(int(color[1] * 255.99), 0, 255));
                        col.push_back(glm::clamp(int(color[2] * 255.99), 0, 255));
                    }
                }
                stbi_flip_vertically_on_write(false);
                stbi_write_png(tex->path.c_str(), width, height, 3, col.data(), 0);
            }

#define SET_TEX_WRAP(TEX, WRAP)                                    \
	if (WRAP == "REPEAT")                                          \
		TEX->WRAP = Texture2DObject::TexWrapEnum::REPEAT;          \
	else if (WRAP == "MIRRORED_REPEAT")                            \
		TEX->WRAP = Texture2DObject::TexWrapEnum::MIRRORED_REPEAT; \
	else if (WRAP == "CLAMP_TO_EDGE")                              \
		TEX->WRAP = Texture2DObject::TexWrapEnum::CLAMP_TO_EDGE;   \
	else if (WRAP == "CLAMP_TO_BORDER")                            \
		TEX->WRAP = Texture2DObject::TexWrapEnum::CLAMP_TO_BORDER; \
	else                                                           \
		throw zeno::Exception(#WRAP + WRAP);

			auto wrapS = get_input<zeno::StringObject>("wrapT")->get();
			SET_TEX_WRAP(tex, wrapS)
			auto wrapT = get_input<zeno::StringObject>("wrapS")->get();
			SET_TEX_WRAP(tex, wrapT)

#undef SET_TEX_WRAP

#define SET_TEX_FILTER(TEX, FILTER)                                           \
	if (FILTER == "NEAREST")                                                  \
		TEX->FILTER = Texture2DObject::TexFilterEnum::NEAREST;                \
	else if (FILTER == "LINEAR")                                              \
		TEX->FILTER = Texture2DObject::TexFilterEnum::LINEAR;                 \
	else if (FILTER == "NEAREST_MIPMAP_NEAREST")                              \
		TEX->FILTER = Texture2DObject::TexFilterEnum::NEAREST_MIPMAP_NEAREST; \
	else if (FILTER == "LINEAR_MIPMAP_NEAREST")                               \
		TEX->FILTER = Texture2DObject::TexFilterEnum::LINEAR_MIPMAP_NEAREST;  \
	else if (FILTER == "NEAREST_MIPMAP_LINEAR")                               \
		TEX->FILTER = Texture2DObject::TexFilterEnum::NEAREST_MIPMAP_LINEAR;  \
	else if (FILTER == "LINEAR_MIPMAP_LINEAR")                                \
		TEX->FILTER = Texture2DObject::TexFilterEnum::LINEAR_MIPMAP_LINEAR;   \
	else                                                                      \
		throw zeno::Exception(#FILTER + FILTER);

			auto minFilter = get_input<zeno::StringObject>("minFilter")->get();
			SET_TEX_FILTER(tex, minFilter)
			auto magFilter = get_input<zeno::StringObject>("magFilter")->get();
			SET_TEX_FILTER(tex, magFilter)

#undef SET_TEX_FILTER

			tex->blockCompression = get_input2<bool>("blockCompression");
			set_output("tex", std::move(tex));
		}
	};

	ZENDEFNODE(
		MakeTexture2D,
		{
			{
				{"readpath", "path"},
				{"heatmap"},
				{(std::string) "enum " + texWrapping, "wrapS", "REPEAT"},
				{(std::string) "enum " + texWrapping, "wrapT", "REPEAT"},
				{(std::string) "enum " + texFiltering, "minFilter", "LINEAR"},
				{(std::string) "enum " + texFiltering, "magFilter", "LINEAR"},
				{"bool", "blockCompression", "false"}
			},
			{
				{"texture", "tex"},
			},
			{},
			{
				"shader",
			},
		});

	struct MakeTextureVDB: zeno::INode 
	{
		const static inline std::string dataTypeKey = "type";
	
		static std::string dataTypeDefaultString() {
			auto name = magic_enum::enum_name(TextureObjectVDB::ElementType::Fp32);
			return std::string(name);
		}

		static std::string dataTypeListString() {
			auto list = magic_enum::enum_names<TextureObjectVDB::ElementType>();

			std::string result;
			for (auto& ele : list) {
				result += " ";
				result += ele;
			}
			return result;
		}

		virtual void apply() override
		{
			auto tex = std::make_shared<zeno::TextureObjectVDB>();

			tex->path = get_input2<std::string>("path");
			tex->channel = get_input2<std::string>("channel");

			auto type = get_input2<std::string>(dataTypeKey);
			auto casted = magic_enum::enum_cast<TextureObjectVDB::ElementType>(type);
			tex->eleType = casted.value_or(TextureObjectVDB::ElementType::Fp32);

			set_output("tex", std::move(tex));
		}
	};

	ZENDEFNODE(
		MakeTextureVDB,
		{
			{
				{"readpath", "path"},
				{"string", "channel", "0"},
				{"enum " + MakeTextureVDB::dataTypeListString(), MakeTextureVDB::dataTypeKey, MakeTextureVDB::dataTypeDefaultString()},
			},
			{
				{"texture", "tex"},
			},
			{},
			{
				"shader",
			},
		});

} // namespace zeno
