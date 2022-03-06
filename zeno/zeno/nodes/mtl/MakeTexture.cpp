#include "zeno/zeno.h"
#include "zeno/types/TextureObject.h"
#include "zeno/types/MaterialObject.h"
#include "zeno/types/StringObject.h"

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

			auto path = get_input<zeno::StringObject>("path")->get();
			tex->path = path;

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

			set_output("tex", std::move(tex));
		}
	};

	ZENDEFNODE(
		MakeTexture2D,
		{
			{
				{"readpath", "path"},
				{(std::string) "enum " + texWrapping, "wrapS", "REPEAT"},
				{(std::string) "enum " + texWrapping, "wrapT", "REPEAT"},
				{(std::string) "enum " + texFiltering, "minFilter", "LINEAR"},
				{(std::string) "enum " + texFiltering, "magFilter", "LINEAR"},
			},
			{
				{"texture", "tex"},
			},
			{},
			{
				"shader",
			},
		});

	struct MaterialTexture2D
		: zeno::INode
	{
		virtual void apply() override
		{
			auto mtl = get_input<zeno::MaterialObject>("mtl");
			auto tex = get_input<zeno::Texture2DObject>("tex");
            int texId = mtl->tex2Ds.size();
			mtl->tex2Ds.push_back(tex);
            auto texCode = "uniform sampler2D zenotex" + std::to_string(texId) + ";\n";
            mtl->common.insert(0, texCode);
			set_output("mtl", std::move(mtl));
		}
	};

	ZENDEFNODE(
		MaterialTexture2D,
		{
			{
				{"material", "mtl"},
				{"texture", "tex"},
			},
			{
				{"material", "mtl"},
			},
			{},
			{
				"shader",
			},
		});

} // namespace zeno
