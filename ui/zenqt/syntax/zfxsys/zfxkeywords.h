#pragma once

#include <QString>

inline QStringList zfxGlobalVar = {"$F", "$PI", "$DT", "$T"};

inline QStringList zfxFunction = {"exp", "sin", "cos", "tan", "asin", "acos", "atan", 
	"dot", "cross", "normalize", "normalizesafe", "radians", "degrees", "fb2i", "ib2i", "abs", "min", "max", "fit", "efit", "sqrt", "clamp", 
	"log", "pow", "fmod", "floor", "ceil", "atan2", "distance", "length", "mix", "vec2", "vec4", "all", "any",
	"applyAffine", "vec3", "round"
};

inline QStringList zfxAttr = { "clr", "pos", "nrm", "rad", "val" };
