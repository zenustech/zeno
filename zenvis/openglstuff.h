#pragma once
#include "MyShader.hpp"
#include "glad/glad.h"
#include "glm/geometric.hpp"
#include "stdafx.hpp"
#include "main.hpp"
#include "IGraphic.hpp"
#include <Hg/FPSCounter.hpp>
#include <sstream>
#include <cstdlib>
#include <cmath>
#include <array>
#include <stb_image_write.h>
#include <Hg/OpenGL/stdafx.hpp>
#include "zenvisapi.hpp"
#include <Scene.hpp>
#include <thread>
#include <chrono>
namespace zenvis {

struct iTexture3D
{
	GLuint id = 0;
	int dimensions = 0;
	bool is_loaded = false;
};
namespace itexture3D
{
	
	inline void init(zenvis::iTexture3D& t, GLfloat* data, int dimensions)
	{
		t.dimensions = dimensions;

		glGenTextures(1, &t.id);
		glBindTexture(GL_TEXTURE_3D, t.id);

		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		glTexStorage3D(GL_TEXTURE_3D, log2(dimensions), GL_RGBA8, dimensions,dimensions,dimensions);
		glTexSubImage3D(GL_TEXTURE_3D, 0, 0,0,0, dimensions,dimensions,dimensions, GL_RGBA, GL_FLOAT, data);
		glGenerateMipmap(GL_TEXTURE_3D);
		glBindTexture(GL_TEXTURE_3D, 0); 
		t.is_loaded = true;
	}
	inline void uninit(iTexture3D& t)
	{
		if (t.is_loaded) 
			glDeleteTextures(1, &t.id);
		t.is_loaded = false;
	}
	inline void activate(iTexture3D& t, GLuint shader_id, const char* samplerName, int textureLocation)
	{
		glUniform1i(glGetUniformLocation(shader_id, samplerName), textureLocation);
		glActiveTexture(GL_TEXTURE0 + textureLocation);
		glBindTexture(GL_TEXTURE_3D, t.id);
	}
	inline void deactivate()
	{
		glBindTexture(GL_TEXTURE_3D, 0);
	}
	inline void clear(iTexture3D& t, const glm::vec4& clearColor)
	{
		glBindTexture(GL_TEXTURE_3D, t.id);
		glClearTexImage(t.id, 0, GL_RGBA, GL_FLOAT, glm::value_ptr(clearColor));
		glBindTexture(GL_TEXTURE_3D, 0);
	}
	inline void generate_mipmaps(iTexture3D& t)
	{
		glBindTexture(GL_TEXTURE_3D, t.id);
		glGenerateMipmap(GL_TEXTURE_3D);
		glBindTexture(GL_TEXTURE_3D, 0);
	}
	inline void fill(GLfloat* data, float value, int w,int h,int d)
	{
		for(int k=0;k<d;k++)
		for(int j=0;j<h;j++)
		for(int i=0;i<w;i++)
		{
			int idx = 4*(i + j*w + k*w*h);
			data[idx+0] = value;
			data[idx+1] = value;
			data[idx+2] = value;
			data[idx+3] = value;
		}
	}
	inline void fill_pixel(GLfloat* data, int x,int y,int z, int w,int h,int d)
	{
		int floats = 4; // r+g+b+a
		int i = floats * (x + w * (y + h * z));

		assert(i < (w*h*d*floats));

		data[i+0] = 1.0;
		data[i+1] = 1.0;
		data[i+2] = 1.0;
		data[i+3] = 1.0;
	}
	inline void fill_corners(GLfloat* data, int w, int h, int d)
	{
		assert(w == h && h == d);

		for (int x=1; x < w; x++) {
			fill_pixel(data, x, 1, 1, w,h,d);
			fill_pixel(data, x, h-1, 1, w,h,d);
			fill_pixel(data, x, 1, d-1, w,h,d);
			fill_pixel(data, x, h-1, d-1, w,h,d);

			fill_pixel(data, 1, x, 1, w,h,d);
			fill_pixel(data, w-1, x, 1, w,h,d);
			fill_pixel(data, w-1, x, d-1, w,h,d);
			fill_pixel(data, 1, x, d-1, w,h,d);

			fill_pixel(data, 1, 1, x, w,h,d);
			fill_pixel(data, w-1, 1, x, w,h,d);
			fill_pixel(data, w-1, h-1, x, w,h,d);
			fill_pixel(data, 1, h-1, x, w,h,d);
		}
	}
}
}
