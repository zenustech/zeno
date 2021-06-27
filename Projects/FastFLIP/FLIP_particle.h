#pragma once
#include "vec.h"

struct FLIP_particle
{
	LosTopos::Vec3f pos;
	LosTopos::Vec3f vort;
	LosTopos::Vec3f vel;
	float length;
	float circ;
	float volm;
	FLIP_particle()
	{
		length = 0;
		for (int i = 0; i < 3; i++)
		{
			pos[i] = 0;
			vel[i] = 0;
			vort[i] = 0;
		}
		circ = 0;
		volm = 1;

	}
	~FLIP_particle()
	{
	}
	FLIP_particle(const FLIP_particle& p)
	{
		pos = p.pos;
		vel = p.vel;
		vort = p.vort;
		length = p.length;
		circ = p.circ;
		volm = p.volm;
	}
	FLIP_particle(const LosTopos::Vec3f& p,
		const LosTopos::Vec3f& v)
	{
		pos = p;
		vel = v;
	}
	FLIP_particle(const LosTopos::Vec3f& p, const LosTopos::Vec3f& v, const LosTopos::Vec3f& w, const float& l, const float& _volm)
	{
		pos = p;
		vel = v;
		vort = w;
		length = l;
		circ = mag(vort) / length;
		volm = _volm;
	}
};

struct minimum_FLIP_particle {
	LosTopos::Vec3f pos;
	LosTopos::Vec3f vel;
	minimum_FLIP_particle()
	{
		for (int i = 0; i < 3; i++)
		{
			pos[i] = 0;
			vel[i] = 0;
		}
	}

	minimum_FLIP_particle(const minimum_FLIP_particle& p)
	{
		pos = p.pos;
		vel = p.vel;
	}

	minimum_FLIP_particle(const LosTopos::Vec3f& p,
		const LosTopos::Vec3f& v)
	{
		pos = p;
		vel = v;
	}
};