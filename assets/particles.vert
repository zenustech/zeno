#version 120

uniform mat4 mVP;
uniform mat4 mInvVP;
uniform mat4 mView;
uniform mat4 mProj;

attribute vec3 vPosition;
attribute vec3 vVelocity;

varying vec3 position;
varying vec3 velocity;

void main()
{
  position = vPosition;
  velocity = vVelocity;

  gl_Position = mVP * vec4(position, 1);
  vec3 vpos = gl_Position.xyz;

  float radius = 100.0;

  gl_PointSize = radius / vpos.z;
}
