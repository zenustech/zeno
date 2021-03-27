#version 330 core

in vec3 vPosition;
in vec2 vTexcoord;
in vec3 vNormal;

out vec3 position;
out vec2 texcoord;
out vec3 iNormal;

uniform mat4x4 mVP;
uniform mat4x4 mInvVP;
uniform mat4x4 mView;
uniform mat4x4 mProj;

void main()
{
  position = vPosition;
  texcoord = vTexcoord;
  iNormal = vNormal;

  gl_Position = mVP * vec4(position, 1.0);
}
