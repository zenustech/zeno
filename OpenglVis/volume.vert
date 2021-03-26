#version 330 core

in vec3 vPosition;

out vec3 position;

uniform mat4x4 mVP;
uniform mat4x4 mInvVP;
uniform mat4x4 mView;
uniform mat4x4 mProj;
uniform mat4x4 mLocal;

void main()
{
  position = (mLocal * vec4(vPosition, 1.0)).xyz;
  gl_Position = mVP * vec4(position, 1.0);
}
