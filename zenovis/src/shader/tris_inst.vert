R"(#version 330

uniform mat4 mVP;
uniform mat4 mInvVP;
uniform mat4 mView;
uniform mat4 mProj;
uniform mat4 mInvView;
uniform mat4 mInvProj;

in vec3 vPosition;
in vec3 vColor;
in vec3 vNormal;
in vec3 vTexCoord;
in vec3 vTangent;

out vec3 position;
out vec3 iColor;
out vec3 iNormal;
out vec3 iTexCoord;
out vec3 iTangent;

void main()
{
  position = vPosition;
  iColor = vColor;
  iNormal = vNormal;
  iTexCoord = vTexCoord;
  iTangent = vTangent;
  gl_Position = mVP * vec4(position, 1.0);
}
)"
