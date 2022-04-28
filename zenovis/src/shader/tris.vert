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
in mat4 mInstModel;

out vec3 position;
out vec3 iColor;
out vec3 iNormal;
out vec3 iTexCoord;
out vec3 iTangent;
void main()
{
  position = vec3(mInstModel * vec4(vPosition, 1.0));
  iColor = vColor;
  iNormal = transpose(inverse(mat3(mInstModel))) * vNormal;
  iTexCoord = vTexCoord;
  iTangent = mat3(mInstModel) * vTangent;
  gl_Position = mVP * vec4(position, 1.0);
}
)"
