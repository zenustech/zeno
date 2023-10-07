R"(#version 130

uniform mat4 mVP;
uniform mat4 mInvVP;
uniform mat4 mView;
uniform mat4 mProj;
uniform mat4 mInvView;
uniform mat4 mInvProj;

in vec3 position;
out vec4 fColor;
void main()
{
  fColor = vec4(0.89, 0.57, 0.15, 1.0);
}
)"
