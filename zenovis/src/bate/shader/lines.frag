R"(#version 130

uniform mat4 mVP;
uniform mat4 mInvVP;
uniform mat4 mView;
uniform mat4 mProj;
uniform mat4 mInvView;
uniform mat4 mInvProj;

in vec3 position;
in vec3 color;
out vec4 fColor;
void main()
{
  fColor = vec4(color, 1.0);
}
)"
