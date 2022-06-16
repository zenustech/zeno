R"(#version 330

uniform mat4 mVP;
uniform mat4 mInvVP;
uniform mat4 mView;
uniform mat4 mProj;
uniform float mPointScale;

in vec3 vPosition;
in vec3 vColor;
in vec3 vNormal;
in vec3 vTexCoord;
in vec3 vTangent;

out vec3 position;
out vec3 color;
out float radius;
out float opacity;
void main()
{
  position = vPosition;
  color = vColor;
  radius = vNormal.x;
  opacity = vNormal.y;

  vec3 posEye = vec3(mView * vec4(position, 1.0));
  float dist = length(posEye);
  if (radius != 0)
    gl_PointSize = max(1, radius * mPointScale / dist);
  else
    gl_PointSize = 1.5;
  gl_Position = mVP * vec4(position, 1.0);
}
)"
