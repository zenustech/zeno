#version 330 core

in vec3 position;
in vec3 velocity;

out vec4 fColor;

uniform mat4x4 mVP;
uniform mat4x4 mInvVP;
uniform mat4x4 mView;
uniform mat4x4 mProj;
uniform mat4x4 mLocal;

void main()
{
  if (length(gl_PointCoord - vec2(0.5)) > 0.5)
    discard;
  vec3 color = vec3(1.0);
  color = mix(vec3(0.2, 0.3, 0.6), vec3(1.1, 0.8, 0.5), length(velocity) * 0.05);
  fColor = vec4(color, 1.0);
}
