uniform mat4x4 mVP;
uniform mat4x4 mInvVP;
uniform mat4x4 mView;
uniform mat4x4 mProj;

in vec3 position;
in vec3 velocity;

out vec4 fColor;

void main()
{
  if (length(gl_PointCoord - vec2(0.5)) > 0.5)
    discard;
  float factor = length(velocity) / max(D_VEL_MAG, 1e-4);
  vec3 color = mix(vec3(0.2, 0.3, 0.6), vec3(1.1, 0.8, 0.5), factor);
  fColor = vec4(color, 1.0);
}
