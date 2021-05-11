#version 120

uniform mat4 mVP;
uniform mat4 mInvVP;
uniform mat4 mView;
uniform mat4 mProj;

varying vec3 position;
varying vec3 velocity;

void main()
{
  if (length(gl_PointCoord - vec2(0.5)) > 0.5)
    discard;
  float factor = length(velocity) / max(float(10), 1e-4);
  vec3 color = mix(vec3(0.2, 0.3, 0.6), vec3(1.1, 0.8, 0.5), factor);
  gl_FragColor = vec4(color, 1.0);
}
