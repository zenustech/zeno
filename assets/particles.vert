uniform mat4x4 mVP;
uniform mat4x4 mInvVP;
uniform mat4x4 mView;
uniform mat4x4 mProj;

in vec3 vPosition;
in vec3 vVelocity;

out vec3 position;
out vec3 velocity;

void main()
{
  position = vPosition;
  velocity = vVelocity;

  gl_Position = mVP * vec4(position, 1.0);
  gl_PointSize = D_POINT_SIZE;
}
