#version 330 core

in vec3 position;

out vec4 fColor;

uniform mat4x4 mVP;
uniform mat4x4 mInvVP;
uniform mat4x4 mView;
uniform mat4x4 mProj;
uniform mat4x4 mLocal;
uniform sampler3D volData;
uniform vec3 boundMin;
uniform vec3 boundMax;

vec3 calcRayDir(vec3 pos)
{
  vec4 vpos = mVP * vec4(pos, 1);
  vec2 uv = vpos.xy / vpos.w;
  vec4 ro = mInvVP * vec4(uv, -1, 1);
  vec4 re = mInvVP * vec4(uv, +1, 1);
  vec3 rd = normalize(re.xyz / re.w - ro.xyz / ro.w);
  return rd;
}

vec4 environTexture(sampler2D tex, vec3 dir)
{
  dir = normalize(dir);
  float disxz = length(dir.xz);
  float u = atan(dir.z, dir.x) / acos(-1.0) * 0.5 + 0.5;
  float v = atan(dir.y, disxz) / acos(-1.0) + 0.5;
  return texture(tex, vec2(u, 1 - v));
}

bool hitAABB(
    vec3 bmin, vec3 bmax, vec3 ro, vec3 rd,
    out float near, out float far)
{
  far = 1e6;
  near = -1e6;

  if (abs(rd.x) < 1e-6) {
    if (ro.x < bmin.x || ro.x > bmax.x)
      return false;
  } else {
    float i1 = (bmin.x - ro.x) / rd.x;
    float i2 = (bmax.x - ro.x) / rd.x;

    far = min(far, max(i1, i2));
    near = max(near, min(i1, i2));

    if (near > far)
      return false;
  }

  if (abs(rd.y) < 1e-6) {
    if (ro.y < bmin.y || ro.y > bmax.y)
      return false;
  } else {
    float i1 = (bmin.y - ro.y) / rd.y;
    float i2 = (bmax.y - ro.y) / rd.y;

    far = min(far, max(i1, i2));
    near = max(near, min(i1, i2));

    if (near > far)
      return false;
  }

  if (abs(rd.z) < 1e-6) {
    if (ro.z < bmin.z || ro.z > bmax.z)
      return false;
  } else {
    float i1 = (bmin.z - ro.z) / rd.z;
    float i2 = (bmax.z - ro.z) / rd.z;

    far = min(far, max(i1, i2));
    near = max(near, min(i1, i2));

    if (near > far)
      return false;
  }

  return true;
}

float sampleVolume(sampler3D tex, vec3 pos) {
  pos = (pos - boundMin) / (boundMax - boundMin);
  const float bound = 0.01;
  if (any(lessThan(pos, vec3(bound))) || any(greaterThan(pos, vec3(1 - bound)))) {
    return 0.0;
  }
  return texture(tex, pos).r;
}

vec3 gradientVolume(sampler3D tex, vec3 pos, float dx) {
  vec2 d = vec2(dx, 0);
  return vec3(
      sampleVolume(tex, pos + d.xyy) - sampleVolume(tex, pos - d.xyy),
      sampleVolume(tex, pos + d.yxy) - sampleVolume(tex, pos - d.yxy),
      sampleVolume(tex, pos + d.yyx) - sampleVolume(tex, pos - d.yyx));
}

void main()
{
  vec3 ro = position;
  vec3 rd = calcRayDir(position);

#if 0
  float near, far;
  if (hitAABB(boundMin, boundMax, ro, rd, near, far)) {
    float dt = max(0.005, (far - near) / 256);
    for (float t = near; t < far; t += dt) {
      vec3 pos = ro + t * rd;
      vec3 grad = gradientVolume(volData, pos, dt * 2);

      if (length(grad) > 1e-3) {
        vec3 nrm = normalize(grad);
        vec3 clr = vec3(abs(dot(nrm, normalize(vec3(1, 2, 3)))));
        fColor = vec4(clr, 1);
        return;
      }
    }
  }

  fColor = vec4(0);
#else
  float near, far;
  float opacity = 1;
  if (hitAABB(boundMin, boundMax, ro, rd, near, far)) {
    float dt = max(0.01, (far - near) / 128);
    for (float t = near; t < far; t += dt) {
      vec3 pos = ro + t * rd;
      float rho = sampleVolume(volData, pos);
      opacity *= exp(-dt * rho);
      if (opacity < 0.05)
        break;
    }
  }

  fColor = vec4(vec3(0.8), 1 - opacity);
#endif
}
