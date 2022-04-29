R"(#version 330

uniform mat4 mVP;
uniform mat4 mInvVP;
uniform mat4 mView;
uniform mat4 mProj;
uniform mat4 mInvView;
uniform mat4 mInvProj;
uniform bool mSmoothShading;
uniform bool mNormalCheck;
uniform bool mRenderWireframe;

in vec3 position;
in vec3 iColor;
in vec3 iNormal;
in vec3 iTexCoord;
in vec3 iTangent;
out vec4 fColor;

void pixarONB(vec3 n, out vec3 b1, out vec3 b2) {
	vec3 up = abs(n.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    b1 = normalize(cross(up, n));
    b2 = cross(n, b1);
}

vec3 pbr(vec3 albedo, float roughness, float metallic, float specular, vec3 nrm, vec3 idir, vec3 odir) {

  vec3 hdir = normalize(idir + odir);
  float NoH = max(1e-5, dot(hdir, nrm));
  float NoL = max(1e-5, dot(idir, nrm));
  float NoV = max(1e-5, dot(odir, nrm));
  float VoH = clamp(dot(odir, hdir), 1e-5, 1.);
  float LoH = clamp(dot(idir, hdir), 1e-5, 1.);

  vec3 f0 = metallic * albedo + (1. - metallic) * 0.16 * specular;
  vec3 fdf = f0 + (1. - f0) * pow(1. - VoH, 5.);

  roughness *= roughness;
  float k = (roughness + 1.) * (roughness + 1.) / 8.;
  float vdf = 0.25 / ((NoV * k + 1. - k) * (NoL * k + 1. - k));

  float alpha2 = max(0., roughness * roughness);
  float denom = 1. - NoH * NoH * (1. - alpha2);
  float ndf = alpha2 / (denom * denom);

  vec3 brdf = fdf * vdf * ndf * f0 + (1. - f0) * albedo;
  return brdf * NoL;
}

vec3 studioShading(vec3 albedo, vec3 view_dir, vec3 normal, vec3 tangent) {
    vec3 color = vec3(0.0);
    vec3 light_dir;

    light_dir = normalize((mInvView * vec4(1., 2., 5., 0.)).xyz);
    color += vec3(0.45, 0.47, 0.5) * pbr(albedo, 0.44, 0.0, 1.0, normal, light_dir, view_dir);

    light_dir = normalize((mInvView * vec4(-4., -2., 1., 0.)).xyz);
    color += vec3(0.3, 0.23, 0.18) * pbr(albedo, 0.37, 0.0, 1.0, normal, light_dir, view_dir);

    light_dir = normalize((mInvView * vec4(3., -5., 2., 0.)).xyz);
    color += vec3(0.15, 0.2, 0.22) * pbr(albedo, 0.48, 0.0, 1.0, normal, light_dir, view_dir);

    color *= 1.2;
    //color = pow(clamp(color, 0., 1.), vec3(1./2.2));
    return color;
}

vec3 calcRayDir(vec3 pos)
{
  vec4 vpos = mView * vec4(pos, 1);
//   vec2 uv = vpos.xy / vpos.w;
//   vec4 ro = mInvVP * vec4(uv, -1, 1);
//   vec4 re = mInvVP * vec4(uv, +1, 1);
//   vec3 rd = normalize(re.xyz / re.w - ro.xyz / ro.w);
  return normalize(vpos.xyz);
}

void main() {
  if (mRenderWireframe) {
    fColor = vec4(0.89, 0.57, 0.15, 1.0);
    return;
  }
  vec3 normal;
  if (mSmoothShading) {
    normal = normalize(iNormal);
  } else {
    normal = normalize(cross(dFdx(position), dFdy(position)));
  }
  vec3 viewdir = -calcRayDir(position);
  vec3 albedo = iColor;
  vec3 normalInView = transpose(inverse(mat3(mView[0].xyz, mView[1].xyz, mView[2].xyz)))*normal;
  if(dot(-viewdir, normalInView)>0)
    normal = - normal;

  //normal = faceforward(normal, -viewdir, normal);
  vec3 tangent = iTangent;
  if (tangent == vec3(0)) {
   vec3 unusedbitan;
   pixarONB(normal, tangent, unusedbitan);
  }

  vec3 color = studioShading(albedo, viewdir, normal, tangent);
  
  fColor = vec4(color, 1);
  
  if (mNormalCheck) {
      float intensity = clamp((mView * vec4(normal, 0)).z, 0, 1) * 0.4 + 0.6;
      if (gl_FrontFacing) {
        fColor = vec4(0.42 * intensity, 0.42 * intensity, 0.93 * intensity, 1);
      } else {
        fColor = vec4(0.87 * intensity, 0.22 * intensity, 0.22 * intensity, 1);
      }
  }
}
)"
