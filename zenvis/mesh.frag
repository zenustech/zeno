#version 330 core

in vec3 position;
in vec2 texcoord;
in vec3 iNormal;

out vec4 fColor;

uniform mat4x4 mVP;
uniform mat4x4 mInvVP;
uniform mat4x4 mView;
uniform mat4x4 mProj;
//uniform sampler2D ourTexture;

struct Light {
  vec3 dir;
  vec3 color;
};

struct Material {
  vec3 albedo;
  float roughness;
  float metallic;
  float specular;
};

uniform Light light;

vec3 pbr(Material material, vec3 nrm, vec3 idir, vec3 odir) {
  float roughness = material.roughness;
  float metallic = material.metallic;
  float specular = material.specular;
  vec3 albedo = material.albedo;

  vec3 hdir = normalize(idir + odir);
  float NoH = max(0, dot(hdir, nrm));
  float NoL = max(0, dot(idir, nrm));
  float NoV = max(0, dot(odir, nrm));
  float VoH = clamp(dot(odir, hdir), 0, 1);
  float LoH = clamp(dot(idir, hdir), 0, 1);

  vec3 f0 = metallic * albedo + (1 - metallic) * 0.16 * specular * specular;
  vec3 fdf = f0 + (1 - f0) * pow(1 - VoH, 5);

  float k = (roughness + 1) * (roughness + 1) / 8;
  float vdf = 0.25 / ((NoV * k + 1 - k) * (NoL * k + 1 - k));

  float alpha2 = max(0, roughness * roughness);
  float denom = 1 - NoH * NoH * (1 - alpha2);
  float ndf = alpha2 / (denom * denom);

  vec3 brdf = fdf * vdf * ndf * f0 + (1 - f0) * albedo;
  return brdf * NoL;
}

void main()
{
  vec3 normal = normalize(iNormal);
  vec3 viewdir = vec3(0, 0, 1);
  Material material;
  material.albedo = vec3(1, 1, 1);
  material.roughness = 0.4;
  material.metallic = 0.0;
  material.specular = 0.5;
  vec3 lightDir = faceforward(light.dir, -light.dir, normal);
  vec3 strength = pbr(material, normal, lightDir, viewdir);
  vec3 color = light.color * strength;
  //color *= texture(ourTexture, texcoord).rgb;
  fColor = vec4(color, 1.0);
}
