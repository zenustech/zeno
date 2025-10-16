R"(#version 120

uniform mat4 mVP;
uniform mat4 mInvVP;
uniform mat4 mView;
uniform mat4 mProj;
uniform mat4 mInvView;
uniform mat4 mInvProj;

varying vec3 position;

void main() {
    gl_FragColor = vec4(0.0f, 1.0f, 1.0f, 1.0f);
}
)";