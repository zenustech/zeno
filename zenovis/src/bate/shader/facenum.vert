R"(#version 120

uniform mat4 mVP;
uniform mat4 mInvVP;
uniform mat4 mView;
uniform mat4 mProj;
uniform mat4 mInvView;
uniform mat4 mInvProj;

uniform mat4 mScale;

attribute vec3 vPosition;

varying vec3 position;

void main() {
    position = vPosition;
    gl_Position = mVP * vec4(position, 1.0);
}
)";