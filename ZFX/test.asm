; vim: ft=nasm

vaddps xmm0, xmm0, xmm0
vblendvps xmm0, xmm0, xmm0, xmm0
vblendvps xmm1, xmm0, xmm0, xmm0
vblendvps xmm0, xmm1, xmm0, xmm0
vblendvps xmm0, xmm0, xmm1, xmm0
vblendvps xmm0, xmm0, xmm0, xmm1
