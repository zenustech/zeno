; vim: ft=nasm

vblendvps xmm0, xmm0, xmm0, xmm0
vblendvps xmm15, xmm0, xmm0, xmm0
vblendvps xmm0, xmm15, xmm0, xmm0
vblendvps xmm0, xmm0, xmm15, xmm0
vblendvps xmm0, xmm0, xmm0, xmm15
