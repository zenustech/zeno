; vim: ft=nasm

vaddpd xmm15, xmm0, xmm15
vsqrtpd xmm15, xmm15
vaddpd xmm0, xmm0, xmm15
