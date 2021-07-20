; vim: ft=nasm

vaddps xmm0, xmm0, xmm0
vroundps xmm0, xmm0, 0
vmaxps xmm0, xmm0, xmm0
vminps xmm0, xmm0, xmm0
vmovss xmm0, xmm0
vmovups ymm0, ymm0
vmovups ymm0, ymm1
vmovups ymm1, ymm0
vaddps ymm2, ymm1, ymm1
vaddps ymm2, ymm0, ymm1
vmovups ymm2, ymm1
vmovups [rax], ymm0
vmovups ymm0, [rax]
vaddps xmm0, xmm0, xmm0
vfmadd231ps xmm0, xmm0, xmm0
vandps xmm0, xmm0, xmm0
vandnps xmm0, xmm0, xmm0
vorps xmm0, xmm0, xmm0
vxorps xmm0, xmm0, xmm0
pxor xmm0, xmm0
por xmm0, xmm0
pandn xmm0, xmm0
pand xmm0, xmm0
