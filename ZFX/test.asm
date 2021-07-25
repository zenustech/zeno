; vim: ft=nasm

vblendvpd ymm0, ymm0, ymm0, ymm0
vblendvpd ymm15, ymm0, ymm0, ymm0
vblendvpd ymm0, ymm15, ymm0, ymm0
vblendvpd ymm0, ymm0, ymm15, ymm0
vblendvpd ymm0, ymm0, ymm0, ymm15
