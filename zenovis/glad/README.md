# `taichi_glad_ready`

Ready to use GLAD as a OpenGL API loader.

## Generation

If you encounter problems with glad, try run following scripts yourself:

```
pip install glad
python -m glad --out-path . --generator c --api gl=3.3 --profile core
```

### Tips

Try `--spec wgl` to specify API spec.

Try `--api gles=2.0` to specify API version.

Try `--extensions GL_ARB_compute_shader,GL_NV_shader_atomic_float` to specify extensions.

Try `--generator c-debug` to hook each API call and replace `check_opengl_errors`.

Also try https://glad.dav1d.de/ for online generation.
