
  for (int i = 0; i < n; i++) {
    float r = img_bytes[i * 3 + 0] / 255.0f;
    float g = img_bytes[i * 3 + 1] / 255.0f;
    float b = img_bytes[i * 3 + 2] / 255.0f;
    img[i] = float4{r, g, b, 1.0f};
  }

  for (int i = 0; i < nx * ny; i++) {
    float4 clr = img_out[i];
    img_bytes[i * 3 + 0] = std::min(255, std::max(0, int(clr.x * 255.0f)));
    img_bytes[i * 3 + 1] = std::min(255, std::max(0, int(clr.y * 255.0f)));
    img_bytes[i * 3 + 2] = std::min(255, std::max(0, int(clr.z * 255.0f)));
  }
