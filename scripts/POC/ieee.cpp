#include <cstdio>

int main() {
    union {
        float f;
        int i;
    } u;
    char buf[256];
    fgets(buf, 250, stdin);
    if (buf[0] == '0' && buf[1] == 'x') {
        sscanf(buf, "0x%X", &u.i);
        printf("%f\n", u.f);
    } else {
        sscanf(buf, "%f", &u.f);
        printf("0x%08X\n", u.i);
    }
    return 0;
}
