#include <zeno/zeno.h>
#include "program.h"
#include "pwrangle.h"
#include "parse.h"
#include <iostream>

using std::cout;
using std::endl;

int main(void)
{
    Context ctx;

    Program prog = parse_program(
        "add @0 @0 #3.14\n"
    );

    std::vector<float> arr(16);
    for (int i = 0; i < 8; i++) {
        arr[i] = 2.718f;
    }
    vectors_wrangle(prog, {&arr});

    for (int i = 0; i < 16; i++) {
        cout << arr[i] << endl;
    }

    return 0;
}
