#include "statements.h"
#include "ir.h"

int main() {
    IRBlock ir;
    ir.emplace_back<StmtConst>(0, 3.14f);
}
