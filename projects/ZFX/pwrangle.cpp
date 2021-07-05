#include "program.h"
#include "pwrangle.h"

void vectors_wrangle(Program const *prog,
    std::vector<std::vector<float> *> const &arrs) {
    if (arrs.size() == 0)
        return;
    size_t size = arrs[0]->size();
    for (int i = 1; i < arrs.size(); i++) {
        size = std::min(arrs[i]->size(), size);
    }
    Context ctx;
    for (int i = 0; i < arrs.size(); i++) {
        ctx.memtable[i] = arrs[i]->data();
    }
    for (int i = 0; i < size; i++) {
        ctx.memindex = i;
        prog->execute(&ctx);
    }
}

void particles_wrangle(Program const *prog, Particles const *pars) {
    std::vector<std::vector<float> *> arrs(prog->channels.size());
    for (int i = 0; i < arrs.size(); i++) {
        auto const &chname = prog->channels[i];
        auto chid = pars->chid_of_name(chname);
        auto parr = &pars->channel(chid);
        arrs[i] = parr;
    }
    vectors_wrangle(prog, arrs);
}
