#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include <stdio.h>

using namespace llvm;

Module *makeLLVMModule(LLVMContext &Context);

int main(int argc, char **argv)
{
  LLVMContext Context;
  Module *Mod = makeLLVMModule(Context);

  raw_fd_ostream r(fileno(stdout), false);
  verifyModule(*Mod, &r);

  //Prints the module IR
  ModulePass *m = createPrintModulePass(outs(), "Module IR printer");
  legacy::PassManager PM;
  PM.add(m);
  PM.run(*Mod);

   // Write IR to a bitcode file
  FILE* mul_add_file = fopen("mul_add.bc", "w+");
  raw_fd_ostream bitcodeWriter(fileno(mul_add_file), true);
  WriteBitcodeToFile(*Mod, bitcodeWriter);

  delete Mod;
  return 0;
}

Module *makeLLVMModule(LLVMContext &Context)
{
  Module *mod = new Module("mul_add", Context);

  FunctionCallee mul_add_fun = mod->getOrInsertFunction("mul_add",
      Type::getInt32Ty(Context),
      Type::getInt32Ty(Context),
      Type::getInt32Ty(Context),
      Type::getInt32Ty(Context));
  Function *mul_add = cast<Function> (mul_add_fun.getCallee());

  mul_add->setCallingConv(CallingConv::C);
  Function::arg_iterator args = mul_add->arg_begin();
  Value *x = args++;
  x->setName("x");
  Value *y = args++;
  y->setName("y");
  Value *z = args++;
  z->setName("z");

  BasicBlock *block = BasicBlock::Create(Context, "entry", mul_add);
  IRBuilder<> builder(block);
  Value *tmp = builder.CreateBinOp(Instruction::Mul, x, y, "tmp");
  Value *tmp2 = builder.CreateBinOp(Instruction::Add, tmp, z, "tmp2");
  builder.CreateRet(tmp2);

  return mod;
}
