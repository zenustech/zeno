# Contributing to Zeno

:+1::tada: First off, thanks for taking the time to contribute! :tada::+1:

Many people is using Zeno, if you find your wanted feature is missing, and you know how to implement it technically.
Please help us improve by submitting your code to us (so called contribute), so that other people could also enjoy this cool feature you made!

Contributions can be a one-line BUG fix, a new example graph, or even a README improvement, up to a brand new simulation algorithm.
We're very happy to see people utilizing Zeno and having fun in both using it and coding it!

# How to play with Git

## Get Zeno Source Code

Before start, we need to clone the source code of Zeno locally to your disk, to do so, you need [Git](https://git-scm.com/download/win).

After installing it, type the following command to `cmd` or `bash`:
```bash
git clone https://github.com/zenustech/zeno.git --depth=1
cd zeno
```

If you find GitHub too slow to download, you may clone from Gitee first (we will switch back to GitHub for creating PRs later):
```bash
git clone https://gitee.com/zenustech/zeno.git
cd zeno
```

## Create your own Fork

Wait, before you start to modifying the source code to implement your cool feature, please create your own 'fork' of Zeno on GitHub.
To do so, just click the 'Fork' button on the right-top of Zeno project main page.

After a few seconds, you will have something like `https://github.com/YOUR_NAME/zeno`, where `YOUR_NAME` is your user name.
Congrats! You've successfully created your own fork of Zeno! Now you can edit it's source code freely.

Now, let's get into the source code you just cloned, and set upstream of it to your own fork (so that you can push to it):
```bash
git remote set-url https://github.com/YOUR_NAME/zeno.git
```
> Don't forget to replace the `YOUR_NAME` to your user name :)

## Start Coding

Now you could start writting codes with your favorite editor and implement the feature, and possibly debug it.
After everying thing is done, let's push the change to your own fork hosted on GitHub:
```bash
git add .
git commit -m "This is my cool feature description"
git push -u origin master
```

## Create Pull Request (PR)

After push succeed, please head to `https://github.com/YOUR_NAME/zeno`, and there should be a 'Compare and Pull Request' button show there.
Click it, then click 'Create Pull Request', click again. The maintainers will see your work soon, they might have some talks with you, have fun!
After the changes are approved, the PR will be merged into codebase, and your cool feature will be available in next release for all Zeno users!

# Communication

Both Chinese and English are supported! Feel free to express your idea in your favorable language!

# How to build Zeno

Zeno is written in C++, which means we need a C++ developing environment to start coding in Zeno.

We support MSVC 2019 or GCC 9+ for compiler, and optionally install require packages via [vcpkg](https://github.com/microsoft/vcpkg).

You may check [BUILD.md](BUILD.md) for the complete build instructions of Zeno.

If you have trouble setting up developing environment, please let us help by opening an [issue](https://github.com/zenustech/zeno/issues)!

# Coding style

Zeno is based on C++17 and Python 3.7+, you may assume all their features are available in coding.

Code style is not forced, also we won't format every code merged in to the repo.

But it would be great if you could follow these simple rules for others to understand your code better and review faster:

```cpp
#include <vector>            // system headers should use '<>' brackets
#include <memory>
#include <tbb/parallel_for_each.h>
#include <zeno/zmt/log.h>    // project headers should also use '<>' for absolute pathing

// and never use 'using namespace std', think about std::data, std::size, std::ref

namespace zeno {   // '{' should stay in same line
                   // namespaces does not indent

namespace {        // use an anonymous namespace for static functions

auto staticFunc(int arg) {   // this function is visible only to this file
    std::vector<int> arr;        // tab size is 4
    for (auto const &x: arr) {   // C++11 range-based for loop when possible
       ZENO_LOG_INFO("{}", x);   // use zeno logger for printing
    }
    
    // ^^ may leave a blank line for logical separation
    tbb::parallel_for_each([&] (auto &x) {
        // tab size in lambda is also 4
        x = x * 2 + arg;  // leave two spaces between operators like '=', '*', '+'
    });
    
    return x;
}

}   // end of anonymous namespace

std::shared_ptr<types::MyType> globalFunc(int arg) {   // this function is visible globally
   auto ret = staticFunc(arg);
   auto ptr = std::make_shared<types::MyType>();  // use smart pointers instead of naked new/delete
   ptr->some_attr = std::move(ret);               // use std::move for optimal zero-copying
   return ptr;
}

}   // end of namespace zeno
```
