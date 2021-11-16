# Contributing to Zeno

:+1::tada: First off, thanks for taking the time to contribute! :tada::+1:

Contributions to Zeno are always welcome. Contributions can take many
forms, such as:

* Raising, responding to, or reacting to Issues or Pull Requests
* Testing new in-progress changes and providing feedback
* Discussing in the [GitHub discussion channel](https://github.com/zenustech/zeno/discussions)
* etc.

Both Chinese and English are supported! Feel free to express your idea in your favorable language!

# Issues

If you have

* found a bug in Zeno and would like to report (bug report)
* a cool feature you'd like to have in Zeno (feature request)
* any question on how to use Zeno (question)

Please feel free to check out [this page](https://github.com/zenustech/zeno/issues) for raising GitHub issues!

Make sure you choose the right *issue template* accordingly to your issue type, like `Bug Report`.
Please take some time to fill the template completely, this help us understand your problem easier.
Therefore *you get a better answer, faster*, thanks for your support!

For bug reports, the *console output* is very helpful to us, make sure you attach it if possible.

Hint: you may use triple back quotes to insert pretty-shown code (for console output) like this:

```md
\`\`\`
This is the Zeno log...
\`\`\`
```

## Discussions

If you feel issues being too formal, check out our [GitHub discussion channel](https://github.com/zenustech/zeno/discussions)
for discussing problems in Q&A style.

## Usage Tips

The GitHub issue tracker is for *bug reports* and *features requests* for the
Zeno project, and on-topic comments and follow-ups to them. It is not for
general discussion, general support or for any other purpose.

Please **search the issue tracker for similar issues** before creating a new
one. There's no point in duplication; if an existing open issue addresses your
problem, please comment there instead of creating a duplicate. However, if the
issue you found is **closed as resolved** (e.g. with a PR or the original user's
problem was resolved), raise a **new issue**, because you've found a new
problem. Reference the original issue if you think that's useful information.

Closed issues which have been inactive for 60 days will be locked, this helps to
keep discussions focussed. If you believe you are still experiencing an issue
which has been closed, please raise a new issue, completing the issue template.

If you do find a similar _open_ issue, **don't just post 'me too' or similar**
responses. This almost never helps resolve the issue, and just causes noise for
the maintainers. Only post if it will aid the maintainers in solving the issue;
if there are existing diagnostics requested in the thread, perform
them and post the results.

Please do not be offended if your Issue or comment is closed or hidden, for any
of the following reasons:

* The issue template was not completed
* The issue or comment is off-topic
* The issue does not represent a Zeno bug or feature request
* The issue cannot be reasonably reproduced using the minimal vimrc
* The issue is a duplicate of an existing issue
* etc.

Issue titles are important. It's not usually helpful to write a title like
`bug report` or `issue with Zeno` or even pasting an error message.
Spend a minute to come up with a consise summary of the problem. This helps with
management of issues, with triage, and above all with searching.

But above all else, please *please* complete the *issue template*. I know it is a
little tedious to get all the various diagnostics, but you *must* provide them,
*even if you think they are irrelevant*. This is important, because the
maintainer(s) can quickly cross-check theories by inspecting the provided
diagnostics without having to spend time asking for them, and waiting for the
response. This means *you get a better answer, faster*. So it's worth it,
honestly.

<!--
### Reproduce your issue with the minimal graph

Many problems can be caused by unexpected configuration or other plugins.
Therefore when raising an issue, you must attempt to reproduce your issue
with the minimal vimrc provided, and to provide any additional changes required
to that file in order to reproduce it. The purpose of this is to ensure that
the issue is not a conflict with another plugin, or a problem unique to your
configuration.

If your issue does _not_ reproduce with the minimal vimrc, then you must say so
in the issue report.
-->

# Pull Requests

Zeno is open to all contributors with ideas great and small! However,
there is a limit to the intended scope of the plugin and the amount of time the
maintainer has to support and... well... maintain features. It's probably well
understood that the contributor's input typically ends when a PR is megred, but
the maintainers have to keep it working forever.

## Small changes

For bug fixes, documentation changes, gadget versin updates, etc. please just
send a PR, I'm super happy to merge these!

If you are unsure, or looking for some pointers, feel free to ask in Gitter, or
mention is in the PR.

## Larger changes

For larger features that might be in any way controvertial, or increase the
complexity of the overall plugin, please and talk to the maintainer(s) via [GitHub
discussions](https://github.com/zenustech/zeno/discussions) first. This saves a lot of
potential back-and-forth and makes sure that we're "on the same page" about the
idea and the ongoing maintenance.

In addition, if you like hacking, feel free to raise a PR tagged with `[RFC]` in
the title and we can discuss the idea. I still prefer to discuss these things on
Gitter rather than back-and-forth on GitHub, though.

Please don't be offended if the maintainer(s) request significant rework for (or
perhaps even dismiss) a PR that's not gone through this process.

Please also don't be offended if the maintainer(s) ask if you're willing to
provide ongoing support for the feature. As an OSS project manned entirely in
what little spare time the maintainer(s) have, we're always looking for
contributions and contributors who will help with support and maintenance of
larger new features.

## PR Guidelines

When contributing pull requests for Zeno, I ask that:

* You provide a clear and complete summary of the change, the use case and how the change was tested.
* You follow the style of the code as-is; ref: [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).
* Your changes worked on both Linux and Windows (GitHub CI will automatically check this).

# Coding Guide

Many people is using Zeno, if you find your wanted feature is missing, and you know how to implement it technically.
Please help us improve by submitting your code to us (so called contribute), so that other people could also enjoy this cool feature you made!

Contributions to codebase can be a one-line BUG fix, a new example graph, or even a README improvement, up to a brand new simulation algorithm.
We're very happy to see people utilizing Zeno and having fun in both using it and coding it!

## How to play with Git

### Get Zeno Source Code

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

### Create your own Fork

Wait, before you start to modifying the source code to implement your cool feature, please create your own 'fork' of Zeno on GitHub.
To do so, just click the 'Fork' button on the right-top of Zeno project main page.

After a few seconds, you will have something like `https://github.com/YOUR_NAME/zeno`, where `YOUR_NAME` is your user name.
Congrats! You've successfully created your own fork of Zeno! Now you can edit it's source code freely.

Now, let's get into the source code you just cloned, and set upstream of it to your own fork (so that you can push to it):
```bash
git remote set-url https://github.com/YOUR_NAME/zeno.git
```
> Don't forget to replace the `YOUR_NAME` to your user name :)

### Start Coding

Now you could start writting codes with your favorite editor and implement the feature, and possibly debug it.
After everying thing is done, let's push the change to your own fork hosted on GitHub:
```bash
git add .
git commit -m "This is my cool feature description"
git push -u origin master
```

> May also check out [this post for Git tutorial](https://www.liaoxuefeng.com/wiki/896043488029600/896067008724000).

### Create Pull Request (PR)

After push succeed, please head to `https://github.com/YOUR_NAME/zeno`, and there should be a 'Compare and Pull Request' button show there.
Click it, then click 'Create Pull Request', click again. The maintainers will see your work soon, they might have some talks with you, have fun!
After the changes are approved, the PR will be merged into codebase, and your cool feature will be available in next release for all Zeno users!

## How to build Zeno

Zeno is written in C++, which means we need a C++ developing environment to start coding in Zeno.

We support MSVC 2019 or GCC 9+ for compiler, and optionally install require packages via [vcpkg](https://github.com/microsoft/vcpkg).

Please check the [building.md](building.md) for the complete build instructions of Zeno.

If you have trouble setting up developing environment, please let us help by opening an [issue](https://github.com/zenustech/zeno/issues)!

### Code Style

Zeno is based on C++20, you may assume all C++17 features and some C++20 features to be available in coding.

> Modules and coroutines are very nice feature in C++20, but we ain't using them for now due to compiler support.
> As an oppisite, concepts and template-lambdas are widely used in the codebase.

We generally follows the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html), excepts a few differences:

- We target C++20 instead of C++17.
- We use `T const &t` instead of `const T& t`
- We allow implicit conversions when used properly
- We allow exceptions as we widely adopted smart pointers for exception-safety
- We avoid using STL streams (`std::cout` and so on) -- use `zmt::format` instead
- We don't add `Copyright blah blah` to codebase since we're programmers, not lawyers
- We mainly use `smallCamelCase` for function methods, `underline_case` for variables
- We don't add trialling underscope like `m_google_style_`, we use `m_zeno_style`

Code style is not forced, we also won't format every code merged in to the repo.

But it would be great if you could follow them for others to understand your code better and review your PR faster.

Example:
```cpp
#include <vector>            // system headers should use '<>' brackets
#include <memory>
#include <tbb/parallel_for_each.h>
#include <zeno/zmt/log.h>    // project headers should also use '<>' for absolute pathing
#include <zeno/zmt/format.h>

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

    std::string expr = zmt::format("the answer is {}", 42);  // instead of fmt or std
    
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

## Write a custom node

See [zenustech/zeno_addon_wizard](https://github.com/zenustech/zeno_addon_wizard) for an example on how to write custom nodes in ZENO.

# Code of conduct

Please see [code of conduct](CODE_OF_CONDUCT.md).
