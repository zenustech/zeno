import os
import subprocess

from conans import ConanFile, tools, CMake
from conans.util import files

class QuickQanavaConan(ConanFile):
    name = "QuickQanava"
    version = "0.10.0"
    license = "BSD License 2.0, Copyright (c) 2018 Karl Wallner <kwallner@mail.de>"
    settings = "os", "compiler", "build_type", "arch"
    generators = "cmake"
    description = "QuickQanava: display graphs and relational content in a Qt application"
    author = "Karl Wallner <kwallner@mail.de>"
    url = 'https://github.com/kwallner/QuickQanava'
    scm = {
        "type": "git",
        "url": "auto",
        "revision": "auto"
    }
    no_copy_source = True

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()
        cmake.install()
        # Yet there are no tests
        #cmake.test()
