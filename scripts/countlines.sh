#!/bin/bash

wc -l `find zen* -type f -regex '.*\.\(cpp\|h\)'` <&-
wc -l `find python/ -type f -regex '.*\.\(py\)'` <&-
