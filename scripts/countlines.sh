#!/bin/bash

wc -l `find zen* Projects/zen* -type f -regex '.*\.\(cpp\|h\)'` <&-
wc -l `find zen* -type f -regex '.*\.\(py\)'` <&-
