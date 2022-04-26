#!/bin/bash

wc -l `find "$1" -type f -regex '.*\.\(cpp\|h\|py\)'` <&-
