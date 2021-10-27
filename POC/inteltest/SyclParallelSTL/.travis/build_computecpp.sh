#!/bin/bash

set -ev

###########################
# Get ComputeCpp
###########################
wget https://computecpp.codeplay.com/downloads/computecpp-ce/latest/ubuntu-16.04-64bit.tar.gz
rm -rf /tmp/ComputeCpp-latest && mkdir /tmp/ComputeCpp-latest/
tar -xzf ubuntu-16.04-64bit.tar.gz -C /tmp/ComputeCpp-latest --strip-components 1
ls -R /tmp/ComputeCpp-latest/
