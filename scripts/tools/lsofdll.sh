#!/bin/bash

lsof -p ${1?pid} | awk '{print $9}' | grep -E '^/(usr/)?(local/)?lib/(x86_64-[^/]*/)?[^/]*\.so'
