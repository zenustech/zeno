#!/bin/bash

find "$1" -iname '*.h' -type f -exec sh -c '(head -n1 {} | grep -v "pragma\|ifndef") && echo -e "\033[31;1m"{}"\033[0m"' \;
