#!/bin/bash

strace --follow-forks -- "$@" 2>&1 >/dev/null | grep '^open' | grep -vE '\) = -[1-9]' | cut -d\" -f2 | grep -E '^/(usr/)?(local/)?lib/(x86_64-[^/]*/)?[^/]*\.so' | sort -u
