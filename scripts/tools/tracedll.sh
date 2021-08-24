#!/bin/bash
set -e

rm -rf /tmp/strace.txt*
strace -ff -o /tmp/strace.txt -- "$@" >&2
cat /tmp/strace.txt* | grep -E '^openat\(' | grep -vE '\) = -[1-9]' | cut -d\" -f2 | grep -E '^/(usr/)?(local/)?lib/(x86_64-[^/]*/)?[^/]*\.so' | sort -u
rm -rf /tmp/strace.txt*
