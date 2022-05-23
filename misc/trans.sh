#!/bin/bash

set -e
lupdate -recursive ui/zenoedit/ -ts ui/zenoedit/res/languages/zh.ts
linguist ui/zenoedit/res/languages/zh.ts
