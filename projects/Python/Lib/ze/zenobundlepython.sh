#!/bin/bash
exec "${zenoedit_executable?no env var named zenoedit_executable}" -invoke python "$@"
