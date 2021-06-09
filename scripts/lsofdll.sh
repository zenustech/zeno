#!/bin/bash

lsof -p ${1?pid} | awk '{print $9}' | grep '\.so'
