#!/bin/bash
for f in /.singularity.d/env/*; do echo "$f";  source "$f"; done

export PS1="\u@\h \W:$"