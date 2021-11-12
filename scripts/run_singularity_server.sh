#!/bin/bash 

IMAGE=./scripts/nv.sif

userName=preizinger 
tmp_dir=$(mktemp -d -t singularity-home-XXXXXXXX) 
echo "$tmp_dir" 

# singularity doesn't like the image to be modified while it's running.
# also, it's faster to have it on local storage
echo "copy image" 
LOCAL_IMAGE=$tmp_dir/nv.sif 
rsync -av --progress $IMAGE "$LOCAL_IMAGE" 
 

mkdir -p "$tmp_dir"/.vscode-server 
mkdir -p "$tmp_dir"/.conda 
mkdir -p "$tmp_dir"/.ipython 
mkdir -p "$tmp_dir"/.jupyter 
mkdir -p "$tmp_dir"/.local 
mkdir -p "$tmp_dir"/.pylint.d 
mkdir -p "$tmp_dir"/.cache 
 
singularity exec -p --nv \
        --bind "$tmp_dir"/.vscode-server:/home/bethge/$userName/.vscode-server \
        --bind "$tmp_dir"/.conda:/home/bethge/$userName/.conda \
        --bind "$tmp_dir"/.ipython:/home/bethge/$userName/.ipython \
        --bind "$tmp_dir"/.jupyter:/home/bethge/$userName/.jupyter \
        --bind "$tmp_dir"/.local:/home/bethge/$userName/.local \
        --bind "$tmp_dir"/.pylint.d:/home/bethge/$userName/.pylint.d \
        --bind "$tmp_dir"/.cache:/home/bethge/$userName/.cache \
        --bind /scratch_local \
        --bind /mnt/qb/bethge \
        "$LOCAL_IMAGE" "$@"
 
rm -rf "$tmp_dir"
