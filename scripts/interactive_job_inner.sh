#!/bin/bash 
set -xe 
 
HOSTNAME=$(hostname | sed s/.novalocal//) 
echo "Running on $HOSTNAME" 
 
# update ssh target "interactive" 
# this requires you to add an entry 
#     Include config.d/interactive 
# to your ~/.ssh/config 
mkdir -p ~/.ssh/config.d 
echo "Host interactive" > ~/.ssh/config.d/interactive 
echo "  Hostname $HOSTNAME" >> ~/.ssh/config.d/interactive 
echo "  User preizinger" >> ~/.ssh/config.d/interactive 
 
SESSION=preizinger-interactive 
 
# reset $TMPDIR. In SLURM-jobs by default it would be 
# TMPDIR="/scratch_local/$(whoami)-${SLURM_JOB_ID}/tmp" 
# but this setting is lost when sshing into the node. 
# Therefore, we unset $TMPDIR already here. 
unset TMPDIR
#make sure we can run this script from within tmux 
TMUX=0 
tmux new-session -d -s $SESSION
tmux send-keys 'echo session started' C-m
# Build correct address parameters. 
# Unfortunately, dropbear seems to listen only on IPv6 
# since recently. I haven't found the reason yet, but it's 
# easiest to just specify the correct addresses. 
ADDRESSES='' 
for IP in $(hostname -I) 
do 
        ADDRESSES="$ADDRESSES -p $IP:12345" 
done 
# -p is for making sure sshd is killed if singularity is stopped 
# - R is for generating key
tmux send-keys "./scripts/run_singularity_server.sh /usr/sbin/dropbear -R -E -F $ADDRESSES -s" C-m
 
while true; do 
        # will fail if session ended 
        tmux has-session -t $SESSION 
        echo "session still running" 
        sleep 10 
done