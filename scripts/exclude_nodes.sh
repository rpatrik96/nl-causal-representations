#!/bin/bash
# Generate an exclude list of nodes for SLURM
#
# Usage:
#   exclude_nodes.sh sbatch [...]
#   sbatch [...] --exclude=$(exclude_nodes.sh) [...]
#
# Example raw output:
#   λ ~/ bash exclude_nodes.sh
#   bg-slurmb-bm-3,slurm-bm-05,slurm-bm-08,slurm-bm-32,slurm-bm-42,slurm-bm-47,slurm-bm-55,slurm-bm-60,slurm-bm-62,slurm-bm-82% 
# 
#
# Based on the MLCloud monitor at
#   http://134.2.168.207:3001/d/VbORxQJnk/slurm-monitoring-dashboard?orgId=1&from=now-3h&to=now&refresh=1m
#
# Up to date version of this script available at
#   https://gist.github.com/stes/52a139e260e25c72a97e2180d5be3bdb

get_exclude_list() {
    curl -XPOST '134.2.168.207:8085/api/v2/query?orgID=1a87e58d4b097066' -sS \
      -H 'Accept:application/csv' \
      -H 'Content-type:application/vnd.flux' \
      -H 'Authorization: Bearer 88JGdt2FvqcPXJwFQi5zGon6D0z7YP54' \
      -d 'from(bucket: "tueslurm")
                |> range(start: -35s)
                |> filter(fn: (r) => r["_measurement"] == "node_state")
                |> filter(fn: (r) => r["_field"] == "responding")
                |> filter(fn: (r) => r["state"] == "down" or r["state"] == "drained")
                |> filter(fn: (r) => r["_value"] == 0)
                |> group(columns: ["_time"])
                |> keep(columns: ["hostname"])'
}

formatted_exclude_list() {
    get_exclude_list \
    | tail -n +2 \
    | cut -f4 -d, \
    | tr -d '\r' \
    | sort | uniq \
    | sed -r '/^\s*$/d' \
    | tr '\n' ',' \
    | sed -e 's/,$//g'
}
echo $(formatted_exclude_list)
mode=$1
shift 1
case $mode in
  sbatch)
    sbatch --exclude=$(formatted_exclude_list) "$@"
    ;;
    
  *)
    formatted_exclude_list
    ;;
esac