squeue --me | awk 'NR>2 {print $1}' | xargs scancel