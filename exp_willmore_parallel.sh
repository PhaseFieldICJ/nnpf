#!/usr/bin/env bash

# Script that tests different possibilities of depth/width of exp_willmore_parallel with constant number of repeatition

N="$1"
shift

script="exp_willmore_parallel.py"
remaining_args=()
scheme=DR
version_prefix=
dry_run=
default_root_dir="exp_willmore_parallel"
resume=

# Check some arguments
while [ "$#" -gt 0 ]
do
    case $1 in
        --scheme)
            scheme="$2"
            shift 2
            ;;

        --version)
            version_prefix="$2_"
            shift 2
            ;;

        --dry_run)
            dry_run=True
            shift
            ;;

        --default_root_dir)
            default_root_dir="$2"
            shift 2
            ;;

        --resume)
            resume=True
            shift
            ;;

        *)
            remaining_args=("${remaining_args[@]}" "$1")
            shift
            ;;
    esac
done

for i in $(seq "$N")
do
    layer=$((2 ** ($i - 1)))
    repeat=$((2 ** ($N - $i)))
    version="${version_prefix}${scheme}_l${layer}_r${repeat}"
    args=("${remaining_args[@]}" --scheme_layers "$layer" --scheme_repeat "$repeat" --scheme "$scheme" --version "$version" --default_root_dir "$default_root_dir")
    if [ "$resume" ]
    then
        args=("${args[@]}" --resume_from_checkpoint "${default_root_dir}/WillmoreParallel/${version}/checkpoints/last.ckpt")
    fi

    echo
    echo "python $script ${args[@]}"
    if [ ! "$dry_run" ]
    then
        python "$script" "${args[@]}"
    fi
done
