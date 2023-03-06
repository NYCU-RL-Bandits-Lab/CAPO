seed=$1
max_episode=$2
horizon=$3
size=$4
python3 main.py --algo ppo --policytype mlp --batch_size 1 --max_episode $max_episode --horizon $horizon --env Chain-v0  \
    --tbdir tb_chain/ppo --tbtag "seed $seed" --lr 0.01 --seed $seed --eps 0.3 --size $size --hidden 16
