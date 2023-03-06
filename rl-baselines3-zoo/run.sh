for seed in {1..10}
do
    # python3 train.py --algo ppo --env asterix -min -n 1000000 --seed $seed
    python3 train.py --algo offpac -params KL:True --env MinAtar/Freeway-v0 --tensorboard tensorboard_5/ --seed $seed
    # python3 train.py --algo offpac --env space_invaders -min -n 3000000 -params KL:True --seed $seed 
    # python3 train.py --algo ppo --env asterix -min -n 3000000 --seed $seed > /dev/null
    # python3 train.py --algo offpac --env asterix -min -n 1000000 --seed $(($seed+1)) -params KL:True > /dev/null &

	# python3 train.py --algo ppo --env freeway -min -n 1000000 --seed $seed
done