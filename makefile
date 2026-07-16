test:
	HYDRA_FULL_ERROR=1 python run_attacks.py \
	    model=microsoft/Phi-3-mini-4k-instruct \
	    dataset=adv_behaviors \
	    datasets.adv_behaviors.idx=2 \
	    attack=gcg \

multirun:
	HYDRA_FULL_ERROR=1 python run_attacks.py -m \
		    model=microsoft/Phi-3-mini-4k-instruct \
		    dataset=adv_behaviors \
		    datasets.adv_behaviors.idx="range(0,1)" \
		    attack=gcg \
		    hydra.launcher.timeout_min=240 \
		    hydra/launcher=submitit_local
