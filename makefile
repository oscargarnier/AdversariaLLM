CURRENT_TARGET_MODEL = meta-llama/Meta-Llama-3.1-8B-Instruct
gcg_sr:
	HYDRA_FULL_ERROR=1 python run_attacks.py \
	    model=$(CURRENT_TARGET_MODEL) \
	    dataset=adv_behaviors \
	    datasets.adv_behaviors.idx=2 \
	    attack=gcg \

defense_test:
	HYDRA_FULL_ERROR=1 python run_inference.py \
	    model=meta-llama/Meta-Llama-3.1-8B-Instruct \
	    dataset=adv_behaviors \
	    datasets.adv_behaviors.idx=2 \
	    attack=gcg \
	    runtime_defense=polyguard
	    

pair_sr:
	HYDRA_FULL_ERROR=1 python run_attacks.py \
	    model=meta-llama/Meta-Llama-3.1-8B-Instruct \
	    dataset=adv_behaviors \
	    datasets.adv_behaviors.idx=77 \
	    attack=pair \


gcg_reinforce_sr:
	HYDRA_FULL_ERROR=1 python run_attacks.py \
	    model=microsoft/Phi-3-mini-4k-instruct \
	    dataset=adv_behaviors \
	    datasets.adv_behaviors.idx=0 \
	    attack=gcg_reinforce \


gcg_multirun:
	HYDRA_FULL_ERROR=1 python run_attacks.py \
		    model=$(CURRENT_TARGET_MODEL)
		    dataset=adv_behaviors \
		    datasets.adv_behaviors.idx="range(0,3)" \
		    attack=gcg \



none:
		    hydra.launcher.timeout_min=240 \
		    hydra/launcher=submitit_local
