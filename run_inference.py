import os
from datetime import datetime

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # determinism
import logging

import hydra
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

from adversariallm.attacks import Attack, AttackResult
from adversariallm.dataset import PromptDataset
from adversariallm.defenses import build_target_system, validate_defense_compatibility
from adversariallm.errors import print_exceptions
from adversariallm.io_utils import RunConfig, filter_config, free_vram, get_mongodb_connection, load_model_and_tokenizer, log_attack, filter_config_for_runtime, load_attack_results, log_inference, load_attack_results_from_jailbreak
from run_judges import run_judges

from bson import ObjectId
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cuda.matmul.allow_tf32 = True
torch._dynamo.config.cache_size_limit = 512 # needed for gemma 3 on AutoDAN

def select_configs(cfg: DictConfig, name: str | ListConfig | None) -> list[tuple[str, DictConfig]]:
    if name is not None:
        if isinstance(name, ListConfig):
            return [(n, cfg[n]) for n in name]
        return [(name, cfg[name])]
    return list(cfg.items())


def collect_configs(cfg: DictConfig) -> list[RunConfig]:
    models_to_run = select_configs(cfg.models, cfg.model)
    datasets_to_run = select_configs(cfg.datasets, cfg.dataset)
    attacks_to_run = select_configs(cfg.attacks, cfg.attack)
    attack_defenses_to_run = select_configs(cfg.defenses, cfg.defense)
    runtime_defenses_to_run = select_configs(cfg.runtime_defenses, cfg.runtime_defense)

    ## This should be skipped because the defense will not be run durring attack

    #for attack, _attack_params in attacks_to_run:
        #for _defense, defense_params in defenses_to_run:
            #validate_defense_compatibility(attack, defense_params)

    all_run_configs = []
    for model, model_params in models_to_run:
        for dataset, dataset_params in datasets_to_run:
            temp_dataset = PromptDataset.from_name(dataset)(dataset_params)
            dset_len = len(temp_dataset)
            dataset_params["idx"] = temp_dataset.config_idx
            for attack, attack_params in attacks_to_run:
                for attack_defense, attack_defense_params in attack_defenses_to_run:
                    for runtime_defense, runtime_defense_params in runtime_defenses_to_run:
                        run_config = RunConfig(
                            experiment_type="runtime_inference",
                            model=model,
                            dataset=dataset,
                            attack=attack,
                            defense=None if attack_defense == "none" else attack_defense,
                            model_params=model_params,
                            dataset_params=dataset_params,
                            attack_params=attack_params,
                            defense_params=None if attack_defense == "none" else attack_defense_params,
                            runtime_defense=None if runtime_defense == "none" else runtime_defense,
                            runtime_defense_params=None if runtime_defense == "none" else runtime_defense_params,
                        )
                        ## This serves two purposes:
                        ## Confirm that the original attack, in this scenario, has in fact been computed
                        ## Filter out any indices that have already been computed and logged to the database

                        run_config = filter_config_for_runtime(run_config, dset_len, overwrite=cfg.overwrite)

                        if run_config is not None:
                            all_run_configs.append(run_config)
    return all_run_configs



def run_inferences(all_run_configs: list[RunConfig], cfg: DictConfig, date_time_string: str) -> None:
    last_model = None
    last_dataset = None
    last_attack = None
    last_runtime_defense = None
    for run_config in all_run_configs:
        # To avoid reloading the model and dataset for every attack,
        # we only reload something if it's different from the last run
        if last_model != run_config.model:
            logging.info(f"Target: {run_config.model}\n{OmegaConf.to_yaml(run_config.model_params, resolve=True)}")
            last_model = run_config.model
            model, tokenizer = load_model_and_tokenizer(run_config.model_params)
        if last_dataset != run_config.dataset:
            logging.info(f"Dataset: {run_config.dataset}\n{OmegaConf.to_yaml(run_config.dataset_params, resolve=True)}")
            last_dataset = run_config.dataset
            dataset = PromptDataset.from_name(run_config.dataset)(run_config.dataset_params)
        if last_attack != run_config.attack:
            logging.info(f"Attack: {run_config.attack}\n{OmegaConf.to_yaml(run_config.attack_params, resolve=True)}")
            last_attack = run_config.attack
        if last_runtime_defense != run_config.runtime_defense:
            logging.info(f"Runtime defense: {run_config.runtime_defense}")
            last_runtime_defense = run_config.runtime_defense

        attack: Attack[AttackResult] = Attack.from_name(run_config.attack)(run_config.attack_params)

        target = build_target_system(
            run_config.runtime_defense_params,
            model=model,
            tokenizer=tokenizer,
        )

        ## This logic if loading from attack log files
        if False:
            attack_config = run_config.to_attack_config()
            if len(attack_config["dataset_params"]["idx"]) == 1:
                attack_config["dataset_params"]["idx"] = attack_config["dataset_params"]["idx"][0]
            attack_doc = get_mongodb_connection().runs.find_one({"config": attack_config})
            if attack_doc is None:
                raise ValueError("Source attack run was not found for runtime inference")

        ## This logic for loading from "jailbreak" log files

        attack_storage_adress = os.path.join(cfg.jailbreak_save_dir, run_config.model, run_config.dataset, run_config.attack, run_config.defense or 'none')

        attack_artifacts = load_attack_results_from_jailbreak(attack_storage_adress)
        results = attack.run_inference(target, attack_artifacts, run_config)  # type: ignore

        log_inference(run_config, results, cfg, date_time_string)



@hydra.main(config_path="./conf", config_name="config", version_base="1.3")
@print_exceptions
def main(cfg: DictConfig) -> None:
    os.makedirs(cfg.save_dir, exist_ok=True)
    date_time_string = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    logging.info("-------------------")
    logging.info(f"Commencing run at `{date_time_string}`")
    logging.info("-------------------")

    # 1. Parse/Collect configs for the judge and the attacks
    OmegaConf.set_struct(cfg, False)
    # Remove classifiers from config to avoid saving to mongodb
    # This way we don't re-run the attacks when only the classifiers change
    judges_to_run = cfg.pop('classifiers')
    OmegaConf.set_struct(cfg, True)
    all_run_configs = collect_configs(cfg)

    # 2. Run the attacks
    run_inferences(all_run_configs, cfg, date_time_string)
    # 3. Run the judges
    ## TODO setup judge running
    if False:
        if judges_to_run is None:
            return
        for judge in judges_to_run:
            # Create judge config
            judge_cfg = OmegaConf.create({
                "classifier": judge,
                "suffixes": [date_time_string.split('/')[-1]],  # Use the timestamp from this run to make sure we only judge this run
                "filter_by": None
            })
            free_vram()
            # Run the judge
            run_judges(judge_cfg)

if __name__ == "__main__":
    main()
