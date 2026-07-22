from dataclasses import dataclass, field
from abc import abstractmethod
from datetime import time
from dataclasses import asdict
import json
import os
from typing import TYPE_CHECKING, Any, List, Union
import numpy as np
from adversariallm.lm_utils.tokenization import prepare_conversation
from torch import Tensor, cat
import transformers
from beartype import beartype
from beartype.typing import Literal, Optional, Generic, TypeVar

from ..dataset import PromptDataset
from ..types import Conversation
from ..io_utils.json_utils import CompactJSONEncoder
from ..lm_utils import generate_ragged_batched 
if TYPE_CHECKING:
    from ..defenses import TargetSystem


@dataclass
class GenerationConfig:
    generate_completions: Literal["all", "best", "last"] = "all"
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    num_return_sequences: int = 1


@beartype
@dataclass(kw_only=True)
class AttackStepResult:
    """Stores results for a single step of an attack algorithm."""
    step: int  # The step number (e.g., iteration)

    # The model's completion(s) in response to model_input. We use a list to store
    # multiple completions in case we are using distributional evaluation.
    model_completions: list[str]
    # Optional raw completions before defense transformation.
    model_completions_raw: Optional[list[str]] = None

    # Judge scores - should be a dict of judge name -> dict[str, list] mapping type to list of scores
    scores: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    # Time taken specifically for this step, per prompt (i.e. if we're running a batched
    # attack, this is step_time / batch_size).
    time_taken: float = 0.0

    # FLOPS, excluding all sampling from the model which is not necessary for the algorithm.
    # For example, GCG only samples from the model once after the optimization, so we do
    # not include sampling FLOPS. PAIR on the other hand relies on generations from the
    # model during the optimization process, so we do include sampling FLOPS for that.
    flops: Optional[int] = None

    # Optional fields, depending on the attack type
    # ---
    # Loss computed at this step (e.g., target loss for GCG)
    loss: Optional[float] = None

    # Sometimes we cannot provide these: (e.g., for embedding attacks)
    # The full input presented to the model at this step (e.g., original_prompt + optimized_suffix)
    model_input: Optional[Conversation] = None
    # Actual unrolled input tokens (i.e. including the system prompt, if present)
    model_input_tokens: Optional[list[int]] = None

    model_input_embeddings: Optional[Union[Tensor, str]] = None

    # Optional defense metadata (kept optional for backward compatibility).
    defense_metadata: Optional[list[dict[str, Any]]] = None

@beartype
@dataclass
class SingleInferenceOutput:
    """Stores the results of a single inference run on an attack artifact."""
    # The original multi-turn conversation from the dataset
    # We include the target response (usually `Sure, here's how to...`) as the
    # assistant message in the original conversation to make it easier to reproduce
    # results in the future.
    text_input: str 

    output: str
    # Results for each step of the attack
    target_config: Any

    # Total time taken for this entire attack run on a **single instance**
    total_time: float = 0.0

@beartype
@dataclass
class InferenceOutput:
    """Stores the results for inference run on an attack artifact

    Contains a list of SingleInferenceOutput objects, one for each instance
    in the dataset processed.
    """
    runs: list[SingleInferenceOutput] = field(default_factory=list)



@beartype
@dataclass
class SingleAttackRunResult:
    """Stores the results of running a single attack on a single conversation."""
    # The original multi-turn conversation from the dataset
    # We include the target response (usually `Sure, here's how to...`) as the
    # assistant message in the original conversation to make it easier to reproduce
    # results in the future.
    original_prompt: Conversation

    # Results for each step of the attack
    steps: list[AttackStepResult] = field(default_factory=list)

    # Total time taken for this entire attack run on a **single instance**
    total_time: float = 0.0


@beartype
@dataclass
class AttackResult:
    """Stores the results for all attack runs across a dataset.

    Contains a list of SingleAttackRunResult objects, one for each instance
    in the dataset processed.
    """
    runs: list[SingleAttackRunResult] = field(default_factory=list)

AttRes = TypeVar("AttRes", bound=AttackResult)

class Attack(Generic[AttRes]):
    def __init__(self, config):
        self.config = config
        transformers.set_seed(config.seed)

    def _jsonable(self, value: Any) -> Any:
        if isinstance(value, Tensor):
            return value.tolist()
        if isinstance(value, dict):
            return {str(key): self._jsonable(inner_value) for key, inner_value in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._jsonable(item) for item in value]
        return value

    def jailbreak_log(self, run: SingleAttackRunResult, storage_address: str, idx: int) -> str:
        """Write a compact jailbreak artifact for a single attack run.

        The artifact stores the original prompt, total run time, and the step with
        the minimum loss. The result is saved as ``idx_<number>.json`` inside the
        provided storage directory.
        """
        if not run.steps:
            raise ValueError("Cannot jailbreak_log a run with no steps")

        scored_steps = [
            (step_index, step)
            for step_index, step in enumerate(run.steps)
            if step.loss is not None
        ]
        if scored_steps:
            best_step_index, best_step = min(scored_steps, key=lambda item: item[1].loss)
        else:
            best_step_index = len(run.steps) - 1
            best_step = run.steps[best_step_index]

        artifact = {
            "original_prompt": self._jsonable(run.original_prompt),
            "total_time": run.total_time,
            "best_step_index": best_step_index,
            "best_step": self._jsonable(asdict(best_step)),
        }
        os.makedirs(storage_address, exist_ok=True)
        log_file = os.path.join(storage_address, f"idx_{int(idx)}.json")
        with open(log_file, "w") as handle:
            json.dump(artifact, handle, cls=CompactJSONEncoder)
        print(f"Stored results at adresss {log_file}")
        return log_file

    @classmethod
    def from_name(cls, name: str) -> type["Attack"]:
        match name:
            case "actor":
                from .actor import ActorAttack

                return ActorAttack
            case "autodan":
                from .autodan import AutoDANAttack

                return AutoDANAttack
            case "ample_gcg":
                from .ample_gcg import AmpleGCGAttack

                return AmpleGCGAttack
            case "beast":
                from .beast import BEASTAttack

                return BEASTAttack
            case "bon":
                from .bon import BonAttack

                return BonAttack
            case "crescendo":
                from .crescendo import CrescendoAttack

                return CrescendoAttack
            case "claudini":
                from .claudini import ClaudiniAttack

                return ClaudiniAttack
            case "direct":
                from .direct import DirectAttack

                return DirectAttack
            case "gcg":
                from .gcg import GCGAttack

                return GCGAttack
            case "gcg_reinforce":
                from .gcg_reinforce import GCGReinforceAttack

                return GCGReinforceAttack
            case "gcg_refusal":
                from .gcg_refusal import GCGRefusalAttack

                return GCGRefusalAttack
            case "human_jailbreaks":
                from .human_jailbreaks import HumanJailbreaksAttack

                return HumanJailbreaksAttack
            case "inpainting":
                from .inpainting import InpaintingAttack

                return InpaintingAttack
            case "jailbreak_r1":
                from .jailbreak_r1 import JailbreakR1Attack

                return JailbreakR1Attack
            case "pair":
                from .pair import PAIRAttack

                return PAIRAttack
            case "pgd":
                from .pgd import PGDAttack

                return PGDAttack
            case "pgd_discrete":
                from .pgd_discrete import PGDDiscreteAttack

                return PGDDiscreteAttack
            case "prefilling":
                from .prefilling import PrefillingAttack

                return PrefillingAttack

            case "random_search":
                from .random_search import RandomSearchAttack

                return RandomSearchAttack
            case _:
                raise ValueError(f"Unknown attack: {name}")


    def run_inference_batch(
            self,
            target: "TargetSystem",
            attack_artifacts: List[Conversation],
            run_config: RunConfig
    ):
        """Run inference with the successful attack artifact on a batch of conversations.

        This is used for runtime defenses, where we want to run inference on the
        original prompt and the attacked prompt, and compare the results.

        Args:
            target: The target system to attack.
            attack_artifacts: The list of attack artifacts to use for inference.

        Returns:
            The result of running inference on the attacked prompt.
        """
        # ===== Identify the best step for each run based on the loss =====
        text_inputs = []
        for run in attack_artifacts:
            losses = [step.loss for step in run.steps if step.loss is not None]
            best_loss_index = np.argmin(losses) if losses else None
            if best_loss_index is not None:
                best_step = run.steps[best_loss_index]
            else:
                print("No loss available for any step, using last step as fallback.")
                best_step = run.steps[-1]  # Fallback to the last step if no loss is available
            text_input = best_step.model_input[0]["content"]
            text_inputs.append(text_input)

        token_list = []
        # ===== Prepare the model inputs for inference =====
        for text_input in text_inputs:
            attack_conversation = [
                {"role": "user", "content": text_input},
                {"role": "assistant", "content": ""},
            ]
            ##TODO check if this call makes sense
            tokens = prepare_conversation(target.tokenizer, attack_conversation, attack_conversation)[0]
            ## Tokens outputs 6 elements, split into 5 for the prompt, and 1 for the assistant message. 
            ## We only want to sample the prompt, so we take the first 5 elements.
            token_list.append(cat(tokens[:5]))

        # ===== Generate completions =====
        batch_completions = generate_ragged_batched(
            target.model,
            target.tokenizer,
            token_list=token_list,
            initial_batch_size=len(token_list),
            max_new_tokens=self.config.generation_config.max_new_tokens,
            temperature=self.config.generation_config.temperature,
            top_p=self.config.generation_config.top_p,
            top_k=self.config.generation_config.top_k,
            num_return_sequences=self.config.generation_config.num_return_sequences,
        )  # (N_steps, N_return_sequences, T)

        #TODO Save experiment metadata in config
        outputs = []
        for i, attack_artifact in enumerate(attack_artifacts):
            single_output = SingleInferenceOutput(
                text_input=text_input,
                output=batch_completions[i][0],
                run_config=run_config
                total_time=0.0,  # Placeholder, can be updated with actual timing if needed
            )
            outputs.append(single_output)
        return outputs

    ## TODO make this runconfig typed, might introduce inmport loops 
    def run_inference(
        self,
        target: "TargetSystem",
        attack_artifacts: AttackResult,
        run_config = None
    ) -> InferenceOutput:
        """Run inference with the successful attack artifact

        """
        ## TODO add this to the config

        batch_size = 4
        outputs = []
        for i in range(0, len(attack_artifacts.runs), batch_size):
            batch_attack_artifacts= attack_artifacts.runs[i:i + batch_size]
            batch_outputs = self.run_inference_batch(target, batch_attack_artifacts, run_config)
            outputs.extend(batch_outputs)
        
        return InferenceOutput(outputs)

        
    
    @abstractmethod
    def run(
        self,
        target: "TargetSystem",
        dataset: PromptDataset,
    ) -> AttRes:
        """Run the attack against the supplied target system.

        Registry validation ensures attacks without runtime-defense support only
        receive the undefended implementation.
        """
        raise NotImplementedError
