import torch

# Constants
BATCH_SIZE = "batch_size"
MAX_LENGTH = "max_length"
LENGTH_PENALTY = "length_penalty"
NO_REPEAT_NGRAM_SIZE = "no_repeat_ngram_size"
NUM_BEAMS = "num_beams"
NUM_RETURN_SEQUENCES = "num_return_sequences"
TEMPERATURE = "temperature"
DO_SAMPLE = "do_sample"


class VQAConfigManager:
    """
    Manages configurations for Visual Question Answering (VQA) inference.
    """

    def __init__(self, args):
        self.args = args

    def _base_configuration(self) -> dict:
        """
        Returns the base configuration.
        """
        return {
            NUM_BEAMS: 5,
            NUM_RETURN_SEQUENCES: 1,
            MAX_LENGTH: 10,
            LENGTH_PENALTY: -1.0,
            NO_REPEAT_NGRAM_SIZE: 0,
            DO_SAMPLE: False,
            TEMPERATURE: 1.0,
        }

    def _config_updater(func):
        """Decorator for update methods to simplify structure."""

        def wrapper(self, config_updates):
            return {**config_updates, **func(self)}

        return wrapper

    @_config_updater
    def _update_based_on_prompt_name(self):
        updates = {}
        if "rationale" in self.args.prompt_name and (
            "mixer" not in self.args.prompt_name or "iterative" not in self.args.prompt_name
        ):
            updates.update({MAX_LENGTH: 128, NO_REPEAT_NGRAM_SIZE: 3})
        if "iterative" in self.args.prompt_name:
            updates.update({MAX_LENGTH: 10, LENGTH_PENALTY: -1.0})
        return updates

    @_config_updater
    def _update_based_on_model_name(self):
        updates = {}
        if "llava" in self.args.model_name:
            updates.update({BATCH_SIZE: 2})
        else:
            updates[BATCH_SIZE] = 8

        if "rationale" not in self.args.prompt_name:
            if "llava" in self.args.model_name or "open_flamingo" in self.args.model_name or "kosmos" in self.args.model_name:
                updates.update({MAX_LENGTH: 30})

        if "minigpt4" in self.args.model_name:
            updates.update({MAX_LENGTH: 128})

        return updates

    @_config_updater
    def _update_based_on_self_consistency(self):
        if self.args.self_consistency:
            return {NUM_BEAMS: 1, NUM_RETURN_SEQUENCES: 30, TEMPERATURE: 0.7, DO_SAMPLE: True}
        return {}

    def _update_based_on_gpu_count(self, config_updates):
        gpu_count = torch.cuda.device_count()
        current_batch_size = config_updates.get(BATCH_SIZE, getattr(self.args, BATCH_SIZE, None))

        if current_batch_size is not None and gpu_count > 1:
            config_updates[BATCH_SIZE] = gpu_count * current_batch_size
        return config_updates

    def update_configs(self) -> dict:
        config_updates = self._base_configuration()
        config_updates = self._update_based_on_prompt_name(config_updates)
        config_updates = self._update_based_on_model_name(config_updates)
        config_updates = self._update_based_on_self_consistency(config_updates)
        config_updates = self._update_based_on_gpu_count(config_updates)
        return config_updates

    def apply_updates_to_args(self):
        config_updates = self.update_configs()
        for key, value in config_updates.items():
            setattr(self.args, key, value)

        return self.args
