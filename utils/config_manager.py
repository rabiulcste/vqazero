# Constants
BATCH_SIZE = "batch_size"
MAX_LENGTH = "max_length"
LENGTH_PENALTY = "length_penalty"
NO_REPEAT_NGRAM_SIZE = "no_repeat_ngram_size"
NUM_BEAMS = "num_beams"
NUM_CAPTIONS = "num_captions"
TEMPERATURE = "temperature"


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
            NUM_CAPTIONS: 1,
            MAX_LENGTH: 10,
            LENGTH_PENALTY: -1.0,
            NO_REPEAT_NGRAM_SIZE: 0,
            BATCH_SIZE: 64,
        }

    def _config_updater(func):
        """Decorator for update methods to simplify structure."""

        def wrapper(self, config_updates):
            return {**config_updates, **func(self)}

        return wrapper

    @_config_updater
    def _update_based_on_ans_parser(self):
        if self.args.vicuna_ans_parser:
            return {MAX_LENGTH: 30}
        return {}

    @_config_updater
    def _update_based_on_prompt_name(self):
        updates = {}
        if "rationale" in self.args.prompt_name and (
            "mixer" not in self.args.prompt_name or "iterative" not in self.args.prompt_name
        ):
            updates.update({MAX_LENGTH: 128, NO_REPEAT_NGRAM_SIZE: 3, BATCH_SIZE: 16})
        if "iterative" in self.args.prompt_name:
            updates.update({MAX_LENGTH: 10, LENGTH_PENALTY: -1.0})
        return updates

    @_config_updater
    def _update_based_on_model_name(self):
        updates = {}
        if "llava" in self.args.model_name:
            updates.update({BATCH_SIZE: 2})

        if "rationale" not in self.args.prompt_name:
            if "xxl" in self.args.model_name or "kosmos" in self.args.model_name or "redpajama" in self.args.model_name:
                updates[BATCH_SIZE] = 16
            if "llava" in self.args.model_name:
                updates.update({MAX_LENGTH: 30})
            if "open_flamingo" in self.args.model_name:
                updates.update({MAX_LENGTH: 30})
        return updates

    @_config_updater
    def _update_based_on_few_shot(self):
        if self.args.few_shot:
            return {BATCH_SIZE: 16}
        return {}

    @_config_updater
    def _update_based_on_vqa_format(self):
        if self.args.vqa_format == "caption_vqa":
            return {BATCH_SIZE: 16}
        return {}

    @_config_updater
    def _update_based_on_self_consistency(self):
        if self.args.self_consistency:
            return {NUM_BEAMS: 1, NUM_CAPTIONS: 30, TEMPERATURE: 0.7}
        return {}

    def update_configs(self) -> dict:
        config_updates = self._base_configuration()
        # config_updates = self._update_based_on_ans_parser(config_updates)
        config_updates = self._update_based_on_prompt_name(config_updates)
        config_updates = self._update_based_on_few_shot(config_updates)
        config_updates = self._update_based_on_vqa_format(config_updates)
        config_updates = self._update_based_on_model_name(config_updates)
        config_updates = self._update_based_on_self_consistency(config_updates)
        return config_updates

    def apply_updates_to_args(self):
        config_updates = self.update_configs()
        for key, value in config_updates.items():
            setattr(self.args, key, value)

        return self.args
