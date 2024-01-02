import torch

from transformers import DecisionTransformerModel
from transformers.modeling_outputs import BaseModelOutput
from models.modeling_mamba import MixerModel

from mamba_ssm.utils.generation import InferenceParams, update_graph_cache

class TrainableDT(DecisionTransformerModel):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, **kwargs):
        output = super().forward(**kwargs)
        # add the DT loss
        action_preds = output[1]
        action_targets = kwargs["actions"]
        attention_mask = kwargs["attention_mask"]
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_targets = action_targets.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = torch.mean((action_preds - action_targets) ** 2)

        return {"loss": loss}

    def original_forward(self, **kwargs):
        return super().forward(**kwargs)

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, max_length = None):
        # This implementation does not condition on past rewards

        # TODO: will have to change this to deal with batched inputs
        states = states.reshape(1, -1, self.config.state_dim)
        actions = actions.reshape(1, -1, self.config.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        max_length = self.config.max_length if max_length is None else max_length

        states = states[:, -max_length:]
        actions = actions[:, -max_length:]
        returns_to_go = returns_to_go[:, -max_length:]
        timesteps = timesteps[:, -max_length:]
        padding = max_length - states.shape[1]

        # pad all tokens to sequence length
        attention_mask = torch.cat([torch.zeros(padding), torch.ones(states.shape[1])])
        attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
        states = torch.cat([torch.zeros((1, padding, self.config.state_dim), device=states.device), states], dim=1).float()
        actions = torch.cat([torch.zeros((1, padding, self.config.act_dim), device=actions.device), actions], dim=1).float()
        returns_to_go = torch.cat([torch.zeros((1, padding, 1), device=returns_to_go.device), returns_to_go], dim=1).float()
        timesteps = torch.cat([torch.zeros((1, padding), dtype=torch.long, device=timesteps.device), timesteps], dim=1)

        state_preds, action_preds, return_preds = self.original_forward(
            states=states,
            actions=actions,
            rewards=rewards,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            attention_mask=attention_mask,
            return_dict=False,
        )

        return action_preds[0, -1] # TODO: will have to change this to deal with batched inputs


class TrainableDM(TrainableDT):
    def __init__(self, config):
        super().__init__(config)

        self.mamba = MixerModel(
            d_model=config.hidden_size,
            n_layer=6,
            ssm_cfg={},
            rms_norm=True,
            residual_in_fp32=True,
            fused_add_norm=True,
            # dtype=torch.bfloat16,
            # device='cuda',
        )

        del self.encoder  # get rid of the GPT-2 model
        self.encoder = self._encoder_forward

        self.is_recurrent = False
        self.inference_params = None
        self.last_action = None
        self.last_timestep = None

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, max_length=None):
        action = super().get_action(states, actions, rewards, returns_to_go, timesteps, max_length=max_length)

        if self.is_recurrent:
            self.last_action = action # TODO: will have to change this to deal with batched inputs
            self.last_timestep = timesteps[-1].reshape(1, 1)

        return action

    def _encoder_forward(self, inputs_embeds, *args, **kwargs):
        if self.is_recurrent:
            hidden_states = self._recurrent_forward(inputs_embeds)
        else:
            hidden_states = self.mamba(inputs_embeds)

        return BaseModelOutput(last_hidden_state=hidden_states)

    def recurrent(self, batch_size):
        self.is_recurrent = True
        self.inference_params = InferenceParams(max_seqlen=self.config.max_length, max_batch_size=batch_size)

    def reset_cache(self):
        self.last_action = None
        self.last_timestep = None
        self.inference_params = InferenceParams(max_seqlen=self.config.max_length, max_batch_size=self.inference_params.max_batch_size)

    def _recurrent_forward(self, inputs_embeds):
        # for recurrent generation, inputs embeds should look like: [[RTG_i, S_i, A_null], ...] and have the shape [batch_size, 3, hidden_size]...
        # ... A_null is there b/c the original DT code requires *something* there for an action (even though it isn't even factored into the computation), so we can just skip it
        inputs_embeds = inputs_embeds[:, 1:] # now it is [[RTG_i, S_i], ...], shape [batch_size, 2, hidden_size]

        # if we are at the beginning of the sequence, we just run the model through long the original sequence is
        if self.inference_params.seqlen_offset == 0:
            hidden_states = self.mamba(inputs_embeds, inference_params=self.inference_params) # shape [batch_size, 2, hidden_size]
            hidden_states = hidden_states[:, 1:] # shape [batch_size, 1, hidden_size] to match when we are not at the beginning of the sequence
            self.inference_params.seqlen_offset += hidden_states.shape[1]
        else:
            if self.last_action is not None:
                # encode the action
                time_embeddings = self.embed_timestep(self.last_timestep)
                action_embedding = self.embed_action(self.last_action)
                action_embedding = action_embedding + time_embeddings

                # prepend the last action to the inputs_embeds
                inputs_embeds = torch.cat([action_embedding, inputs_embeds], dim=1)  # now it is [[A_{i-1}, RTG_i, S_i], ...], shape [batch_size, 3, hidden_size]

            for i in range(inputs_embeds.shape[1] - 1):
                hidden_states = self.mamba(inputs_embeds[:, i:i+1], inference_params=self.inference_params) # shape [batch_size, 1, hidden_size]
                self.inference_params.seqlen_offset += hidden_states.shape[1]

        # the DT expects hidden_states to be shaped like [batch_size, 3, hidden_size] where the action preds are in [:, 1, :] and the state/reward preds are in [:, 2, :], so we just repeat
        hidden_states = hidden_states.repeat(1, 3, 1)

        return hidden_states

    def parallel(self):
        self.is_recurrent = False
