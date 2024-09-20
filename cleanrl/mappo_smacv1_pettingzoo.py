# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass
from typing import Literal

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from smac_pettingzoo import smacv1_pettingzoo_v1
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases (wandb)"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""

    # Env specific arguments
    env_id: str = "3m"
    """the id of the environment"""

    # Algorithm specific arguments
    total_steps: int = 1e7
    """total timesteps of the experiments"""
    learning_rate: float = 5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 32
    """the number of parallel game environments"""
    num_eval_envs: int = 1
    """the number of parallel game environments"""
    num_steps_per_episode: int = 200
    """the number of steps to run in each environment per policy rollout (each episode)"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 1
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 10.0
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    ## Network specific arguments
    share_policy: bool = True
    """if toggled, the policy network will be shared among agents"""
    hidden_size: int = 64
    """the hidden size of the network"""
    num_layers: int = 2
    """the number of layers of the network"""
    num_layers_after_rnn: int = 1
    """the number of layers after the RNN layer"""
    use_recurrent_policy: bool = True
    """if toggled, use a recurrent policy"""
    num_recurrent_layers: int = 1
    """the number of recurrent layers"""
    data_chunk_length: int = 10
    """the length of the data chunks to train a recurrent_policy"""
    use_value_norm: bool = True
    """if toggled, use running mean and std to normalize the value function outputs"""
    use_feature_norm: bool = True
    """if toggled, use layernorm to normalize the features"""
    action_aggregation: Literal["mean", "prod"] = "mean"
    """the action aggregation method"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id: str) -> callable:
    """
    Create a function that will create a parallelenv pettingzoo environment
    """

    def thunk():
        env = smacv1_pettingzoo_v1.parallel_env(env_id, {})
        return env

    return thunk


def layer_init(layer: nn.Module, bias_const: float = 0.0) -> nn.Module:
    """
    Initialize the layers with the orthogonal initialization
    """
    gain = nn.init.calculate_gain("relu")
    torch.nn.init.orthogonal_(layer.weight, gain)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class MLPBase(nn.Module):
    """
    The base class for the MLP network:
    ----------
    # Input Layer
    Input LayerNorm
    Input MLP
    Input Layer LayerNorm
    ReLU
    ----------
    # Hidden Layers
    Hidden MLP
    Hidden LayerNorm
    ReLU
    ----------
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        layers = []

        layers.append(nn.LayerNorm(input_size))
        layers.append(layer_init(nn.Linear(input_size, hidden_size)))
        layers.append(nn.LayerNorm(hidden_size))
        layers.append(nn.ReLU())

        # hidden layers
        for _ in range(num_layers):
            layers.append(layer_init(nn.Linear(hidden_size, hidden_size)))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.tensor):
        return self.mlp(x)


class ActLayer(nn.Module):
    def __init__(self, input_size: int, num_layers: int, act_space: gym.Space):
        """
        :param num_layers: (int) the number of hidden layers
        """
        super().__init__()
        assert isinstance(act_space, gym.spaces.Discrete), "only discrete action space is supported in smacv1"
        action_size = act_space.n

        layers = []
        for _ in range(num_layers):
            layers.append(layer_init(nn.Linear(input_size, input_size)))
            layers.append(nn.LayerNorm(input_size))
            layers.append(nn.ReLU())

        layers.append(layer_init(nn.Linear(input_size, action_size)))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.tensor):
        return self.network(x)


class Actor(nn.Module):
    def __init__(
        self,
        act_space: gym.Space,
        obs_space: gym.Space,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_layers_after_rnn: int = 1,
        use_recurrent_policy: bool = True,
        num_recurrent_layers: int = 1,
    ):
        super().__init__()
        input_size = np.array(obs_space.shape).prod()
        self.mlp = nn.Sequential(MLPBase(input_size, hidden_size, num_layers))
        if use_recurrent_policy:
            self.rnn = nn.GRU(hidden_size, hidden_size, num_recurrent_layers, batch_first=True)
            for name, param in self.rnn.named_parameters():
                if "bias" in name:
                    nn.init.constant_(param, 0)
                elif "weight_ih_l" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "weight_hh_l" in name:
                    hidden_size = param.shape[1]
                    for i in range(3):
                        start = i * hidden_size
                        end = (i + 1) * hidden_size
                        weight = param.data[start:end]
                        if i == 0 or i == 1:
                            nn.init.orthogonal_(weight, gain=1)
                        else:
                            nn.init.orthogonal_(weight, gain=nn.init.calculate_gain("tanh"))
            self.rnn_layer_norm = nn.LayerNorm(hidden_size)
            self.actlayer = ActLayer(hidden_size, num_layers_after_rnn, act_space)

    def get_hidden_state(
        self,
        x: torch.tensor,
        rnn_state: torch.tensor,
        mask: torch.tensor,
    ) -> tuple[torch.tensor, torch.tensor]:
        """
        :param x: (torch.tensor) input tensor, size [batch_size, obs_shape]
        :param rnn_state: (torch.tensor) rnn hidden state, size [batch_size, hidden_size]
        :param mask: (torch.tensor) mask for rnn state, 0 if the sequence is done, 1 otherwise, size [batch_size]

        :return: (tuple) hidden state and rnn state
        """
        x = self.mlp(x)
        if self.use_recurrent_policy:
            mask = mask.view(-1, 1).type_as(rnn_state)
            mask = mask.unsqueeze(0)  # [1, batch_size, 1]
            rnn_state = rnn_state * mask

            x = x.unsqueeze(1)  # [batch_size, 1, hidden_size]
            x, rnn_state = self.rnn(x, rnn_state)
            x = x.squeeze(1)  # [batch_size, hidden_size]
            x = self.rnn_layer_norm(x)
        return x, rnn_state

    def get_action(
        self,
        x: torch.tensor,
        rnn_state: torch.tensor,
        mask: torch.tensor,
        deterministic: bool = False,
        available_actions: torch.tensor = None,
    ) -> tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        """
        :param x: (torch.tensor) input tensor, size [batch_size, obs_shape]
        :param rnn_state: (torch.tensor) rnn hidden state, size [batch_size, hidden_size]
        :param mask: (torch.tensor) mask for rnn state, 0 if the sequence is done, 1 otherwise, size [batch_size]. For example, the done mask for each episode.

        :return: (torch.tensor) action, size [batch_size]
        :return: (torch.tensor) action log probabilities, size [batch_size]
        :return: (torch.tensor) entropy, size [batch_size]
        :return: (torch.tensor) rnn hidden state, size [batch_size, hidden_size]
        """
        x, rnn_state = self.get_hidden_state(x, rnn_state, mask)
        logits = self.actlayer(x)
        logits[available_actions == 0] = float("-inf")
        probs = Categorical(logits=logits)
        if deterministic:
            action = probs.mode()
        else:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), rnn_state

    def get_prob(
        self,
        x: torch.tensor,
        rnn_state: torch.tensor,
        mask: torch.tensor,
        action: torch.tensor,
        available_actions: torch.tensor = None,
    ) -> tuple[torch.tensor, torch.tensor]:
        """
        :param x: (torch.tensor) input tensor, size [batch_size, obs_shape]
        :param rnn_state: (torch.tensor) rnn hidden state, size [batch_size, hidden_size]
        :param mask: (torch.tensor) mask for rnn state, 0 if the sequence is done, 1 otherwise, size [batch_size]. For example, the done mask for each episode.
        :param actions: (torch.tensor) actions, size [batch_size]
        :param available_actions: (torch.tensor) denotes which actions are available to agent (if None, all actions available)

        :return: (torch.tensor) action log probabilities, size [batch_size]
        :return: (torch.tensor) entropy, size [batch_size]
        """
        x, rnn_state = self.get_hidden_state(x, rnn_state, mask)
        logits = self.actlayer(x)
        logits[available_actions == 0] = float("-inf")
        probs = Categorical(logits=logits)
        return probs.log_prob(action), probs.entropy()


class Critic(nn.Module):
    def __init__(self, obs_space: gym.Space, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        input_size = np.array(obs_space.shape).prod()
        self.mlp = nn.Sequential(MLPBase(input_size, hidden_size, num_layers))
        self.valuelayer = layer_init(nn.Linear(hidden_size, 1))

    def get_value(self, x: torch.tensor) -> torch.tensor:
        """
        :param x: (torch.tensor) input tensor, size [batch_size, obs_shape]

        :return: (torch.tensor) value, size [batch_size]
        """
        x = self.mlp(x)
        return self.valuelayer(x)


class MultiAgentPolicy(nn.Module):
    """
    The policy module for multi-agent environments.
    The critic is shared while the actors can be individual for each agent.
    """

    def __init__(
        self,
        act_space: gym.Space,
        obs_space: gym.Space,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_layers_after_run: int = 1,
        use_recurrent_policy: bool = True,
        num_recurrent_layers: int = 1,
        use_value_norm: bool = True,
        share_policy: bool = True,
    ):
        super().__init__()
        # self.critic = nn.Sequential(
        #     layer_init(nn.Linear(np.array(obs_space.shape).prod(), hidden_size)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(64, 64)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(64, 1)),
        # )
        # self.actor = nn.Sequential(
        #     layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(64, 64)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        # )

        # TODO: valuenorm here

    def get_value(self, x):
        pass

    def get_action(self, x):
        pass

    def get_action_prob(self, x):
        pass

    def get_action_and_value(self, x):
        pass


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps_per_episode)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_steps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps_per_episode, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps_per_episode, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps_per_episode, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps_per_episode, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps_per_episode, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps_per_episode, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps_per_episode):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps_per_episode)):
                if t == args.num_steps_per_episode - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
