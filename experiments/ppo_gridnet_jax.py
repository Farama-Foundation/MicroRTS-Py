# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
# https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
import argparse
import os
import random
import subprocess
import pandas as pd
import time
from distutils.util import strtobool
from typing import List, Sequence

os.environ[
    "XLA_PYTHON_CLIENT_MEM_FRACTION"
] = "0.7"  # see https://github.com/google/jax/discussions/6332#discussioncomment-1279991

# import envpool
import flax
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from torch.utils.tensorboard import SummaryWriter

from gym.spaces import MultiDiscrete
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor, VecVideoRecorder
# from torch.distributions.categorical import Categorical

from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="gym-microrts",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="MicroRTSGridModeVecEnv",
        help="the id of the environment")
    parser.add_argument('--partial-obs', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, the game will have partial observability')
    parser.add_argument('--num-bot-envs', type=int, default=0,
        help='the number of bot game environment; 16 bot envs measn 16 games')
    parser.add_argument('--num-selfplay-envs', type=int, default=24,
        help='the number of self play envs; 16 self play envs means 8 games')
    parser.add_argument('--num-models', type=int, default=100,
        help='the number of models saved')
    parser.add_argument('--max-eval-workers', type=int, default=4,
        help='the maximum number of eval workers (skips evaluation when set to 0)')
    parser.add_argument('--train-maps', nargs='+', default=["maps/16x16/basesWorkers16x16A.xml"],
        help='the list of maps used during training')
    parser.add_argument('--eval-maps', nargs='+', default=["maps/16x16/basesWorkers16x16A.xml"],
        help='the list of maps used during evaluation')

    parser.add_argument("--total-timesteps", type=int, default=10000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.num_envs = args.num_selfplay_envs + args.num_bot_envs
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_updates = args.total_timesteps // args.batch_size
    args.save_frequency = max(1, int(args.num_updates // args.num_models))
    # fmt: on
    return args


class MicroRTSStatsRecorder(VecEnvWrapper):
    def __init__(self, env, gamma=0.99) -> None:
        super().__init__(env)
        self.gamma = gamma

    def reset(self):
        obs = self.venv.reset()
        self.raw_rewards = [[] for _ in range(self.num_envs)]
        self.ts = np.zeros(self.num_envs, dtype=np.float32)
        self.raw_discount_rewards = [[] for _ in range(self.num_envs)]
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        newinfos = list(infos[:])
        for i in range(len(dones)):
            self.raw_rewards[i] += [infos[i]["raw_rewards"]]
            self.raw_discount_rewards[i] += [
                (self.gamma ** self.ts[i])
                * np.concatenate((infos[i]["raw_rewards"], infos[i]["raw_rewards"].sum()), axis=None)
            ]
            self.ts[i] += 1
            if dones[i]:
                info = infos[i].copy()
                raw_returns = np.array(self.raw_rewards[i]).sum(0)
                raw_names = [str(rf) for rf in self.rfs]
                raw_discount_returns = np.array(self.raw_discount_rewards[i]).sum(0)
                raw_discount_names = ["discounted_" + str(rf) for rf in self.rfs] + ["discounted"]
                info["microrts_stats"] = dict(zip(raw_names, raw_returns))
                info["microrts_stats"].update(dict(zip(raw_discount_names, raw_discount_returns)))
                self.raw_rewards[i] = []
                self.raw_discount_rewards[i] = []
                self.ts[i] = 0
                newinfos[i] = info
        return obs, rews, dones, newinfos


def run_evaluation(model_path: str, output_path: str, eval_maps: List[str]):
    args = [
        "python",
        "league.py",
        "--evals",
        model_path,
        "--update-db",
        "false",
        "--cuda",
        "false",
        "--output-path",
        output_path,
        "--model-type",
        "ppo_gridnet",
        "--maps",
        *eval_maps,
    ]
    fd = subprocess.Popen(args)
    print(f"Evaluating {model_path}")
    return_code = fd.wait()
    assert return_code == 0
    return (model_path, output_path)


class TrueskillWriter:
    def __init__(self, track, writer, league_path: str, league_step_path: str):
        self.track = track
        self.writer = writer
        self.trueskill_df = pd.read_csv(league_path)
        self.trueskill_step_df = pd.read_csv(league_step_path)
        self.trueskill_step_df["type"] = self.trueskill_step_df["name"]
        self.trueskill_step_df["step"] = 0
        # xxx(okachaiev): not sure we need this copy
        self.preset_trueskill_step_df = self.trueskill_step_df.copy()

    def on_evaluation_done(self, future):
        if future.cancelled():
            return
        model_path, output_path = future.result()
        league = pd.read_csv(output_path, index_col="name")
        assert model_path in league.index
        model_global_step = int(model_path.split("/")[-1][:-3])
        self.writer.add_scalar("charts/trueskill", league.loc[model_path]["trueskill"], model_global_step)
        print(f"global_step={model_global_step}, trueskill={league.loc[model_path]['trueskill']}")

        # table visualization logic
        if self.track:
            trueskill_data = {
                "name": league.loc[model_path].name,
                "mu": league.loc[model_path]["mu"],
                "sigma": league.loc[model_path]["sigma"],
                "trueskill": league.loc[model_path]["trueskill"],
            }
            self.trueskill_df = self.trueskill_df.append(trueskill_data, ignore_index=True)
            wandb.log({"trueskill": wandb.Table(dataframe=self.trueskill_df)})
            trueskill_data["type"] = "training"
            trueskill_data["step"] = model_global_step
            self.trueskill_step_df = self.trueskill_step_df.append(trueskill_data, ignore_index=True)
            preset_trueskill_step_df_clone = self.preset_trueskill_step_df.copy()
            preset_trueskill_step_df_clone["step"] = model_global_step
            self.trueskill_step_df = self.trueskill_step_df.append(preset_trueskill_step_df_clone, ignore_index=True)
            wandb.log({"trueskill_step": wandb.Table(dataframe=self.trueskill_step_df)})

class Network(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(
            32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        x = nn.relu(x)
        return x


class Critic(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        return nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(x)


class Actor(nn.Module):
    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, x):
        x = nn.ConvTranspose(
            32,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(
            self.action_dim,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(x)
        return x


@flax.struct.dataclass
class AgentParams:
    network_params: flax.core.FrozenDict
    actor_params: flax.core.FrozenDict
    critic_params: flax.core.FrozenDict


@flax.struct.dataclass
class Storage:
    obs: jnp.array
    actions: jnp.array
    logprobs: jnp.array
    dones: jnp.array
    values: jnp.array
    invalid_action_masks: jnp.array
    advantages: jnp.array
    returns: jnp.array


if __name__ == "__main__":
    args = parse_args()
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
    key = jax.random.PRNGKey(args.seed)
    key, network_key, actor_key, critic_key = jax.random.split(key, 4)

    # env setup
    envs = MicroRTSGridModeVecEnv(
        num_selfplay_envs=args.num_selfplay_envs,
        num_bot_envs=args.num_bot_envs,
        partial_obs=args.partial_obs,
        max_steps=2000,
        render_theme=2,
        ai2s=[microrts_ai.coacAI for _ in range(args.num_bot_envs - 6)]
        + [microrts_ai.randomBiasedAI for _ in range(min(args.num_bot_envs, 2))]
        + [microrts_ai.lightRushAI for _ in range(min(args.num_bot_envs, 2))]
        + [microrts_ai.workerRushAI for _ in range(min(args.num_bot_envs, 2))],
        map_paths=[args.train_maps[0]],
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
        cycle_maps=args.train_maps,
    )
    envs = MicroRTSStatsRecorder(envs, args.gamma)
    envs = VecMonitor(envs)
    if args.capture_video:
        envs = VecVideoRecorder(
            envs, f"videos/{run_name}", record_video_trigger=lambda x: x % 100000 == 0, video_length=2000
        )
    assert isinstance(envs.action_space, MultiDiscrete), "only MultiDiscrete action space is supported"

    eval_executor = None
    if args.max_eval_workers > 0:
        from concurrent.futures import ThreadPoolExecutor

        eval_executor = ThreadPoolExecutor(max_workers=args.max_eval_workers, thread_name_prefix="league-eval-")

    # EVALUATION LOGIC:
    trueskill_writer = TrueskillWriter(
        args.track, writer, "gym-microrts-static-files/league.csv", "gym-microrts-static-files/league.csv"
    )

    def linear_schedule(count):
        # anneal learning rate linearly after one training iteration which contains
        # (args.num_minibatches * args.update_epochs) gradient updates
        frac = 1.0 - (count // (args.num_minibatches * args.update_epochs)) / args.num_updates
        return args.learning_rate * frac

    mapsize = (16, 16)
    action_space_shape = (*mapsize, len(envs.action_plane_space.nvec))
    invalid_action_shape = (*mapsize, envs.action_plane_space.nvec.sum())


    network = Network()
    actor = Actor(action_dim=envs.action_plane_space.nvec.sum())
    critic = Critic()
    network_params = network.init(network_key, np.array([envs.observation_space.sample()]))
    agent_state = TrainState.create(
        apply_fn=None,
        params=AgentParams(
            network_params,
            actor.init(actor_key, network.apply(network_params, np.array([envs.observation_space.sample()]))),
            critic.init(critic_key, network.apply(network_params, np.array([envs.observation_space.sample()]))),
        ),
        tx=optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=linear_schedule if args.anneal_lr else args.learning_rate, eps=1e-5
            ),
        ),
    )
    network.apply = jax.jit(network.apply)
    actor.apply = jax.jit(actor.apply)
    critic.apply = jax.jit(critic.apply)
    hidden = network.apply(network_params, np.array([envs.observation_space.sample()]))
    print(hidden.shape)
    logits = actor.apply(agent_state.params.actor_params, hidden)
    print(logits.shape)
    grid_logits = logits.reshape(-1, envs.action_plane_space.nvec.sum())
    nvec = [0,] + envs.action_plane_space.nvec.tolist()
    # action = jnp.zeros(logits.shape[:-1] + (len(envs.action_plane_space),))
    # j = 0
    # for i in range(len(nvec) - 1):
    #     key, subkey = jax.random.split(key)
    #     action = action.at[...,i].set(jax.random.categorical(key, logits[...,j:j+nvec[i+1]]))
    #     print(j+nvec[i], j+nvec[i+1])
    #     j += nvec[i+1]
    envs.reset()
    invalid_action_mask = envs.get_action_mask(flatten=False)
    j = 0
    action_list = []
    logprob_list = []
    for i in range(len(nvec) - 1):
        key, subkey = jax.random.split(key)
        action_component_logits = jnp.where(invalid_action_mask[...,j:j+nvec[i+1]], logits[...,j:j+nvec[i+1]], -1e+8)
        action_list += [jax.random.categorical(subkey, action_component_logits)]
        logprob_list += [-optax.softmax_cross_entropy(logits = action_component_logits, labels=jax.nn.one_hot(action_list[-1], nvec[i+1]))]
        j += nvec[i+1]
    action = jnp.stack(action_list, axis=-1)
    logprob = jnp.stack(logprob_list, axis=-1).sum((1,2,3))

    # ALGO Logic: Storage setup
    storage = Storage(
        obs=jnp.zeros((args.num_steps, args.num_envs) + envs.observation_space.shape),
        actions=jnp.zeros((args.num_steps, args.num_envs) + action_space_shape, dtype=jnp.int32),
        logprobs=jnp.zeros((args.num_steps, args.num_envs)),
        dones=jnp.zeros((args.num_steps, args.num_envs)),
        values=jnp.zeros((args.num_steps, args.num_envs)),
        invalid_action_masks=jnp.zeros((args.num_steps, args.num_envs) + invalid_action_shape),
        advantages=jnp.zeros((args.num_steps, args.num_envs)),
        returns=jnp.zeros((args.num_steps, args.num_envs)),
    )
    rewards = np.zeros((args.num_steps, args.num_envs))

    @jax.jit
    def get_action_and_value(
        agent_state: TrainState,
        next_obs: np.ndarray,
        next_done: np.ndarray,
        invalid_action_mask: np.ndarray,
        storage: Storage,
        step: int,
        key: jax.random.PRNGKey,
    ):
        hidden = network.apply(agent_state.params.network_params, next_obs)
        logits = actor.apply(agent_state.params.actor_params, hidden)
        j = 0
        action_list = []
        logprob_list = []
        for i in range(len(nvec) - 1):
            key, subkey = jax.random.split(key)
            action_component_logits = jnp.where(invalid_action_mask[...,j:j+nvec[i+1]], logits[...,j:j+nvec[i+1]], -1e+8)
            action_list += [jax.random.categorical(subkey, action_component_logits)]
            logprob_list += [-optax.softmax_cross_entropy(logits = action_component_logits, labels=jax.nn.one_hot(action_list[-1], nvec[i+1]))]
            j += nvec[i+1]
        action = jnp.stack(action_list, axis=-1)
        logprob = jnp.stack(logprob_list, axis=-1).sum((1,2,3))
        value = critic.apply(agent_state.params.critic_params, hidden)
        storage = storage.replace(
            obs=storage.obs.at[step].set(next_obs),
            dones=storage.dones.at[step].set(next_done),
            actions=storage.actions.at[step].set(action),
            logprobs=storage.logprobs.at[step].set(logprob),
            values=storage.values.at[step].set(value.squeeze()),
            invalid_action_masks=storage.invalid_action_masks.at[step].set(invalid_action_mask),
        )
        return storage, action, key 

    @jax.jit
    def get_action_and_value2(
        params: flax.core.FrozenDict,
        x: np.ndarray,
        action: np.ndarray,
        invalid_action_mask: np.ndarray,
    ):
        hidden = network.apply(agent_state.params.network_params, x)
        logits = actor.apply(agent_state.params.actor_params, hidden)
        j = 0
        logprob_list = []
        entropy_list = []
        for i in range(len(nvec) - 1):
            action_component_logits = jnp.where(invalid_action_mask[...,j:j+nvec[i+1]], logits[...,j:j+nvec[i+1]], -1e+8)
            logprob_list += [-optax.softmax_cross_entropy(logits = action_component_logits, labels=jax.nn.one_hot(action[...,i], nvec[i+1]))]
            normalized_action_component_logits = action_component_logits - jax.scipy.special.logsumexp(action_component_logits, axis=-1, keepdims=True)
            normalized_action_component_logits = normalized_action_component_logits.clip(min=jnp.finfo(logits.dtype).min)
            p_log_p = normalized_action_component_logits * jax.nn.softmax(normalized_action_component_logits)
            entropy_list += [-p_log_p.sum(-1)]
            j += nvec[i+1]
        logprob = jnp.stack(logprob_list, axis=-1).sum((1,2,3))
        entropy = jnp.stack(entropy_list, axis=-1).sum((1,2,3))
        value = critic.apply(params.critic_params, hidden).squeeze()
        return logprob, entropy, value

    @jax.jit
    def compute_gae(
        agent_state: TrainState,
        next_obs: np.ndarray,
        next_done: np.ndarray,
        rewards: np.ndarray,
        storage: Storage,
    ):
        storage = storage.replace(advantages=storage.advantages.at[:].set(0.0))
        next_value = critic.apply(
            agent_state.params.critic_params, network.apply(agent_state.params.network_params, next_obs)
        ).squeeze()
        lastgaelam = 0
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - storage.dones[t + 1]
                nextvalues = storage.values[t + 1]
            delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - storage.values[t]
            lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            storage = storage.replace(advantages=storage.advantages.at[t].set(lastgaelam))
        storage = storage.replace(returns=storage.advantages + storage.values)
        return storage

    @jax.jit
    def update_ppo(
        agent_state: TrainState,
        storage: Storage,
        key: jax.random.PRNGKey,
    ):
        b_obs = storage.obs.reshape((-1,) + envs.observation_space.shape)
        b_logprobs = storage.logprobs.reshape(-1)
        b_actions = storage.actions.reshape((-1,) + action_space_shape)
        b_invalid_action_masks = storage.invalid_action_masks.reshape((-1,) + invalid_action_shape)
        b_advantages = storage.advantages.reshape(-1)
        b_returns = storage.returns.reshape(-1)

        def ppo_loss(params, x, a, logp, ivam, mb_advantages, mb_returns):
            newlogprob, entropy, newvalue = get_action_and_value2(params, x, a, ivam)
            logratio = newlogprob - logp
            ratio = jnp.exp(logratio)
            approx_kl = ((ratio - 1) - logratio).mean()

            if args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * jnp.clip(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

            # Value loss
            v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
            return loss, (pg_loss, v_loss, entropy_loss, jax.lax.stop_gradient(approx_kl))

        ppo_loss_grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)

        # clipfracs = []
        for _ in range(args.update_epochs):
            key, subkey = jax.random.split(key)
            b_inds = jax.random.permutation(subkey, args.batch_size, independent=True)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                (loss, (pg_loss, v_loss, entropy_loss, approx_kl)), grads = ppo_loss_grad_fn(
                    agent_state.params,
                    b_obs[mb_inds],
                    b_actions[mb_inds],
                    b_logprobs[mb_inds],
                    b_invalid_action_masks[mb_inds],
                    b_advantages[mb_inds],
                    b_returns[mb_inds],
                )
                agent_state = agent_state.apply_gradients(grads=grads)
        return agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = envs.reset()
    next_done = np.zeros(args.num_envs)
    # get_action_and_value(agent_state, next_obs, next_done, invalid_action_mask, storage, 0, key)
    # get_action_and_value2(
    #     agent_state.params,
    #     storage.obs.reshape((-1,) + envs.observation_space.shape),
    #     storage.actions.reshape((-1,) + action_space_shape),
    #     storage.invalid_action_masks.reshape((-1,) + invalid_action_shape),
    # )
    # raise
    for update in range(1, args.num_updates + 1):
        step_time = 0
        inference_time = 0
        get_mask_time = 0
        rollout_time_start = time.time()
        update_time_start = time.time()
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            get_mask_time_start = time.time()
            invalid_action_mask = envs.get_action_mask(flatten=False)
            get_mask_time += time.time() - get_mask_time_start

            inference_time_start = time.time()
            storage, action, key = get_action_and_value(agent_state, next_obs, next_done, invalid_action_mask, storage, step, key)
            inference_time += time.time() - inference_time_start

            # TRY NOT TO MODIFY: execute the game and log data.
            # raise
            step_time_start = time.time()
            next_obs, rewards[step], next_done, infos = envs.step(np.array(action).reshape(envs.num_envs, -1))
            step_time += time.time() - step_time_start
            for info in infos:
                if "episode" in info.keys():
                    # print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    # for k in info["microrts_stats"]:
                    #     writer.add_scalar(f"charts/episodic_return/{k}", info["microrts_stats"][k], global_step)
                    # break
        rollout_time = time.time() - rollout_time_start

        training_time_start = time.time()
        storage = compute_gae(agent_state, next_obs, next_done, rewards, storage)
        agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key = update_ppo(
            agent_state,
            storage,
            key,
        )


        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", agent_state.opt_state[1].hyperparams["learning_rate"].item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        # writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        # writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/loss", loss.item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        print(f"times, step={step_time}, inference={inference_time}, get_mask={get_mask_time}, rollout={rollout_time}, training={time.time() - training_time_start}, update={time.time() - update_time_start}")

    envs.close()
    writer.close()
