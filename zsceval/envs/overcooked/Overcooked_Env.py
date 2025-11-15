import os
import pickle
import pprint

import gym
import imageio
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from loguru import logger

from zsceval.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction
from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import (
    BASE_REW_SHAPING_PARAMS,
    SHAPED_INFOS,
    OvercookedGridworld,
)
from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_trajectory import (
    DEFAULT_TRAJ_KEYS,
    TIMESTEP_TRAJ_KEYS,
)
from zsceval.envs.overcooked.overcooked_ai_py.planning.planners import (
    NO_COUNTERS_PARAMS,
    MediumLevelPlanner,
)
from zsceval.envs.overcooked.overcooked_ai_py.utils import mean_and_std_err
from zsceval.envs.overcooked.overcooked_ai_py.visualization.state_visualizer import (
    StateVisualizer,
)
from zsceval.envs.overcooked.script_agent.script_agent import SCRIPT_AGENTS
from zsceval.utils.train_util import setup_seed

DEFAULT_ENV_PARAMS = {"horizon": 400}

MAX_HORIZON = 1e10


class OvercookedEnv:
    """An environment wrapper for the OvercookedGridworld Markov Decision Process.

    The environment keeps track of the current state of the agent, updates
    it as the agent takes actions, and provides rewards to the agent.
    """

    def __init__(
        self,
        mdp,
        start_state_fn=None,
        horizon=MAX_HORIZON,
        debug=False,
        evaluation: bool = False,
        use_random_player_pos: bool = False,
        use_random_terrain_state: bool = False,
        num_initial_state: int = 5,
        replay_return_threshold: float = 0.75,
    ):
        """
        mdp (OvercookedGridworld or function): either an instance of the MDP or a function that returns MDP instances
        start_state_fn (OvercookedState): function that returns start state for the MDP, called at each environment reset
        horizon (float): number of steps before the environment returns done=True
        """
        if isinstance(mdp, OvercookedGridworld):
            self.mdp_generator_fn = lambda: mdp
        elif callable(mdp) and isinstance(mdp(), OvercookedGridworld):
            self.mdp_generator_fn = mdp
        else:
            raise ValueError("Mdp should be either OvercookedGridworld instance or a generating function")

        self.horizon = horizon
        self.start_state_fn = start_state_fn
        self.evaluation = evaluation
        self.use_random_player_pos = use_random_player_pos
        self.use_random_terrain_state = use_random_terrain_state
        self.reset()

        if self.horizon >= MAX_HORIZON and self.state.order_list is None and debug:
            print("Environment has (near-)infinite horizon and no terminal states")

    def __repr__(self):
        """Standard way to view the state of an environment programatically
        is just to print the Env object"""
        return self.mdp.state_string(self.state)

    def print_state_transition(self, a_t, r_t, info):
        print(
            "Timestep: {}\nJoint action taken: {} \t Reward: {} + shape * {} \n{}\n".format(
                self.t,
                tuple(Action.ACTION_TO_CHAR[a] for a in a_t),
                r_t,
                info["shaped_r"],
                self,
            )
        )

    @property
    def env_params(self):
        return {"start_state_fn": self.start_state_fn, "horizon": self.horizon}

    def display_states(self, *states):
        old_state = self.state
        for s in states:
            self.state = s
            print(self)
        self.state = old_state

    @staticmethod
    def print_state(mdp, s):
        e = OvercookedEnv(mdp, s)
        print(e)

    def copy(self):
        return OvercookedEnv(
            mdp=self.mdp.copy(),
            start_state_fn=self.start_state_fn,
            horizon=self.horizon,
        )

    def step(self, joint_action):
        """Performs a joint action, updating the environment state
        and providing a reward.

        On being done, stats about the episode are added to info:
            ep_sparse_r: the environment sparse reward, given only at soup delivery
            ep_shaped_r: the component of the reward that is due to reward shaped (excluding sparse rewards)
            ep_length: length of rollout
        """
        assert not self.is_done()
        self.t += 1
        next_state, mdp_infos = self.mdp.get_state_transition(self.state, joint_action)

        # Update game_stats
        self._update_game_stats(mdp_infos)

        # Update state and done
        self.state = next_state
        done = self.is_done()
        env_info = self._prepare_info_dict([{}, {}], mdp_infos)

        if done:
            self._add_episode_info(env_info)

        timestep_sparse_reward = sum(mdp_infos["sparse_reward_by_agent"])

        return next_state, timestep_sparse_reward, done, env_info

    def reset(self):
        """Resets the environment. Does NOT reset the agent."""
        self.mdp = self.mdp_generator_fn()
        if self.start_state_fn is None:
            self.state = self.mdp.get_standard_start_state()
        elif type(self.start_state_fn) in [float, int]:
            # p = np.random.uniform(0, 1)
            p = np.random.uniform(0, 1)
            if p <= self.start_state_fn and not self.evaluation:
                # logger.error("Random start state")
                self.state = self.mdp.get_random_start_state(self.use_random_terrain_state, self.use_random_player_pos)
            else:
                self.state = self.mdp.get_standard_start_state()
        else:
            self.state = self.start_state_fn()
        # assert self.mdp.start_player_positions == list(self.state.player_positions)
        self.cumulative_sparse_rewards = 0
        self.cumulative_shaped_rewards = 0
        self.t = 0

        rewards_dict = {
            "cumulative_sparse_rewards_by_agent": np.array([0] * self.mdp.num_players),
            "cumulative_shaped_rewards_by_agent": np.array([0] * self.mdp.num_players),
            "cumulative_category_rewards_by_agent": np.zeros((self.mdp.num_players, len(SHAPED_INFOS))),
        }

        self.game_stats = {**rewards_dict}

    def is_done(self):
        """Whether the episode is over."""
        return self.t >= self.horizon or self.mdp.is_terminal(self.state)

    def _prepare_info_dict(self, joint_agent_action_info, mdp_infos):
        """
        The normal timestep info dict will contain infos specifc to each agent's action taken,
        and reward shaping information.
        """
        # Get the agent action info, that could contain info about action probs, or other
        # custom user defined information
        env_info = {"agent_infos": [joint_agent_action_info[agent_idx] for agent_idx in range(self.mdp.num_players)]}
        # TODO: This can be further simplified by having all the mdp_infos copied over to the env_infos automatically
        env_info["sparse_r_by_agent"] = mdp_infos["sparse_reward_by_agent"]
        env_info["shaped_r_by_agent"] = mdp_infos["shaped_reward_by_agent"]
        env_info["shaped_info_by_agent"] = mdp_infos["shaped_info_by_agent"]
        env_info["phi_s"] = mdp_infos["phi_s"] if "phi_s" in mdp_infos else None
        env_info["phi_s_prime"] = mdp_infos["phi_s_prime"] if "phi_s_prime" in mdp_infos else None
        return env_info

    # MARK: info
    def _add_episode_info(self, env_info):
        env_info["episode"] = {
            "ep_game_stats": self.game_stats,
            "ep_sparse_r": sum(self.game_stats["cumulative_sparse_rewards_by_agent"]),
            "ep_shaped_r": sum(self.game_stats["cumulative_shaped_rewards_by_agent"]),
            "ep_sparse_r_by_agent": self.game_stats["cumulative_sparse_rewards_by_agent"],
            "ep_shaped_r_by_agent": self.game_stats["cumulative_shaped_rewards_by_agent"],
            "ep_category_r_by_agent": self.game_stats["cumulative_category_rewards_by_agent"],
            "ep_length": self.t,
        }
        return env_info

    def vectorize_shaped_info(self, shaped_info_by_agent):
        from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import (
            SHAPED_INFOS,
        )

        def vectorize(d: dict):
            # return np.array([v for k, v in d.items()])
            return np.array([d[k] for k in SHAPED_INFOS])

        shaped_info_by_agent = np.stack([vectorize(shaped_info) for shaped_info in shaped_info_by_agent])
        return shaped_info_by_agent

    def _update_game_stats(self, infos):
        """
        Update the game stats dict based on the events of the current step
        NOTE: the timer ticks after events are logged, so there can be events from time 0 to time self.horizon - 1
        """
        self.game_stats["cumulative_sparse_rewards_by_agent"] += np.array(infos["sparse_reward_by_agent"])
        self.game_stats["cumulative_shaped_rewards_by_agent"] += np.array(infos["shaped_reward_by_agent"])
        self.game_stats["cumulative_category_rewards_by_agent"] += self.vectorize_shaped_info(
            infos["shaped_info_by_agent"]
        )

        """for event_type, bool_list_by_agent in infos["event_infos"].items():
            # For each event type, store the timestep if it occurred
            event_occurred_by_idx = [int(x) for x in bool_list_by_agent]
            for idx, event_by_agent in enumerate(event_occurred_by_idx):
                if event_by_agent:
                    self.game_stats[event_type][idx].append(self.state.timestep)"""

    def execute_plan(self, start_state, joint_action_plan, display=False):
        """Executes action_plan (a list of joint actions) from a start
        state in the mdp and returns the resulting state."""
        self.state = start_state
        done = False
        if display:
            print(f"Starting state\n{self}")
        for joint_action in joint_action_plan:
            self.step(joint_action)
            done = self.is_done()
            if display:
                print(self)
            if done:
                break
        successor_state = self.state
        self.reset()
        return successor_state, done

    def run_agents(self, agent_pair, include_final_state=False, display=False, display_until=np.Inf):
        """
        Trajectory returned will a list of state-action pairs (s_t, joint_a_t, r_t, done_t).
        """
        assert (
            self.cumulative_sparse_rewards == self.cumulative_shaped_rewards == 0
        ), "Did not reset environment before running agents"
        trajectory = []
        done = False

        if display:
            print(self)
        while not done:
            s_t = self.state
            a_t = agent_pair.joint_action(s_t)

            # Break if either agent is out of actions
            if any([a is None for a in a_t]):
                break

            s_tp1, r_t, done, info = self.step(a_t)
            trajectory.append((s_t, a_t, r_t, done))

            if display and self.t < display_until:
                self.print_state_transition(a_t, r_t, info)

        assert len(trajectory) == self.t, f"{len(trajectory)} vs {self.t}"

        # Add final state
        if include_final_state:
            trajectory.append((s_tp1, (None, None), 0, True))

        return (
            np.array(trajectory),
            self.t,
            self.cumulative_sparse_rewards,
            self.cumulative_shaped_rewards,
        )

    def get_rollouts(
        self,
        agent_pair,
        num_games,
        display=False,
        final_state=False,
        agent_idx=0,
        reward_shaping=0.0,
        display_until=np.Inf,
        info=True,
    ):
        """
        Simulate `num_games` number rollouts with the current agent_pair and returns processed
        trajectories.

        Only returns the trajectories for one of the agents (the actions _that_ agent took),
        namely the one indicated by `agent_idx`.

        Returning excessive information to be able to convert trajectories to any required format
        (baselines, stable_baselines, etc)

        NOTE: standard trajectories format used throughout the codebase
        """
        trajectories = {
            # With shape (n_timesteps, game_len), where game_len might vary across games:
            "ep_observations": [],
            "ep_actions": [],
            "ep_rewards": [],  # Individual dense (= sparse + shaped * rew_shaping) reward values
            "ep_dones": [],  # Individual done values
            # With shape (n_episodes, ):
            "ep_returns": [],  # Sum of dense and sparse rewards across each episode
            "ep_returns_sparse": [],  # Sum of sparse rewards across each episode
            "ep_lengths": [],  # Lengths of each episode
            "mdp_params": [],  # Custom MDP params to for each episode
            "env_params": [],  # Custom Env params for each episode
        }

        for _ in tqdm.trange(num_games):
            agent_pair.set_mdp(self.mdp)

            trajectory, time_taken, tot_rews_sparse, tot_rews_shaped = self.run_agents(
                agent_pair,
                display=display,
                include_final_state=final_state,
                display_until=display_until,
            )
            obs, actions, rews, dones = (
                trajectory.T[0],
                trajectory.T[1],
                trajectory.T[2],
                trajectory.T[3],
            )
            trajectories["ep_observations"].append(obs)
            trajectories["ep_actions"].append(actions)
            trajectories["ep_rewards"].append(rews)
            trajectories["ep_dones"].append(dones)
            trajectories["ep_returns"].append(tot_rews_sparse + tot_rews_shaped * reward_shaping)
            trajectories["ep_returns_sparse"].append(tot_rews_sparse)
            trajectories["ep_lengths"].append(time_taken)
            trajectories["mdp_params"].append(self.mdp.mdp_params)
            trajectories["env_params"].append(self.env_params)

            self.reset()
            agent_pair.reset()

        mu, se = mean_and_std_err(trajectories["ep_returns"])
        if info:
            print(
                "Avg reward {:.2f} (std: {:.2f}, se: {:.2f}) over {} games of avg length {}".format(
                    mu,
                    np.std(trajectories["ep_returns"]),
                    se,
                    num_games,
                    np.mean(trajectories["ep_lengths"]),
                )
            )

        # Converting to numpy arrays
        trajectories = {k: np.array(v) for k, v in trajectories.items()}
        return trajectories


import os
import pickle
import pprint

import gym
import imageio
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from loguru import logger

# --- 필요한 import 구문들 (사용자 코드에서 가져옴) ---
from zsceval.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction
from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import (
    BASE_REW_SHAPING_PARAMS,
    SHAPED_INFOS,
    OvercookedGridworld,
)
from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_trajectory import (
    DEFAULT_TRAJ_KEYS,
    TIMESTEP_TRAJ_KEYS,
)
from zsceval.envs.overcooked.overcooked_ai_py.planning.planners import (
    NO_COUNTERS_PARAMS,
    MediumLevelPlanner,
)
from zsceval.envs.overcooked.overcooked_ai_py.utils import mean_and_std_err
from zsceval.envs.overcooked.overcooked_ai_py.visualization.state_visualizer import (
    StateVisualizer,
)
from zsceval.envs.overcooked.script_agent.script_agent import SCRIPT_AGENTS
from zsceval.utils.train_util import setup_seed
# OvercookedEnv 클래스 정의가 이 코드 위에 있다고 가정합니다.
# class OvercookedEnv: ...
# ---------------------------------------------------


class Overcooked(gym.Env):
    """
    Wrapper for the Env class above that is SOMEWHAT compatible with the standard gym API.
    (docstring 생략)
    """

    env_name = "Overcooked-v0"

    def __init__(
        self,
        all_args,
        run_dir,
        baselines_reproducible=True,
        featurize_type=None, # 💡 [수정] 기본값을 None으로 변경하여 내부에서 동적 할당
        stuck_time=4,
        rank=None,
        evaluation=False,
    ):
        """
        base_env: OvercookedEnv
        featurize_fn(mdp, state): fn used to featurize states returned in the 'both_agent_obs' field
        """
        if baselines_reproducible:
            np.random.seed(0)
            
        self.all_args = all_args
        self.num_agents = all_args.num_agents # 💡 [추가] num_agents를 먼저 정의
        self.agent_idx = 0
        self._initial_reward_shaping_factor = all_args.initial_reward_shaping_factor
        self.reward_shaping_factor = all_args.reward_shaping_factor
        self.reward_shaping_horizon = all_args.reward_shaping_horizon
        self.use_phi = all_args.use_phi
        self.use_hsp = all_args.use_hsp
        self.store_traj = getattr(all_args, "store_traj", False)
        self.rank = rank
        self.random_index = all_args.random_index
        
        if self.use_hsp:
            # 💡 [수정] HSP 관련 로직을 N명 에이전트에 맞게 일반화
            self.hsp_weights = []
            for i in range(self.num_agents):
                weight_str = getattr(all_args, f"w{i}", None)
                if weight_str:
                    self.hsp_weights.append(self.string2array(weight_str))
            
            w_dict = {f"w{i}": w for i, w in enumerate(self.hsp_weights)}
            logger.debug("hsp weights:\n" + pprint.pformat(w_dict, compact=True, width=120))
            self.cumulative_hidden_reward = np.zeros(self.num_agents)

        self.use_available_actions = getattr(all_args, "use_available_actions", True)
        # Only render if use_render is True AND this env's rank is within n_render_rollout_threads
        n_render_threads = getattr(all_args, "n_render_rollout_threads", 1)
        self.use_render = all_args.use_render and (rank is not None and rank < n_render_threads)
        self.layout_name = all_args.layout_name
        self.episode_length = all_args.episode_length
        self.random_start_prob = getattr(all_args, "random_start_prob", 0.0)
        self.stuck_time = stuck_time
        self.history_sa = []
        self.traj_num = 0
        self.step_count = 0
        self.run_dir = run_dir
        mdp_params = {"layout_name": all_args.layout_name, "start_order_list": None}
        
        mdp_params.update({"rew_shaping_params": BASE_REW_SHAPING_PARAMS})
        env_params = {
            "horizon": all_args.episode_length,
            "evaluation": evaluation,
            "use_random_player_pos": all_args.use_random_player_pos,
            "use_random_terrain_state": all_args.use_random_terrain_state,
            "num_initial_state": all_args.num_initial_state,
            "replay_return_threshold": all_args.replay_return_threshold,
        }
        self.mdp_fn = lambda: OvercookedGridworld.from_layout_name(**mdp_params)
        self.base_mdp = self.mdp_fn()
        # base_mdp 생성 후 num_players가 올바르게 설정되었는지 확인
        assert self.base_mdp.num_players == self.num_agents, \
            f"Layout '{self.layout_name}' supports {self.base_mdp.num_players} players, but num_agents is set to {self.num_agents}."

        self.base_env = OvercookedEnv(
            self.mdp_fn,
            start_state_fn=self.random_start_prob if self.random_start_prob > 0 else None,
            **env_params,
        )
        self.mlp = MediumLevelPlanner.from_pickle_or_compute(
            mdp=self.base_mdp, mlp_params=NO_COUNTERS_PARAMS, force_compute=False
        )
        self.use_agent_policy_id = dict(all_args._get_kwargs()).get(
            "use_agent_policy_id", False
        )
        self.agent_policy_id = [-1.0 for _ in range(self.num_agents)]
        self.use_timestep_feature = all_args.use_timestep_feature
        self.featurize_fn_ppo = lambda state: self.base_mdp.lossless_state_encoding(
            state,
            add_timestep=self.use_timestep_feature,
            horizon=self.episode_length,
            add_identity=all_args.use_identity_feature,
        )
        self.featurize_fn_bc = lambda state: self.base_mdp.featurize_state(state)
        self.featurize_fn_mapping = {
            "ppo": self.featurize_fn_ppo,
            "bc": self.featurize_fn_bc,
        }
        
        # 💡 [수정] featurize_type이 None이면 에이전트 수에 맞게 동적으로 생성
        if featurize_type is None:
            featurize_type = tuple(["ppo"] * self.num_agents)
        self.reset_featurize_type(featurize_type=featurize_type)

        # 💡 [수정] script_agent 초기화 및 population 로직을 N명에 맞게 일반화
        self.script_agent = [None] * self.num_agents
        if self.all_args.algorithm_name == "population":
            assert not self.random_index
            # all_args에 agent0_policy_name, agent1_policy_name, ... 등이 있다고 가정
            policy_names = [getattr(all_args, f"agent{i}_policy_name") for i in range(self.num_agents)]
            for player_idx, policy_name in enumerate(policy_names):
                if policy_name.startswith("script:"):
                    self.script_agent[player_idx] = SCRIPT_AGENTS[policy_name[7:]]()
                    self.script_agent[player_idx].reset(self.base_env.mdp, self.base_env.state, player_idx)

    def reset_featurize_type(self, featurize_type):
            """
            Sets the feature representation type for each agent and resets the
            observation and action spaces accordingly.
            """
            # 💡 [수정] assert 구문을 self.num_agents를 사용하도록 변경
            assert len(featurize_type) == self.num_agents, \
                f"Length of featurize_type ({len(featurize_type)}) must match num_agents ({self.num_agents})."
            
            self.featurize_type = featurize_type
            # 이 lambda 함수는 featurize_type의 길이에 따라 동작하므로 수정 필요 없음
            self.featurize_fn = lambda state: [
                self.featurize_fn_mapping[f](state)[i] * (255 if f == "ppo" else 1)
                for i, f in enumerate(self.featurize_type)
            ]

            # observation/action space를 self.num_agents에 맞게 설정
            self.observation_space = []
            self.share_observation_space = []
            self.action_space = []
            self._setup_observation_space()

            # 💡 [수정] 하드코딩된 '2'를 self.num_agents로 변경하여 루프를 일반화
            for i in range(self.num_agents):
                self.observation_space.append(self._observation_space(featurize_type[i]))
                self.action_space.append(gym.spaces.Discrete(len(Action.ALL_ACTIONS)))
                # 각 에이전트는 자신만의 공유 관측 공간을 가짐
                self.share_observation_space.append(self._setup_share_observation_space())
                
    def _anneal(self, start_v, curr_t, end_t, end_v=0, start_t=0):
        if end_t == 0:
            # No annealing if horizon is zero
            return start_v
        else:
            off_t = curr_t - start_t
            # Calculate the new value based on linear annealing formula
            fraction = max(1 - float(off_t) / (end_t - start_t), 0)
            return fraction * start_v + (1 - fraction) * end_v

    def onehot2idx(self, onehot):
        idx = []
        for a in onehot:
            idx.append(np.argmax(a))
        return idx

    def string2array(self, weight):
        w = []
        for s in weight.split(","):
            w.append(float(s))
        return np.array(w)

    def _action_convertor(self, action):
        return [a[0] for a in list(action)]

    def _observation_space(self, featurize_type):
        return {"ppo": self.ppo_observation_space, "bc": self.bc_observation_space}[featurize_type]

    def _setup_observation_space(self):
        dummy_state = self.base_env.mdp.get_standard_start_state()

        # ppo observation
        # featurize_fn_ppo = lambda state: self.base_mdp.lossless_state_encoding(state)
        featurize_fn_ppo = self.featurize_fn_ppo
        obs_shape = featurize_fn_ppo(dummy_state)[0].shape
        high = np.ones(obs_shape) * float("inf")
        low = np.ones(obs_shape) * 0
        self.ppo_observation_space = gym.spaces.Box(np.float32(low), np.float32(high), dtype=np.float32)

        # bc observation
        featurize_fn_bc = lambda state: self.base_mdp.featurize_state(state, self.mlp)
        obs_shape = featurize_fn_bc(dummy_state)[0].shape
        high = np.ones(obs_shape) * 100
        low = np.ones(obs_shape) * -100
        self.bc_observation_space = gym.spaces.Box(np.float32(low), np.float32(high), dtype=np.float32)

    def _setup_share_observation_space(self):
        dummy_state = self.base_env.mdp.get_standard_start_state()
        share_obs_shape = self.featurize_fn_ppo(dummy_state)[0].shape
        if self.use_agent_policy_id:
            share_obs_shape = [
                share_obs_shape[0],
                share_obs_shape[1],
                share_obs_shape[2] + 1,
            ]
        share_obs_shape = [
            share_obs_shape[0],
            share_obs_shape[1],
            share_obs_shape[2] * self.num_agents,
        ]
        high = np.ones(share_obs_shape) * float("inf")
        low = np.ones(share_obs_shape) * 0

        return gym.spaces.Box(np.float32(low), np.float32(high), dtype=np.float32)

    def _set_agent_policy_id(self, agent_policy_id):
        self.agent_policy_id = agent_policy_id

    def _gen_share_observation(self, state):
            # 1. 모든 에이전트의 개별 관측(observation)을 리스트로 받습니다.
            share_obs = list(self.featurize_fn_ppo(state))

            # 2. (선택 사항) 각 관측에 에이전트별 policy ID를 추가합니다.
            # 이 로직은 이미 self.num_agents를 사용하므로 수정할 필요가 없습니다.
            if self.use_agent_policy_id:
                for a in range(self.num_agents):
                    share_obs[a] = np.concatenate(
                        [
                            share_obs[a],
                            np.ones((*share_obs[a].shape[:2], 1), dtype=np.float32) * self.agent_policy_id[a],
                        ],
                        axis=-1,
                    )

            # 3. 💡 [수정] 각 에이전트의 관점에서 공유 관측을 생성하고 리스트에 저장합니다.
            all_share_obs = []
            for i in range(self.num_agents):
                # i번째 에이전트의 관점에서 관측 리스트의 순서를 재배열합니다.
                # 예: i=1, num_agents=3 -> [obs_1, obs_2, obs_0]
                # np.roll을 사용하면 리스트를 간단하게 순환시킬 수 있습니다.
                # -i 만큼 roll하면 i번째 요소가 맨 앞으로 옵니다.
                ordered_obs = np.roll(share_obs, -i, axis=0)
                
                # 재정렬된 관측들을 채널(feature) 축(axis=-1)으로 연결합니다.
                concatenated_obs = np.concatenate(ordered_obs, axis=-1) * 255
                all_share_obs.append(concatenated_obs)
                
            # 4. 모든 공유 관측을 하나의 numpy 배열로 합쳐서 반환합니다.
            # 최종 shape: (num_agents, height, width, channels * num_agents)
            return np.stack(all_share_obs, axis=0)

    def _get_available_actions(self):
        available_actions = np.ones((self.num_agents, len(Action.ALL_ACTIONS)), dtype=np.uint8)
        if self.use_available_actions:
            state = self.base_env.state
            interact_index = Action.ACTION_TO_INDEX["interact"]
            for agent_idx in range(self.num_agents):
                player = state.players[agent_idx]
                pos = player.position
                o = player.orientation
                for move_i, move in enumerate(Direction.ALL_DIRECTIONS):
                    new_pos = Action.move_in_direction(pos, move)
                    if new_pos not in self.base_mdp.get_valid_player_positions() and o == move:
                        available_actions[agent_idx, move_i] = 0

                i_pos = Action.move_in_direction(pos, o)
                terrain_type = self.base_mdp.get_terrain_type_at_pos(i_pos)

                if (
                    terrain_type == " "
                    or (
                        terrain_type == "X"
                        and (
                            (not player.has_object() and not state.has_object(i_pos))
                            or (player.has_object() and state.has_object(i_pos))
                        )
                    )
                    or (terrain_type in ["O", "T", "D"] and player.has_object())
                    or (
                        terrain_type == "P"
                        and (not player.has_object() or player.get_object().name not in ["dish", "onion", "tomato"])
                    )
                    or (terrain_type == "S" and (not player.has_object() or player.get_object().name not in ["soup"]))
                ):
                    available_actions[agent_idx, interact_index] = 0
                # assert available_actions[agent_idx].sum() > 0
        return available_actions

    def step(self, action):
            """
            action:
                (agent_0_action, agent_1_action, ...) 형태의 튜플.
                RL 에이전트(self.agent_idx)의 행동이 항상 0번 인덱스로 전달됨.

            returns:
                observation: RL 에이전트의 관점에서 정렬된 관측 리스트.
                share_obs: 각 에이전트의 관점에서 정렬된 공유 관측.
                reward: RL 에이전트의 관점에서 정렬된 보상 리스트.
                done: 종료 신호 리스트.
                info: 환경 정보 딕셔너리.
                available_actions: RL 에이전트의 관점에서 정렬된 사용 가능 행동.
            """
            self.step_count += 1
            action_indices = self._action_convertor(action)
            assert all(self.action_space[0].contains(a) for a in action_indices), f"{action_indices!r} invalid"

            # 💡 1. 행동 처리 일반화
            # 정책의 관점에서 받은 행동 리스트 (RL 에이전트 행동이 0번 인덱스)
            policy_joint_action = [Action.INDEX_TO_ACTION[a] for a in action_indices]

            # 스크립트 에이전트가 있다면, 해당 에이전트의 실제 인덱스에 행동을 덮어씀
            # 이 단계에서는 아직 policy 관점의 순서를 유지
            for i in range(self.num_agents):
                # self.agent_idx를 고려하여 실제 스크립트 에이전트의 인덱스를 계산
                actual_agent_idx = (i + self.agent_idx) % self.num_agents
                if self.script_agent[actual_agent_idx] is not None:
                    # 스크립트 에이전트의 행동은 실제 인덱스를 기준으로 결정됨
                    action_for_script_agent = self.script_agent[actual_agent_idx].step(self.base_env.mdp, self.base_env.state, actual_agent_idx)
                    # i번째 (정책 관점) 행동을 덮어씀
                    policy_joint_action[i] = action_for_script_agent

            # 환경에 전달하기 위해 실제 에이전트 순서로 재배열
            env_joint_action = list(policy_joint_action)
            if self.agent_idx != 0:
                # np.roll을 사용해 RL 에이전트의 행동을 올바른 위치로 이동
                env_joint_action = list(np.roll(env_joint_action, self.agent_idx, axis=0))

            joint_action = tuple(env_joint_action)

            # 💡 로그 및 히스토리 저장은 실제 환경 순서(joint_action)를 따름
            if self.stuck_time > 0:
                self.history_sa[-1][1] = joint_action
            if self.store_traj:
                self.traj_to_store.append(joint_action)

            # 환경 스텝 실행
            next_state, sparse_reward, done, info = self.base_env.step(joint_action)

            # 💡 2. 보상 계산 일반화
            dense_reward = info["shaped_r_by_agent"]
            shaped_rewards = [] # 실제 에이전트 순서의 보상

            if self.use_hsp:
                from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import SHAPED_INFOS
                shaped_info = info["shaped_info_by_agent"]
                vec_shaped_info = np.array([[agent_info[k] for k in SHAPED_INFOS] for agent_info in shaped_info]).astype(np.float32)

                hidden_rewards_for_log = np.zeros(self.num_agents)
                
                for i in range(self.num_agents):
                    # 💡 [수정] 유연한 Weight 선택 로직
                    # 1) 정책 관점 인덱스 i를 실제 환경 인덱스로 변환
                    actual_agent_idx = (i + self.agent_idx) % self.num_agents
                    
                    # 2) 해당 에이전트의 weight 선택
                    if actual_agent_idx < len(self.hsp_weights):
                        # 각 에이전트마다 개별 weight가 있는 경우 (독립적 파라미터 네트워크)
                        weight = self.hsp_weights[actual_agent_idx]
                    elif len(self.hsp_weights) == 2:
                        # 2개의 weight만 있는 경우: main vs partner 구조 (기존 방식)
                        weight = self.hsp_weights[0] if (i == self.agent_idx) else self.hsp_weights[1]
                    elif len(self.hsp_weights) == 1:
                        # 1개의 weight만 있는 경우: 모두 동일 (공유 파라미터 네트워크)
                        weight = self.hsp_weights[0]
                    else:
                        raise ValueError(f"Invalid HSP weights configuration: {len(self.hsp_weights)} weights for {self.num_agents} agents")
                    
                    hidden_r = np.dot(weight[:-1], vec_shaped_info[i]) + sparse_reward * weight[-1]
                    hidden_rewards_for_log[i] = hidden_r
                    
                    final_shaped_r = hidden_r + (self.reward_shaping_factor * dense_reward[i] if (i != self.agent_idx) else 0)
                    shaped_rewards.append([final_shaped_r])

                self.cumulative_hidden_reward += hidden_rewards_for_log
            else:
                for i in range(self.num_agents):
                    r = sparse_reward + self.reward_shaping_factor * dense_reward[i]
                    shaped_rewards.append([r])

            self.history_sa = self.history_sa[1:] + [[next_state, None]]

            # Stuck 정보 계산
            stuck_info = []
            for agent_id in range(self.num_agents):
                stuck, history_a = self.is_stuck(agent_id)
                stuck_info.append([stuck, [Action.ACTION_TO_INDEX[a] for a in history_a] if stuck else []])
            info["stuck"] = stuck_info

            # 💡 렌더링 및 Trajectory 저장은 실제 환경 데이터를 사용
            if self.use_render:
                self.traj["ep_states"][0].append(self.base_env.state)
                self.traj["ep_actions"][0].append(joint_action) # 실제 환경 순서의 행동
                self.traj["ep_rewards"][0].append(sparse_reward)
                self.traj["ep_dones"][0].append(done)
                self.traj["ep_infos"][0].append(info)
                if done:
                    self.traj["ep_returns"].append(info["episode"]["ep_sparse_r"])
                    self.traj["mdp_params"].append(self.base_mdp.mdp_params)
                    self.traj["env_params"].append(self.base_env.env_params)
                    self.render()

            if done:
                if self.store_traj: self._store_trajectory()
                if self.use_hsp: info["episode"]["ep_hidden_r_by_agent"] = self.cumulative_hidden_reward
                info["bad_transition"] = True
            else:
                info["bad_transition"] = False
                
            # 💡 3. 반환 데이터 처리 일반화
            # 실제 에이전트 순서의 관측값 리스트
            all_obs = self.featurize_fn(next_state)
            
            # 정책에 전달하기 위해 RL 에이전트의 데이터가 0번 인덱스에 오도록 재배열
            policy_obs = list(np.roll(all_obs, -self.agent_idx, axis=0))
            policy_rewards = list(np.roll(shaped_rewards, -self.agent_idx, axis=0))
            
            # 공유 관측 생성 (내부적으로 각 에이전트 관점에서 생성됨)
            share_obs = self._gen_share_observation(self.base_env.state)
            dones = [done] * self.num_agents
            
            available_actions = self._get_available_actions()
            # RL 에이전트의 관점에서 사용 가능 행동 재배열
            policy_available_actions = np.roll(available_actions, -self.agent_idx, axis=0)

            return policy_obs, share_obs, policy_rewards, dones, info, policy_available_actions

    def anneal_reward_shaping_factor(self, timesteps):
        """
        Set the current reward shaping factor such that we anneal linearly until self.reward_shaping_horizon
        timesteps, given that we are currently at timestep "timesteps"
        """
        new_factor = self._anneal(self._initial_reward_shaping_factor, timesteps, self.reward_shaping_horizon)
        self.set_reward_shaping_factor(new_factor)

    def set_reward_shaping_factor(self, factor):
        self.reward_shaping_factor = factor

    def reset(self, reset_choose=True):
            """
            환경을 리셋하고, RL 에이전트의 관점에서 정렬된 첫 관측값과
            사용 가능 행동을 반환합니다.
            """
            # 💡 [수정 없음] 환경 리셋 로직은 그대로 사용합니다.
            if reset_choose:
                self.traj_num += 1
                self.step_count = 0
                self.base_env.reset()

            # 💡 [수정] self.agent_idx를 0부터 num_agents-1 사이에서 랜덤하게 선택합니다.
            if self.random_index:
                self.agent_idx = np.random.choice(self.num_agents)
            else:
                self.agent_idx = 0

            # 💡 [수정 없음] 스크립트 에이전트 리셋 로직은 이미 일반화되어 있습니다.
            for a in range(self.num_agents):
                if self.script_agent[a] is not None:
                    self.script_agent[a].reset(self.base_env.mdp, self.base_env.state, a)

            self.mdp = self.base_env.mdp
            
            # 💡 [수정] 모든 에이전트의 관측값을 하나의 리스트로 받습니다.
            all_obs = self.featurize_fn(self.base_env.state)
            if self.stuck_time > 0:
                self.history_sa = [None for _ in range(self.stuck_time - 1)] + [[self.base_env.state, None]]

            # 💡 [수정] RL 에이전트의 관측값이 0번 인덱스에 오도록 재배열합니다.
            policy_obs = list(np.roll(all_obs, -self.agent_idx, axis=0))

            # 💡 [수정 없음] 렌더링 및 Trajectory 저장 로직은 그대로 사용합니다.
            if self.use_render:
                self.init_traj()

            if self.store_traj:
                self.traj_to_store = []
                self.traj_to_store.append(self.base_env.state.to_dict())

            # 💡 [수정] HSP 누적 보상을 self.num_agents에 맞게 초기화합니다.
            if self.use_hsp:
                self.cumulative_hidden_reward = np.zeros(self.num_agents)

            # 💡 [수정 없음] 공유 관측 생성 함수는 이미 일반화되었습니다.
            share_obs = self._gen_share_observation(self.base_env.state)
            
            # 💡 [수정] 모든 에이전트의 사용 가능 행동을 numpy 배열로 받습니다.
            available_actions = self._get_available_actions()
            
            # 💡 [수정] RL 에이전트의 사용 가능 행동이 0번 인덱스(row)에 오도록 재배열합니다.
            policy_available_actions = np.roll(available_actions, -self.agent_idx, axis=0)

            return policy_obs, share_obs, policy_available_actions

    def is_stuck(self, agent_id):
        if self.stuck_time == 0 or None in self.history_sa:
            return False, []
        history_s = [sa[0] for sa in self.history_sa]
        history_a = [sa[1][agent_id] for sa in self.history_sa[:-1]]  # last action is None
        player_s = [s.players[agent_id] for s in history_s]
        pos_and_ors = [p.pos_and_or for p in player_s]
        cur_po = pos_and_ors[-1]
        if all([po[0] == cur_po[0] and po[1] == cur_po[1] for po in pos_and_ors]):
            return True, history_a
        return False, []

    def init_traj(self):
        # return
        self.traj = {k: [] for k in DEFAULT_TRAJ_KEYS}
        for key in TIMESTEP_TRAJ_KEYS:
            self.traj[key].append([])

    def render(self):
        # raise NotImplementedError
        # try:
        save_dir = f"{self.run_dir}/gifs/rank_{self.rank}_traj_num_{self.traj_num}"
        save_dir = os.path.expanduser(save_dir)
        StateVisualizer().display_rendered_trajectory(self.traj, img_directory_path=save_dir, ipython_display=False)
        for img_path in os.listdir(save_dir):
            img_path = save_dir + "/" + img_path
        imgs = []
        imgs_dir = os.listdir(save_dir)
        # Filter only PNG files before sorting
        imgs_dir = [f for f in imgs_dir if f.endswith('.png')]
        imgs_dir = sorted(imgs_dir, key=lambda x: int(x.split(".")[0]))
        for img_path in imgs_dir:
            img_path = save_dir + "/" + img_path
            try:
                imgs.append(imageio.imread(img_path))
            except Exception as e:
                print(f"Warning: Failed to read image {img_path}: {e}")
                continue
        
        if not imgs:
            print("Warning: No images were successfully loaded, skipping gif creation")
        else:
            imageio.mimsave(save_dir + f'/reward_{self.traj["ep_returns"][0]}.gif', imgs, duration=0.05)
        imgs_dir = os.listdir(save_dir)
        for img_path in imgs_dir:
            img_path = save_dir + "/" + img_path
            if "png" in img_path:
                os.remove(img_path)
        # except Exception as e:
        #    print('failed to render traj: ', e)

    def fake_render(self):
        state = self.base_env.state
        mdp = self.base_mdp
        """String representation of the current state"""
        players_dict = {player.position: player for player in state.players}

        plt.cla()
        plt.clf()
        plt.axis([0, len(mdp.terrain_mtx[0]), 0, len(mdp.terrain_mtx)])
        grid_string = ""
        for y, terrain_row in enumerate(mdp.terrain_mtx):
            for x, element in enumerate(terrain_row):
                plt_x = x + 0.5
                plt_y = len(mdp.terrain_mtx) - y - 0.5
                plt_str = ""
                if (x, y) in players_dict.keys():
                    player = players_dict[(x, y)]
                    orientation = player.orientation
                    assert orientation in Direction.ALL_DIRECTIONS

                    grid_string += Action.ACTION_TO_CHAR[orientation]
                    plt_str = Action.ACTION_TO_CHAR[orientation]
                    player_object = player.held_object
                    if player_object:
                        grid_string += player_object.name[:1]
                        plt_str += player_object.name[:1]
                    else:
                        player_idx_lst = [i for i, p in enumerate(state.players) if p.position == player.position]
                        assert len(player_idx_lst) == 1
                        grid_string += str(player_idx_lst[0])
                        plt_str += str(player_idx_lst[0])
                else:
                    if element == "X" and state.has_object((x, y)):
                        state_obj = state.get_object((x, y))
                        grid_string = grid_string + element + state_obj.name[:1]
                        plt_str += element + state_obj.name[:1]

                    elif element == "P" and state.has_object((x, y)):
                        soup_obj = state.get_object((x, y))
                        soup_type, num_items, cook_time = soup_obj.state
                        if soup_type == "onion":
                            grid_string += "ø"
                            plt_str += "ø"
                        elif soup_type == "tomato":
                            grid_string += "†"
                            plt_str += "†"
                        else:
                            raise ValueError()

                        if num_items == mdp.num_items_for_soup:
                            grid_string += str(cook_time)
                            plt_str += str(cook_time)

                        # NOTE: do not currently have terminal graphics
                        # support for cooking times greater than 3.
                        elif num_items == 2:
                            grid_string += "="
                            plt_str += "="
                        else:
                            grid_string += "-"
                            plt_str += "-"
                    else:
                        grid_string += element + " "
                        plt_str = element
                plt.text(plt_x, plt_y, plt_str, ha="center", fontsize=30, alpha=0.4)
            grid_string += "\n"

        if state.order_list is not None:
            grid_string += "Current orders: {}/{} are any's\n".format(
                len(state.order_list),
                len([order == "any" for order in state.order_list]),
            )
        save_dir = f"{self.run_dir}/gifs/traj_num_{self.traj_num}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(f"{save_dir}/step_{self.step_count}.png")
        if self.step_count == self.episode_length:
            imgs = []
            for s in range(0, self.episode_length + 1):
                img = imageio.imread(f"{save_dir}/step_{s}.png")
                imgs.append(img)
            imageio.mimsave(f"{save_dir}/traj.gif", imgs, duration=0.1)
            imgs_dir = os.listdir(save_dir)
            for img_path in imgs_dir:
                img_path = save_dir + "/" + img_path
                if "png" in img_path:
                    os.remove(img_path)

        return grid_string

    def _store_trajectory(self):
        if not os.path.exists(f"{self.run_dir}/trajs/"):
            os.makedirs(f"{self.run_dir}/trajs/")
        save_dir = f"{self.run_dir}/trajs/traj_{self.rank}_{self.traj_num}.pkl"
        pickle.dump(self.traj_to_store, open(save_dir, "wb"))

    def seed(self, seed):
        setup_seed(seed)
        super().seed(seed)
