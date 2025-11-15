import itertools
import os
import pickle
import time

import numpy as np

from zsceval.envs.overcooked.overcooked_ai_py.data.planners import (
    PLANNERS_DIR,
    load_saved_action_manager,
)
from zsceval.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction
from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import (
    OvercookedGridworld,
    OvercookedState,
    PlayerState,
)
from zsceval.envs.overcooked.overcooked_ai_py.planning.search import Graph, SearchTree
from zsceval.envs.overcooked.overcooked_ai_py.utils import (
    manhattan_distance,
    pos_distance,
)

# Run planning logic with additional checks and
# computation to prevent or identify possible minor errors
SAFE_RUN = False


NO_COUNTERS_PARAMS = {
    "start_orientations": False,
    "wait_allowed": False,
    "counter_goals": [],
    "counter_drop": [],
    "counter_pickup": [],
    "same_motion_goals": True,
}

NO_COUNTERS_START_OR_PARAMS = {
    "start_orientations": True,
    "wait_allowed": False,
    "counter_goals": [],
    "counter_drop": [],
    "counter_pickup": [],
    "same_motion_goals": True,
}


class MotionPlanner:
    """A planner that computes optimal plans for a single agent to
    arrive at goal positions and orientations in an OvercookedGridworld.

    Args:
        mdp (OvercookedGridworld): gridworld of interest
        counter_goals (list): list of positions of counters we will consider
                              as valid motion goals
    """

    def __init__(self, mdp, counter_goals=[]):
        self.mdp = mdp

        # If positions facing counters should be
        # allowed as motion goals
        self.counter_goals = counter_goals

        # Graph problem that solves shortest path problem
        # between any position & orientation start-goal pair
        self.graph_problem = self._graph_from_grid()
        self.motion_goals_for_pos = self._get_goal_dict()

        self.all_plans = self._populate_all_plans()

    def get_plan(self, start_pos_and_or, goal_pos_and_or):
        """
        Returns pre-computed plan from initial agent position
        and orientation to a goal position and orientation.

        Args:
            start_pos_and_or (tuple): starting (pos, or) tuple
            goal_pos_and_or (tuple): goal (pos, or) tuple
        """
        plan_key = (start_pos_and_or, goal_pos_and_or)
        action_plan, pos_and_or_path, plan_cost = self.all_plans[plan_key]
        return action_plan, pos_and_or_path, plan_cost

    def get_gridworld_distance(self, start_pos_and_or, goal_pos_and_or):
        """Number of actions necessary to go from starting position
        and orientations to goal position and orientation (not including
        interaction action)"""
        assert self.is_valid_motion_start_goal_pair(
            start_pos_and_or, goal_pos_and_or
        ), "Goal position and orientation were not a valid motion goal"
        _, _, plan_cost = self.get_plan(start_pos_and_or, goal_pos_and_or)
        # Removing interaction cost
        return plan_cost - 1

    def get_gridworld_pos_distance(self, pos1, pos2):
        """Minimum (over possible orientations) number of actions necessary
        to go from starting position to goal position (not including
        interaction action)."""
        # NOTE: currently unused, pretty bad code. If used in future, clean up
        min_cost = np.Inf
        for d1, d2 in itertools.product(Direction.ALL_DIRECTIONS, repeat=2):
            start = (pos1, d1)
            end = (pos2, d2)
            if self.is_valid_motion_start_goal_pair(start, end):
                plan_cost = self.get_gridworld_distance(start, end)
                if plan_cost < min_cost:
                    min_cost = plan_cost
        return min_cost

    def _populate_all_plans(self):
        """Pre-computes all valid plans"""
        all_plans = {}
        valid_pos_and_ors = self.mdp.get_valid_player_positions_and_orientations()
        valid_motion_goals = filter(self.is_valid_motion_goal, valid_pos_and_ors)
        for start_motion_state, goal_motion_state in itertools.product(valid_pos_and_ors, valid_motion_goals):
            if not self.is_valid_motion_start_goal_pair(start_motion_state, goal_motion_state):
                continue
            action_plan, pos_and_or_path, plan_cost = self._compute_plan(start_motion_state, goal_motion_state)
            plan_key = (start_motion_state, goal_motion_state)
            all_plans[plan_key] = (action_plan, pos_and_or_path, plan_cost)
        return all_plans

    def is_valid_motion_start_goal_pair(self, start_pos_and_or, goal_pos_and_or, debug=False):
        if not self.is_valid_motion_goal(goal_pos_and_or):
            return False
        if not self.positions_are_connected(start_pos_and_or, goal_pos_and_or):
            return False
        return True

    def is_valid_motion_goal(self, goal_pos_and_or):
        """Checks that desired single-agent goal state (position and orientation)
        is reachable and is facing a terrain feature"""
        goal_position, goal_orientation = goal_pos_and_or
        if goal_position not in self.mdp.get_valid_player_positions():
            return False

        # Restricting goals to be facing a terrain feature
        pos_of_facing_terrain = Action.move_in_direction(goal_position, goal_orientation)
        facing_terrain_type = self.mdp.get_terrain_type_at_pos(pos_of_facing_terrain)
        if facing_terrain_type == " " or (
            facing_terrain_type == "X" and pos_of_facing_terrain not in self.counter_goals
        ):
            return False
        return True

    def _compute_plan(self, start_motion_state, goal_motion_state):
        """Computes optimal action plan for single agent movement

        Args:
            start_motion_state (tuple): starting positions and orientations
            positions_plan (list): positions path followed by agent
            goal_motion_state (tuple): goal positions and orientations
        """
        assert self.is_valid_motion_start_goal_pair(start_motion_state, goal_motion_state)
        positions_plan = self._get_position_plan_from_graph(start_motion_state, goal_motion_state)
        action_plan, pos_and_or_path, plan_length = self.action_plan_from_positions(
            positions_plan, start_motion_state, goal_motion_state
        )
        return action_plan, pos_and_or_path, plan_length

    def positions_are_connected(self, start_pos_and_or, goal_pos_and_or):
        return self.graph_problem.are_in_same_cc(start_pos_and_or, goal_pos_and_or)

    def _get_position_plan_from_graph(self, start_node, end_node):
        """Recovers positions to be reached by agent after the start node to reach the end node"""
        node_path = self.graph_problem.get_node_path(start_node, end_node)
        assert node_path[0] == start_node and node_path[-1] == end_node
        positions_plan = [state_node[0] for state_node in node_path[1:]]
        return positions_plan

    def action_plan_from_positions(self, position_list, start_motion_state, goal_motion_state):
        """
        Recovers an action plan reaches the goal motion position and orientation, and executes
        and interact action.

        Args:
            position_list (list): list of positions to be reached after the starting position
                                  (does not include starting position, but includes ending position)
            start_motion_state (tuple): starting position and orientation
            goal_motion_state (tuple): goal position and orientation

        Returns:
            action_plan (list): list of actions to reach goal state
            pos_and_or_path (list): list of (pos, or) pairs visited during plan execution
                                    (not including start, but including goal)
        """
        goal_position, goal_orientation = goal_motion_state
        action_plan, pos_and_or_path = [], []
        position_to_go = list(position_list)
        curr_pos, curr_or = start_motion_state

        # Get agent to goal position
        while position_to_go and curr_pos != goal_position:
            next_pos = position_to_go.pop(0)
            action = Action.determine_action_for_change_in_pos(curr_pos, next_pos)
            action_plan.append(action)
            curr_or = action if action != Action.STAY else curr_or
            pos_and_or_path.append((next_pos, curr_or))
            curr_pos = next_pos

        # Fix agent orientation if necessary
        if curr_or != goal_orientation:
            new_pos, _ = self.mdp._move_if_direction(curr_pos, curr_or, goal_orientation)
            assert new_pos == goal_position
            action_plan.append(goal_orientation)
            pos_and_or_path.append((goal_position, goal_orientation))

        # Add interact action
        action_plan.append(Action.INTERACT)
        pos_and_or_path.append((goal_position, goal_orientation))

        return action_plan, pos_and_or_path, len(action_plan)

    def _graph_from_grid(self):
        """Creates a graph adjacency matrix from an Overcooked MDP class."""
        state_decoder = {}
        for state_index, motion_state in enumerate(self.mdp.get_valid_player_positions_and_orientations()):
            state_decoder[state_index] = motion_state

        pos_encoder = {motion_state: state_index for state_index, motion_state in state_decoder.items()}
        num_graph_nodes = len(state_decoder)

        adjacency_matrix = np.zeros((num_graph_nodes, num_graph_nodes))
        for state_index, start_motion_state in state_decoder.items():
            for (
                action,
                successor_motion_state,
            ) in self._get_valid_successor_motion_states(start_motion_state):
                adj_pos_index = pos_encoder[successor_motion_state]
                adjacency_matrix[state_index][adj_pos_index] = self._graph_action_cost(action)

        return Graph(adjacency_matrix, pos_encoder, state_decoder)

    def _graph_action_cost(self, action):
        """Returns cost of a single-agent action"""
        assert action in Action.ALL_ACTIONS
        return 1

    def _get_valid_successor_motion_states(self, start_motion_state):
        """Get valid motion states one action away from the starting motion state."""
        start_position, start_orientation = start_motion_state
        return [
            (
                action,
                self.mdp._move_if_direction(start_position, start_orientation, action),
            )
            for action in Action.ALL_ACTIONS
        ]

    def min_cost_between_features(self, pos_list1, pos_list2, manhattan_if_fail=False):
        """
        Determines the minimum number of timesteps necessary for a player to go from any
        terrain feature in list1 to any feature in list2 and perform an interact action
        """
        min_dist = np.Inf
        min_manhattan = np.Inf
        for pos1, pos2 in itertools.product(pos_list1, pos_list2):
            for mg1, mg2 in itertools.product(self.motion_goals_for_pos[pos1], self.motion_goals_for_pos[pos2]):
                if not self.is_valid_motion_start_goal_pair(mg1, mg2):
                    if manhattan_if_fail:
                        pos0, pos1 = mg1[0], mg2[0]
                        curr_man_dist = manhattan_distance(pos0, pos1)
                        if curr_man_dist < min_manhattan:
                            min_manhattan = curr_man_dist
                    continue
                curr_dist = self.get_gridworld_distance(mg1, mg2)
                if curr_dist < min_dist:
                    min_dist = curr_dist

        # +1 to account for interaction action
        if manhattan_if_fail and min_dist == np.Inf:
            min_dist = min_manhattan
        min_cost = min_dist + 1
        return min_cost

    def min_cost_to_feature(self, start_pos_and_or, feature_pos_list, with_argmin=False, debug=False):
        """
        Determines the minimum number of timesteps necessary for a player to go from the starting
        position and orientation to any feature in feature_pos_list and perform an interact action
        """
        start_pos = start_pos_and_or[0]
        assert self.mdp.get_terrain_type_at_pos(start_pos) != "X"
        min_dist = np.Inf
        best_feature = None
        for feature_pos in feature_pos_list:
            for feature_goal in self.motion_goals_for_pos[feature_pos]:
                if not self.is_valid_motion_start_goal_pair(start_pos_and_or, feature_goal, debug=debug):
                    continue
                curr_dist = self.get_gridworld_distance(start_pos_and_or, feature_goal)
                if curr_dist < min_dist:
                    best_feature = feature_pos
                    min_dist = curr_dist
        # +1 to account for interaction action
        min_cost = min_dist + 1
        if with_argmin:
            # assert best_feature is not None, "{} vs {}".format(start_pos_and_or, feature_pos_list)
            return min_cost, best_feature
        return min_cost

    def _get_goal_dict(self):
        """Creates a dictionary of all possible goal states for all possible
        terrain features that the agent might want to interact with."""
        terrain_feature_locations = []
        for terrain_type, pos_list in self.mdp.terrain_pos_dict.items():
            if terrain_type != " ":
                terrain_feature_locations += pos_list
        return {
            feature_pos: self._get_possible_motion_goals_for_feature(feature_pos)
            for feature_pos in terrain_feature_locations
        }

    def _get_possible_motion_goals_for_feature(self, goal_pos):
        """Returns a list of possible goal positions (and orientations)
        that could be used for motion planning to get to goal_pos"""
        goals = []
        valid_positions = self.mdp.get_valid_player_positions()
        for d in Direction.ALL_DIRECTIONS:
            adjacent_pos = Action.move_in_direction(goal_pos, d)
            if adjacent_pos in valid_positions:
                goal_orientation = Direction.OPPOSITE_DIRECTIONS[d]
                motion_goal = (adjacent_pos, goal_orientation)
                goals.append(motion_goal)
        return goals


# 💡 JointMotionPlanner 클래스 전체를 아래 코드로 교체하세요.
class JointMotionPlanner(object):
    """A planner that computes optimal plans for N agents to
    arrive at goal positions and orientations in a OvercookedGridworld."""

    def __init__(self, mdp, params, debug=False):
        self.mdp = mdp
        self.params = params
        self.start_orientations = params["start_orientations"]
        self.same_motion_goals = params["same_motion_goals"]
        self.motion_planner = MotionPlanner(mdp, counter_goals=params["counter_goals"])
        self.joint_graph_problem = self._joint_graph_from_grid()
        self.all_plans = self._populate_all_plans(debug)

    def get_low_level_action_plan(self, start_jm_state, goal_jm_state):
        assert self.is_valid_joint_motion_pair(start_jm_state, goal_jm_state), f"start: {start_jm_state} \t end: {goal_jm_state} was not a valid motion goal pair"
        
        if self.start_orientations:
            plan_key = (start_jm_state, goal_jm_state)
        else:
            starting_positions = tuple(player_pos_and_or[0] for player_pos_and_or in start_jm_state)
            goal_positions = tuple(player_pos_and_or[0] for player_pos_and_or in goal_jm_state)
            
            # [수정] N명 중 2명이라도 목표가 같은지 확인
            goals_are_same = len(set(goal_positions)) != len(goal_positions)

            if any([s == g for s, g in zip(starting_positions, goal_positions)]) or (SAFE_RUN and goals_are_same):
                return self._obtain_plan(start_jm_state, goal_jm_state)

            dummy_orientation = Direction.NORTH
            dummy_start_jm_state = tuple((pos, dummy_orientation) for pos in starting_positions)
            plan_key = (dummy_start_jm_state, goal_jm_state)
        
        if plan_key not in self.all_plans:
             return None, None, None
        
        joint_action_plan, end_jm_state, plan_lengths = self.all_plans[plan_key]
        return joint_action_plan, end_jm_state, plan_lengths

    def _populate_all_plans(self, debug=False):
        all_plans = {}
        if self.start_orientations:
            valid_joint_start_states = self.mdp.get_valid_joint_player_positions_and_orientations()
        else:
            valid_joint_start_states = self.mdp.get_valid_joint_player_positions()

        valid_player_states = self.mdp.get_valid_player_positions_and_orientations()
        
        # [수정] repeat=2 -> repeat=self.mdp.num_players
        possible_joint_goal_states = list(itertools.product(valid_player_states, repeat=self.mdp.num_players))
        valid_joint_goal_states = list(filter(self.is_valid_joint_motion_goal, possible_joint_goal_states))

        if debug:
            print("Number of plans being pre-calculated: ", len(valid_joint_start_states) * len(valid_joint_goal_states))

        for joint_start_state, joint_goal_state in itertools.product(valid_joint_start_states, valid_joint_goal_states):
            if not self.start_orientations:
                dummy_orientation = Direction.NORTH
                joint_start_state = tuple((pos, dummy_orientation) for pos in joint_start_state)

            if not self.is_valid_jm_start_goal_pair(joint_start_state, joint_goal_state):
                continue

            joint_action_list, end_statuses, plan_lengths = self._obtain_plan(joint_start_state, joint_goal_state)
            if joint_action_list is not None:
                plan_key = (joint_start_state, joint_goal_state)
                all_plans[plan_key] = (joint_action_list, end_statuses, plan_lengths)
        return all_plans

    def is_valid_joint_motion_goal(self, joint_motion_goal):
        # [수정] 직접 set을 이용하여 N명 중 2명 이상이 같은 곳에 있는지 확인
        positions = [s[0] for s in joint_motion_goal]
        orientations = [s[1] for s in joint_motion_goal]
        if not self.same_motion_goals and len(set(positions)) != len(positions):
            return False
        return not self.mdp.is_joint_position_collision(positions)
        
    def is_valid_jm_start_goal_pair(self, joint_start_state, joint_goal_state):
        if not self.is_valid_joint_motion_goal(joint_goal_state):
            return False
        check_valid_fn = self.motion_planner.is_valid_motion_start_goal_pair
        # [수정] range(2) -> range(self.mdp.num_players)
        return all([check_valid_fn(joint_start_state[i], joint_goal_state[i]) for i in range(self.mdp.num_players)])

    def _obtain_plan(self, joint_start_state, joint_goal_state):
        action_plans, pos_and_or_paths, plan_lengths = self._get_plans_from_single_planner(joint_start_state, joint_goal_state)
        
        if any(p is None for p in pos_and_or_paths):
            return None, None, None

        have_conflict = self.plans_have_conflict(joint_start_state, joint_goal_state, pos_and_or_paths, plan_lengths)

        if not have_conflict:
            joint_action_plan, end_pos_and_orientations = self._join_single_agent_action_plans(pos_and_or_paths, plan_lengths)
            return joint_action_plan, end_pos_and_orientations, plan_lengths

        elif self._agents_are_in_same_position(joint_goal_state):
            return self._handle_path_conflict_with_same_goal(joint_start_state, joint_goal_state, action_plans, pos_and_or_paths)

        return self._compute_plan_from_joint_graph(joint_start_state, joint_goal_state)

    def _get_plans_from_single_planner(self, joint_start_state, joint_goal_state):
        single_agent_motion_plans = [self.motion_planner.get_plan(start, goal) for start, goal in zip(joint_start_state, joint_goal_state)]
        action_plans, pos_and_or_paths, plan_lengths = [], [], []
        
        for plan in single_agent_motion_plans:
            if plan is None:
                action_plans.append(None)
                pos_and_or_paths.append(None)
                plan_lengths.append(np.inf)
            else:
                action_plan, pos_and_or_path, plan_len = plan
                action_plans.append(action_plan)
                pos_and_or_paths.append(pos_and_or_path)
                plan_lengths.append(plan_len)
        
        return action_plans, pos_and_or_paths, plan_lengths

    # 💡 plans_have_conflict 함수 전체를 아래 코드로 교체하세요.
    def plans_have_conflict(self, joint_start_state, joint_goal_state, pos_and_or_paths, plan_lengths):
        """Check if the sequence of pos_and_or_paths for N agents conflict."""
        num_players = len(pos_and_or_paths)
        if num_players <= 1:
            return False

        min_length = min(plan_lengths)
        
        # t=0 시점의 시작 위치를 가져옵니다.
        prev_positions = tuple(s[0] for s in joint_start_state)

        for t in range(min_length):
            # [수정] t 시점의 모든 플레이어(N명)의 위치로 curr_positions 튜플을 동적으로 생성합니다.
            curr_positions = tuple(pos_and_or_paths[i][t][0] for i in range(num_players))
            # 💡 [디버깅] 충돌 확인 직전의 데이터를 출력합니다.
            print("--- DEBUG CONFLICT CHECK ---")
            print(f"Timestep: {t}")
            print(f"Num Players: {num_players}")
            print(f"Prev Positions (len={len(prev_positions)}): {prev_positions}")
            print(f"Curr Positions (len={len(curr_positions)}): {curr_positions}")
            
            # 이제 is_transition_collision 함수는 항상 올바른 길이의 튜플을 받습니다.
            if self.mdp.is_transition_collision(prev_positions, curr_positions):
                return True
            
            # 다음 루프를 위해 현재 위치를 이전 위치로 업데이트합니다.
            prev_positions = curr_positions
            
        return False
    # 💡 _join_single_agent_action_plans 함수를 교체하세요
    def _join_single_agent_action_plans(self, pos_and_or_paths, plan_lengths):
        """
        Joins N single agent action plans into a joint action plan.
        If plans are of different lengths, the shorter plans will be padded with STAY actions.
        """
        joint_action_plan = []
        # np.inf를 제외하고 실제 경로 길이 중 가장 긴 것을 찾습니다.
        max_plan_len = max(l for l in plan_lengths if l != np.inf)
        if max_plan_len == 0:
            return [], tuple(p[0] for p in pos_and_or_paths)

        for t in range(max_plan_len):
            joint_action = []
            for i in range(self.mdp.num_players):
                # 각 에이전트의 경로가 끝나면 STAY 행동을 추가합니다.
                action = pos_and_or_paths[i][t][1] if t < plan_lengths[i] else Action.STAY
                joint_action.append(action)
            joint_action_plan.append(tuple(joint_action))

        end_pos_and_ors = tuple(pos_and_or_paths[i][-1] for i in range(self.mdp.num_players))
        return joint_action_plan, end_pos_and_ors

    # 💡 _handle_path_conflict_with_same_goal 함수를 교체하세요
    def _handle_path_conflict_with_same_goal(self, joint_start_state, joint_goal_state, action_plans, pos_and_or_paths):
        """
        A simple conflict resolution scheme for N agents wanting to go to the same goal.
        The agent that arrives first moves, the others wait. This is not optimal but is a simple solution.
        """
        plan_lengths = [len(p) if p is not None else np.inf for p in action_plans]
        
        # 경로가 없는 에이전트가 있다면 처리 불가
        if any(l == np.inf for l in plan_lengths):
            return None, None, None

        first_arriver_idx = np.argmin(plan_lengths)
        
        joint_action_plan = []
        for t in range(plan_lengths[first_arriver_idx]):
            joint_action = []
            for i in range(self.mdp.num_players):
                action = action_plans[first_arriver_idx][t] if i == first_arriver_idx else Action.STAY
                joint_action.append(action)
            joint_action_plan.append(tuple(joint_action))

        final_pos_and_ors = list(joint_start_state)
        final_pos_and_ors[first_arriver_idx] = pos_and_or_paths[first_arriver_idx][-1]
        
        return joint_action_plan, tuple(final_pos_and_ors), plan_lengths

    # 💡 _handle_conflict_with_same_goal_idx 함수는 더 이상 필요 없으므로 삭제하거나 주석 처리하세요.
    # def _handle_conflict_with_same_goal_idx(...)

    # 💡 is_valid_joint_motion_goal 함수를 교체하세요
    def is_valid_joint_motion_goal(self, joint_goal_state):
        """Checks whether the goal joint positions and orientations are a valid goal for N players."""
        if not self.same_motion_goals and self._agents_are_in_same_position(joint_goal_state):
            return False

        # 모든 에이전트가 동일한 연결 공간(connected component)에 있는지 확인합니다.
        # 맵이 나뉘어 있는 경우를 대비합니다.
        if self.mdp.num_players > 1:
            all_in_same_cc = all(self.motion_planner.graph_problem.are_in_same_cc(joint_goal_state[0], joint_goal_state[i]) for i in range(1, self.mdp.num_players))
            multi_cc_map = len(self.motion_planner.graph_problem.connected_components) > 1
            if multi_cc_map and not all_in_same_cc:
                return False

        return all([self.motion_planner.is_valid_motion_goal(player_state) for player_state in joint_goal_state])

    # 💡 is_valid_joint_motion_pair 함수를 교체하세요
    def is_valid_joint_motion_pair(self, joint_start_state, joint_goal_state):
        if not self.is_valid_joint_motion_goal(joint_goal_state):
            return False
        return all(
            [
                self.motion_planner.is_valid_motion_start_goal_pair(joint_start_state[i], joint_goal_state[i])
                for i in range(self.mdp.num_players)
            ]
        )
    
    # 💡 _agents_are_in_same_position 함수는 이미 N-Player를 지원하므로 그대로 둡니다.
    def _agents_are_in_same_position(self, joint_motion_state):
        agent_positions = [player_pos_and_or[0] for player_pos_and_or in joint_motion_state]
        return len(agent_positions) != len(set(agent_positions))

    # 💡 _compute_plan_from_joint_graph 함수 전체를 아래 코드로 교체하세요.
    def _compute_plan_from_joint_graph(self, joint_start_state, joint_goal_state):
        """
        Compute joint action plan for N agents to achieve a
        certain position and orientation with the joint motion graph.
        """
        # [수정] 경로를 찾기 전에, 시작점과 목표점이 같은 연결 공간에 있는지 먼저 확인합니다.
        if not self.joint_graph_problem.are_in_same_cc(joint_start_state, joint_goal_state):
            return None, None, None

        # [수정] 위치만 사용하는 대신, '위치+방향'의 완전한 상태를 키로 사용합니다.
        state_path = self.joint_graph_problem.get_node_path(joint_start_state, joint_goal_state)
        
        if state_path is None or len(state_path) < 2:
            return None, None, None

        action_path = []
        for i in range(len(state_path) - 1):
            curr_state, next_state = state_path[i], state_path[i+1]
            # [수정] .encoder -> ._encoder 로 변경
            encoded_curr = self.joint_graph_problem._encoder[curr_state]
            encoded_next = self.joint_graph_problem._encoder[next_state]
            
            if encoded_next not in self.joint_graph_problem.adj_dict[encoded_curr]:
                return None, None, None

            actions = self.joint_graph_problem.adj_dict[encoded_curr][encoded_next]
            action_path.append(actions[0])
        
        plan_lengths = [len(action_path)] * self.mdp.num_players
        return action_path, state_path[-1], plan_lengths

    # 💡 joint_action_plan_from_positions 함수를 교체하세요
    def joint_action_plan_from_positions(self, joint_positions, joint_start_state, joint_goal_state):
        """Finds an action plan for N agents"""
        action_plans = []
        for i in range(self.mdp.num_players):
            agent_position_sequence = [joint_position[i] for joint_position in joint_positions]
            action_plan, _, _ = self.motion_planner.action_plan_from_positions(
                agent_position_sequence, joint_start_state[i], joint_goal_state[i]
            )
            action_plans.append(action_plan)

        finishing_times = tuple(len(plan) for plan in action_plans)
        joint_action_plan, end_pos_and_orientations = self._join_single_agent_action_plans(None, action_plans, finishing_times) # pos_and_or_paths not needed here
        return joint_action_plan, end_pos_and_orientations, finishing_times

    # 💡 _joint_graph_from_grid 함수 전체를 아래의 최종 코드로 교체하세요.
    def _joint_graph_from_grid(self):
        """
        Calculates the joint motion graph.
        The nodes are all valid joint positions AND ORIENTATIONS, 
        and edges are valid joint actions.
        """
        # 1. 그래프의 모든 노드가 될 '상태(위치+방향)' 목록을 가져옵니다.
        all_joint_motion_states = self.mdp.get_valid_joint_player_positions_and_orientations()

        # 2. encoder와 decoder(암호표)를 생성합니다.
        state_encoder = {state: i for i, state in enumerate(all_joint_motion_states)}
        state_decoder = {i: state for state, i in state_encoder.items()}

        # 그래프는 이제 복잡한 상태 대신 간단한 숫자 인덱스를 사용합니다.
        joint_graph_int_indices = {}

        # 3. 모든 상태에 대해 루프를 돌면서 그래프를 구축합니다.
        for start_jm_state, start_node_index in state_encoder.items():
            
            # 현재 상태에서 갈 수 있는 다음 상태들을 계산합니다.
            successor_jm_states_dict = self._get_valid_successor_joint_positions(start_jm_state)
            
            # 다음 상태들을 숫자 인덱스로 변환합니다.
            successor_node_indices_dict = {}
            for successor_jm_state, actions in successor_jm_states_dict.items():
                if successor_jm_state in state_encoder:
                    successor_node_index = state_encoder[successor_jm_state]
                    successor_node_indices_dict[successor_node_index] = actions
            
            joint_graph_int_indices[start_node_index] = successor_node_indices_dict
        
        # 4. 최종적으로 그래프와 encoder, decoder를 모두 전달하여 Graph 객체를 생성합니다.
        return Graph(joint_graph_int_indices, state_encoder, state_decoder)

    def _graph_joint_action_cost(self, joint_action):
        """The cost used in the graph shortest-path problem for a certain joint-action"""
        num_of_non_stay_actions = len([a for a in joint_action if a != Action.STAY])
        # NOTE: Removing the possibility of having 0 cost joint_actions
        if num_of_non_stay_actions == 0:
            return 1
        return num_of_non_stay_actions

    # 💡 _get_valid_successor_joint_positions 함수 전체를 아래 코드로 교체
    def _get_valid_successor_joint_positions(self, start_jm_state):
        """
        Computes all successor joint motion states (pos_and_or) from a given start state.
        NOTE: The name of the function is misleading, it should be _get_valid_successor_joint_motion_states
        """
        successor_motion_states = {}
        
        # 모든 가능한 합동 행동 조합을 생성합니다.
        all_joint_actions = list(itertools.product(Action.ALL_ACTIONS, repeat=self.mdp.num_players))

        for joint_action in all_joint_actions:
            # 시작 위치와 방향으로부터 더미 플레이어 상태를 생성합니다.
            dummy_player_states = self.mdp.get_players_from_positions_and_orientations(start_jm_state)

            # 다음 위치와 '방향'을 모두 계산합니다.
            new_positions, new_orientations = self.mdp.compute_new_positions_and_orientations(
                dummy_player_states, joint_action
            )
            
            # 다음 상태를 (위치, 방향) 쌍의 튜플로 만듭니다.
            new_jm_state = tuple(zip(new_positions, new_orientations))

            # 이 튜플을 딕셔너리의 키로 사용하여 에러를 해결합니다.
            if new_jm_state not in successor_motion_states:
                successor_motion_states[new_jm_state] = []
            successor_motion_states[new_jm_state].append(joint_action)

        return successor_motion_states

    def derive_state(self, start_state, end_pos_and_ors, action_plans):
        """
        Given a start state, end position and orientations, and an action plan, recovers
        the resulting state without executing the entire plan.
        """
        if len(action_plans) == 0:
            return start_state

        end_state = start_state.deepcopy()
        end_players = []
        for player, end_pos_and_or in zip(end_state.players, end_pos_and_ors):
            new_player = player.deepcopy()
            position, orientation = end_pos_and_or
            new_player.update_pos_and_or(position, orientation)
            end_players.append(new_player)

        end_state.players = tuple(end_players)

        # Resolve environment effects for t - 1 turns
        plan_length = len(action_plans)
        assert plan_length > 0
        for _ in range(plan_length - 1):
            self.mdp.step_environment_effects(end_state)

        # Interacts
        last_joint_action = tuple(a if a == Action.INTERACT else Action.STAY for a in action_plans[-1])

        self.mdp.resolve_interacts(end_state, last_joint_action)
        self.mdp.resolve_movement(end_state, last_joint_action)
        self.mdp.step_environment_effects(end_state)
        return end_state


class MediumLevelActionManager:
    """
    Manager for medium level actions (specific joint motion goals).
    Determines available medium level actions for each state.

    Args:
        mdp (OvercookedGridWorld): gridworld of interest
        start_orientations (bool): whether the JointMotionPlanner should store plans for
                                   all starting positions & orientations or just for unique
                                   starting positions
    """

    def __init__(self, mdp, params):
        self.mdp = mdp

        self.params = params
        self.wait_allowed = params["wait_allowed"]
        self.counter_drop = params["counter_drop"]
        self.counter_pickup = params["counter_pickup"]

        self.joint_motion_planner = JointMotionPlanner(mdp, params)
        self.motion_planner = self.joint_motion_planner.motion_planner

    def save_to_file(self, filename):
        with open(filename, "wb") as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    # 💡 joint_ml_actions 함수 전체를 아래 코드로 교체하세요.
    def joint_ml_actions(self, state):
        """Determine all possible joint medium level actions for a certain state"""
        # 1. 모든 플레이어의 가능한 행동들을 리스트의 리스트 형태로 받습니다.
        # 예: [[p1_action1, p1_action2], [p2_action1], [p3_action1, ...]]
        all_agents_actions = [self.get_medium_level_actions(state, player) for player in state.players]

        # 2. * (splat) 연산자를 사용하여 모든 플레이어의 행동 리스트를 product에 전달합니다.
        joint_ml_actions = list(itertools.product(*all_agents_actions))

        # ml actions are nothing but specific joint motion goals
        valid_joint_ml_actions = list(filter(lambda a: self.is_valid_ml_action(state, a), joint_ml_actions))

        # HACK: Could cause things to break... (이하 동일 로직)
        if len(valid_joint_ml_actions) == 0:
            all_agents_actions = [
                self.get_medium_level_actions(state, player, waiting_substitute=True) for player in state.players
            ]
            joint_ml_actions = list(itertools.product(*all_agents_actions))
            valid_joint_ml_actions = list(filter(lambda a: self.is_valid_ml_action(state, a), joint_ml_actions))
            if len(valid_joint_ml_actions) == 0:
                print(
                    "WARNING: Found state without valid actions even after adding waiting substitute actions. State: {}".format(
                        state
                    )
                )
        return valid_joint_ml_actions

    def is_valid_ml_action(self, state, ml_action):
        return self.joint_motion_planner.is_valid_jm_start_goal_pair(state.players_pos_and_or, ml_action)

    def get_medium_level_actions(self, state, player, waiting_substitute=False):
        """
        Determine valid medium level actions for a player.

        Args:
            state (OvercookedState): current state
            waiting_substitute (bool): add a substitute action that takes the place of
                                       a waiting action (going to closest feature)

        Returns:
            player_actions (list): possible motion goals (pairs of goal positions and orientations)
        """
        player_actions = []
        counter_pickup_objects = self.mdp.get_counter_objects_dict(state, self.counter_pickup)
        if not player.has_object():
            onion_pickup = self.pickup_onion_actions(state, counter_pickup_objects)
            tomato_pickup = self.pickup_tomato_actions(state, counter_pickup_objects)
            dish_pickup = self.pickup_dish_actions(state, counter_pickup_objects)
            soup_pickup = self.pickup_counter_soup_actions(state, counter_pickup_objects)
            player_actions.extend(onion_pickup + tomato_pickup + dish_pickup + soup_pickup)

        else:
            player_object = player.get_object()
            pot_states_dict = self.mdp.get_pot_states(state)

            # No matter the object, we can place it on a counter
            if len(self.counter_drop) > 0:
                player_actions.extend(self.place_obj_on_counter_actions(state))

            if player_object.name == "soup":
                player_actions.extend(self.deliver_soup_actions())
            elif player_object.name == "onion":
                player_actions.extend(self.put_onion_in_pot_actions(pot_states_dict))
            elif player_object.name == "tomato":
                player_actions.extend(self.put_tomato_in_pot_actions(pot_states_dict))
            elif player_object.name == "dish":
                # Not considering all pots (only ones close to ready) to reduce computation
                # NOTE: could try to calculate which pots are eligible, but would probably take
                # a lot of compute
                player_actions.extend(self.pickup_soup_with_dish_actions(pot_states_dict, only_nearly_ready=False))
            else:
                raise ValueError("Unrecognized object")

        if self.wait_allowed:
            player_actions.extend(self.wait_actions(player))

        if waiting_substitute:
            # Trying to mimic a "WAIT" action by adding the closest allowed feature to the avaliable actions
            # This is because motion plans that aren't facing terrain features (non counter, non empty spots)
            # are not considered valid
            player_actions.extend(self.go_to_closest_feature_actions(player))

        is_valid_goal_given_start = lambda goal: self.motion_planner.is_valid_motion_start_goal_pair(
            player.pos_and_or, goal
        )
        player_actions = list(filter(is_valid_goal_given_start, player_actions))
        return player_actions

    def pickup_onion_actions(self, state, counter_objects):
        onion_dispenser_locations = self.mdp.get_onion_dispenser_locations()
        onion_pickup_locations = onion_dispenser_locations + counter_objects["onion"]
        return self._get_ml_actions_for_positions(onion_pickup_locations)

    def pickup_tomato_actions(self, state, counter_objects):
        tomato_dispenser_locations = self.mdp.get_tomato_dispenser_locations()
        tomato_pickup_locations = tomato_dispenser_locations + counter_objects["tomato"]
        return self._get_ml_actions_for_positions(tomato_pickup_locations)

    def pickup_dish_actions(self, state, counter_objects):
        dish_dispenser_locations = self.mdp.get_dish_dispenser_locations()
        dish_pickup_locations = dish_dispenser_locations + counter_objects["dish"]
        return self._get_ml_actions_for_positions(dish_pickup_locations)

    def pickup_counter_soup_actions(self, state, counter_objects):
        soup_pickup_locations = counter_objects["soup"]
        return self._get_ml_actions_for_positions(soup_pickup_locations)

    def place_obj_on_counter_actions(self, state):
        all_empty_counters = set(self.mdp.get_empty_counter_locations(state))
        valid_empty_counters = [c_pos for c_pos in self.counter_drop if c_pos in all_empty_counters]
        return self._get_ml_actions_for_positions(valid_empty_counters)

    def deliver_soup_actions(self):
        serving_locations = self.mdp.get_serving_locations()
        return self._get_ml_actions_for_positions(serving_locations)

    def put_onion_in_pot_actions(self, pot_states_dict):
        partially_full_onion_pots = pot_states_dict["onion"]["partially_full"]
        fillable_pots = partially_full_onion_pots + pot_states_dict["empty"]
        return self._get_ml_actions_for_positions(fillable_pots)

    def put_tomato_in_pot_actions(self, pot_states_dict):
        partially_full_tomato_pots = pot_states_dict["tomato"]["partially_full"]
        fillable_pots = partially_full_tomato_pots + pot_states_dict["empty"]
        return self._get_ml_actions_for_positions(fillable_pots)

    def pickup_soup_with_dish_actions(self, pot_states_dict, only_nearly_ready=False):
        ready_pot_locations = pot_states_dict["onion"]["ready"] + pot_states_dict["tomato"]["ready"]
        nearly_ready_pot_locations = pot_states_dict["onion"]["cooking"] + pot_states_dict["tomato"]["cooking"]
        if not only_nearly_ready:
            partially_full_pots = (
                pot_states_dict["tomato"]["partially_full"] + pot_states_dict["onion"]["partially_full"]
            )
            nearly_ready_pot_locations = nearly_ready_pot_locations + pot_states_dict["empty"] + partially_full_pots
        return self._get_ml_actions_for_positions(ready_pot_locations + nearly_ready_pot_locations)

    def go_to_closest_feature_actions(self, player):
        feature_locations = (
            self.mdp.get_onion_dispenser_locations()
            + self.mdp.get_tomato_dispenser_locations()
            + self.mdp.get_pot_locations()
            + self.mdp.get_dish_dispenser_locations()
        )
        closest_feature_pos = self.motion_planner.min_cost_to_feature(
            player.pos_and_or, feature_locations, with_argmin=True
        )[1]
        return self._get_ml_actions_for_positions([closest_feature_pos])

    def wait_actions(self, player):
        waiting_motion_goal = (player.position, player.orientation)
        return [waiting_motion_goal]

    def _get_ml_actions_for_positions(self, positions_list):
        """Determine what are the ml actions (joint motion goals) for a list of positions

        Args:
            positions_list (list): list of target terrain feature positions
        """
        possible_motion_goals = []
        for pos in positions_list:
            # All possible ways to reach the target feature
            for motion_goal in self.joint_motion_planner.motion_planner.motion_goals_for_pos[pos]:
                possible_motion_goals.append(motion_goal)
        return possible_motion_goals


class MediumLevelPlanner:
    """
    A planner that computes optimal plans for two agents to deliver a certain number of dishes
    in an OvercookedGridworld using medium level actions (single motion goals) in the corresponding
    A* search problem.
    """

    def __init__(self, mdp, mlp_params, ml_action_manager=None):
        self.mdp = mdp
        self.params = mlp_params
        self.ml_action_manager = ml_action_manager if ml_action_manager else MediumLevelActionManager(mdp, mlp_params)
        self.jmp = self.ml_action_manager.joint_motion_planner
        self.mp = self.jmp.motion_planner

    @staticmethod
    def from_action_manager_file(filename):
        mlp_action_manager = load_saved_action_manager(filename)
        mdp = mlp_action_manager.mdp
        params = mlp_action_manager.params
        return MediumLevelPlanner(mdp, params, mlp_action_manager)

    @staticmethod
    def from_pickle_or_compute(mdp, mlp_params, custom_filename=None, force_compute=False):
        assert isinstance(mdp, OvercookedGridworld)

        filename = custom_filename if custom_filename is not None else mdp.layout_name + "_am.pkl"

        if force_compute:
            return MediumLevelPlanner.compute_mlp(filename, mdp, mlp_params)

        try:
            mlp = MediumLevelPlanner.from_action_manager_file(filename)
        except (FileNotFoundError, ModuleNotFoundError, EOFError) as e:
            print("Recomputing planner due to:", e)
            return MediumLevelPlanner.compute_mlp(filename, mdp, mlp_params)

        if mlp.ml_action_manager.params != mlp_params or mlp.mdp != mdp:
            print("Mlp with different params or mdp found, computing from scratch")
            return MediumLevelPlanner.compute_mlp(filename, mdp, mlp_params)

        # print("Loaded MediumLevelPlanner from {}".format(os.path.join(PLANNERS_DIR, filename)))
        return mlp

    @staticmethod
    def compute_mlp(filename, mdp, mlp_params):
        final_filepath = os.path.join(PLANNERS_DIR, filename)
        print(f"Computing MediumLevelPlanner to be saved in {final_filepath}")
        start_time = time.time()
        mlp = MediumLevelPlanner(mdp, mlp_params=mlp_params)
        print(f"It took {time.time() - start_time} seconds to create mlp")
        mlp.ml_action_manager.save_to_file(final_filepath)
        return mlp

    def get_low_level_action_plan(self, start_state, h_fn, delivery_horizon=4, debug=False, goal_info=False):
        """
        Get a plan of joint-actions executable in the environment that will lead to a goal number of deliveries

        Args:
            state (OvercookedState): starting state

        Returns:
            full_joint_action_plan (list): joint actions to reach goal
        """
        start_state = start_state.deepcopy()
        ml_plan, cost = self.get_ml_plan(start_state, h_fn, delivery_horizon=delivery_horizon, debug=debug)

        if start_state.order_list is None:
            start_state.order_list = ["any"] * delivery_horizon

        full_joint_action_plan = self.get_low_level_plan_from_ml_plan(
            start_state, ml_plan, h_fn, debug=debug, goal_info=goal_info
        )
        assert cost == len(full_joint_action_plan), "A* cost {} but full joint action plan cost {}".format(
            cost, len(full_joint_action_plan)
        )
        if debug:
            print(f"Found plan with cost {cost}")
        return full_joint_action_plan

    def get_low_level_plan_from_ml_plan(self, start_state, ml_plan, heuristic_fn, debug=False, goal_info=False):
        t = 0
        full_joint_action_plan = []
        curr_state = start_state
        curr_motion_state = start_state.players_pos_and_or
        prev_h = heuristic_fn(start_state, t, debug=False)

        if len(ml_plan) > 0 and goal_info:
            print("First motion goal: ", ml_plan[0][0])

        if debug:
            print("Start state")
            OvercookedEnv.print_state(self.mdp, start_state)

        for joint_motion_goal, goal_state in ml_plan:
            (
                joint_action_plan,
                end_motion_state,
                plan_costs,
            ) = self.ml_action_manager.joint_motion_planner.get_low_level_action_plan(
                curr_motion_state, joint_motion_goal
            )
            curr_plan_cost = min(plan_costs)
            full_joint_action_plan.extend(joint_action_plan)
            t += 1

            if debug:
                print(t)
                OvercookedEnv.print_state(self.mdp, goal_state)

            if SAFE_RUN:
                s_prime, _ = OvercookedEnv.execute_plan(self.mdp, curr_state, joint_action_plan)
                assert s_prime == goal_state

            curr_h = heuristic_fn(goal_state, t, debug=False)
            self.check_heuristic_consistency(curr_h, prev_h, curr_plan_cost)
            curr_motion_state, prev_h, curr_state = end_motion_state, curr_h, goal_state
        return full_joint_action_plan

    def check_heuristic_consistency(self, curr_heuristic_val, prev_heuristic_val, actual_edge_cost):
        delta_h = curr_heuristic_val - prev_heuristic_val
        assert (
            actual_edge_cost >= delta_h
        ), "Heuristic was not consistent. \n Prev h: {}, Curr h: {}, Actual cost: {}, Δh: {}".format(
            prev_heuristic_val, curr_heuristic_val, actual_edge_cost, delta_h
        )

    def get_ml_plan(self, start_state, h_fn, delivery_horizon=4, debug=False):
        """
        Solves A* Search problem to find optimal sequence of medium level actions
        to reach the goal number of deliveries

        Returns:
            ml_plan (list): plan not including starting state in form
                [(joint_action, successor_state), ..., (joint_action, goal_state)]
            cost (int): A* Search cost
        """
        start_state = start_state.deepcopy()
        if start_state.order_list is None:
            start_state.order_list = ["any"] * delivery_horizon
        else:
            start_state.order_list = start_state.order_list[:delivery_horizon]

        expand_fn = lambda state: self.get_successor_states(state)
        goal_fn = lambda state: state.num_orders_remaining == 0
        heuristic_fn = lambda state: h_fn(state)

        search_problem = SearchTree(start_state, goal_fn, expand_fn, heuristic_fn, debug=debug)
        ml_plan, cost = search_problem.A_star_graph_search(info=True)
        return ml_plan[1:], cost

    def get_successor_states(self, start_state):
        """Successor states for medium-level actions are defined as
        the first state in the corresponding motion plan in which
        one of the two agents' subgoals is satisfied.

        Returns: list of
            joint_motion_goal: ((pos1, or1), (pos2, or2)) specifying the
                                motion plan goal for both agents

            successor_state:   OvercookedState corresponding to state
                               arrived at after executing part of the motion plan
                               (until one of the agents arrives at his goal status)

            plan_length:       Time passed until arrival to the successor state
        """
        if self.mdp.is_terminal(start_state):
            return []

        start_jm_state = start_state.players_pos_and_or
        successor_states = []
        for goal_jm_state in self.ml_action_manager.joint_ml_actions(start_state):
            (
                joint_motion_action_plans,
                end_pos_and_ors,
                plan_costs,
            ) = self.jmp.get_low_level_action_plan(start_jm_state, goal_jm_state)
            end_state = self.jmp.derive_state(start_state, end_pos_and_ors, joint_motion_action_plans)

            if SAFE_RUN:
                assert end_pos_and_ors[0] == goal_jm_state[0] or end_pos_and_ors[1] == goal_jm_state[1]
                s_prime, _ = OvercookedEnv.execute_plan(self.mdp, start_state, joint_motion_action_plans, display=False)
                assert end_state == s_prime, [
                    OvercookedEnv.print_state(self.mdp, s_prime),
                    OvercookedEnv.print_state(self.mdp, end_state),
                ]

            successor_states.append((goal_jm_state, end_state, min(plan_costs)))
        return successor_states

    def get_successor_states_fixed_other(self, start_state, other_agent, other_agent_idx):
        """
        Get the successor states of a given start state, assuming that the other agent is fixed and will act according to the passed in model
        """
        if self.mdp.is_terminal(start_state):
            return []

        player = start_state.players[1 - other_agent_idx]
        ml_actions = self.ml_action_manager.get_medium_level_actions(start_state, player)

        if len(ml_actions) == 0:
            ml_actions = self.ml_action_manager.get_medium_level_actions(start_state, player, waiting_substitute=True)

        successor_high_level_states = []
        for ml_action in ml_actions:
            action_plan, end_state, cost = self.get_embedded_low_level_action_plan(
                start_state, ml_action, other_agent, other_agent_idx
            )

            if not self.mdp.is_terminal(end_state):
                # Adding interact action and deriving last state
                other_agent_action = other_agent.action(end_state)
                last_joint_action = (
                    (Action.INTERACT, other_agent_action)
                    if other_agent_idx == 1
                    else (other_agent_action, Action.INTERACT)
                )
                action_plan = action_plan + (last_joint_action,)
                cost = cost + 1

                end_state, _ = self.embedded_mdp_step(
                    end_state,
                    Action.INTERACT,
                    other_agent_action,
                    other_agent.agent_index,
                )

            successor_high_level_states.append((action_plan, end_state, cost))
        return successor_high_level_states

    def get_embedded_low_level_action_plan(self, state, goal_pos_and_or, other_agent, other_agent_idx):
        """Find action plan for a specific motion goal with A* considering the other agent"""
        other_agent.set_agent_index(other_agent_idx)
        agent_idx = 1 - other_agent_idx

        expand_fn = lambda state: self.embedded_mdp_succ_fn(state, other_agent)
        goal_fn = (
            lambda state: state.players[agent_idx].pos_and_or == goal_pos_and_or or state.num_orders_remaining == 0
        )
        heuristic_fn = lambda state: sum(pos_distance(state.players[agent_idx].position, goal_pos_and_or[0]))

        search_problem = SearchTree(state, goal_fn, expand_fn, heuristic_fn)
        state_action_plan, cost = search_problem.A_star_graph_search(info=False)
        action_plan, state_plan = zip(*state_action_plan)
        action_plan = action_plan[1:]
        end_state = state_plan[-1]
        return action_plan, end_state, cost

    def embedded_mdp_succ_fn(self, state, other_agent):
        other_agent_action = other_agent.action(state)

        successors = []
        for a in Action.ALL_ACTIONS:
            successor_state, joint_action = self.embedded_mdp_step(
                state, a, other_agent_action, other_agent.agent_index
            )
            cost = 1
            successors.append((joint_action, successor_state, cost))
        return successors

    def embedded_mdp_step(self, state, action, other_agent_action, other_agent_index):
        if other_agent_index == 0:
            joint_action = (other_agent_action, action)
        else:
            joint_action = (action, other_agent_action)
        if not self.mdp.is_terminal(state):
            results, _, _ = self.mdp.get_state_transition(state, joint_action)
            successor_state = results
        else:
            print("Tried to find successor of terminal")
            assert False, f"state {state} \t action {action}"
            successor_state = state
        return successor_state, joint_action


class HighLevelAction:
    """A high level action is given by a set of subsequent motion goals"""

    def __init__(self, motion_goals):
        self.motion_goals = motion_goals

    def _check_valid(self):
        for goal in self.motion_goals:
            assert len(goal) == 2
            pos, orient = goal
            assert orient in Direction.ALL_DIRECTIONS
            assert type(pos) is tuple
            assert len(pos) == 2

    def __getitem__(self, i):
        """Get ith motion goal of the HL Action"""
        return self.motion_goals[i]


class HighLevelActionManager:
    """
    Manager for high level actions. Determines available high level actions
    for each state and player.
    """

    def __init__(self, medium_level_planner):
        self.mdp = medium_level_planner.mdp

        self.wait_allowed = medium_level_planner.params["wait_allowed"]
        self.counter_drop = medium_level_planner.params["counter_drop"]
        self.counter_pickup = medium_level_planner.params["counter_pickup"]

        self.mlp = medium_level_planner
        self.ml_action_manager = medium_level_planner.ml_action_manager
        self.mp = medium_level_planner.mp

    def joint_hl_actions(self, state):
        hl_actions_a0, hl_actions_a1 = tuple(self.get_high_level_actions(state, player) for player in state.players)
        joint_hl_actions = list(itertools.product(hl_actions_a0, hl_actions_a1))

        assert self.mlp.params["same_motion_goals"]
        valid_joint_hl_actions = joint_hl_actions

        if len(valid_joint_hl_actions) == 0:
            print("WARNING: found a state without high level successors")
        return valid_joint_hl_actions

    def get_high_level_actions(self, state, player):
        player_hl_actions = []
        counter_pickup_objects = self.mdp.get_counter_objects_dict(state, self.counter_pickup)
        if player.has_object():
            place_obj_ml_actions = self.ml_action_manager.get_medium_level_actions(state, player)

            # HACK to prevent some states not having successors due to lack of waiting actions
            if len(place_obj_ml_actions) == 0:
                place_obj_ml_actions = self.ml_action_manager.get_medium_level_actions(
                    state, player, waiting_substitute=True
                )

            place_obj_hl_actions = [HighLevelAction([ml_action]) for ml_action in place_obj_ml_actions]
            player_hl_actions.extend(place_obj_hl_actions)
        else:
            pot_states_dict = self.mdp.get_pot_states(state)
            player_hl_actions.extend(self.get_onion_and_put_in_pot(state, counter_pickup_objects, pot_states_dict))
            player_hl_actions.extend(self.get_tomato_and_put_in_pot(state, counter_pickup_objects, pot_states_dict))
            player_hl_actions.extend(self.get_dish_and_soup_and_serve(state, counter_pickup_objects, pot_states_dict))
        return player_hl_actions

    def get_dish_and_soup_and_serve(self, state, counter_objects, pot_states_dict):
        """Get all sequences of medium-level actions (hl actions) that involve a player getting a dish,
        going to a pot and picking up a soup, and delivering the soup."""
        dish_pickup_actions = self.ml_action_manager.pickup_dish_actions(state, counter_objects)
        pickup_soup_actions = self.ml_action_manager.pickup_soup_with_dish_actions(pot_states_dict)
        deliver_soup_actions = self.ml_action_manager.deliver_soup_actions()
        hl_level_actions = list(itertools.product(dish_pickup_actions, pickup_soup_actions, deliver_soup_actions))
        return [HighLevelAction(hl_action_list) for hl_action_list in hl_level_actions]

    def get_onion_and_put_in_pot(self, state, counter_objects, pot_states_dict):
        """Get all sequences of medium-level actions (hl actions) that involve a player getting an onion
        from a dispenser and placing it in a pot."""
        onion_pickup_actions = self.ml_action_manager.pickup_onion_actions(state, counter_objects)
        put_in_pot_actions = self.ml_action_manager.put_onion_in_pot_actions(pot_states_dict)
        hl_level_actions = list(itertools.product(onion_pickup_actions, put_in_pot_actions))
        return [HighLevelAction(hl_action_list) for hl_action_list in hl_level_actions]

    def get_tomato_and_put_in_pot(self, state, counter_objects, pot_states_dict):
        """Get all sequences of medium-level actions (hl actions) that involve a player getting an tomato
        from a dispenser and placing it in a pot."""
        tomato_pickup_actions = self.ml_action_manager.pickup_tomato_actions(state, counter_objects)
        put_in_pot_actions = self.ml_action_manager.put_tomato_in_pot_actions(pot_states_dict)
        hl_level_actions = list(itertools.product(tomato_pickup_actions, put_in_pot_actions))
        return [HighLevelAction(hl_action_list) for hl_action_list in hl_level_actions]


class HighLevelPlanner:
    """A planner that computes optimal plans for two agents to
    deliver a certain number of dishes in an OvercookedGridworld
    using high level actions in the corresponding A* search problems
    """

    def __init__(self, hl_action_manager):
        self.hl_action_manager = hl_action_manager
        self.mlp = self.hl_action_manager.mlp
        self.jmp = self.mlp.ml_action_manager.joint_motion_planner
        self.mp = self.jmp.motion_planner
        self.mdp = self.mlp.mdp

    def get_successor_states(self, start_state):
        """Determines successor states for high-level actions"""
        successor_states = []

        if self.mdp.is_terminal(start_state):
            return successor_states

        for joint_hl_action in self.hl_action_manager.joint_hl_actions(start_state):
            _, end_state, hl_action_cost = self.perform_hl_action(joint_hl_action, start_state)

            successor_states.append((joint_hl_action, end_state, hl_action_cost))
        return successor_states

    def perform_hl_action(self, joint_hl_action, curr_state):
        """Determines the end state for a high level action, and the corresponding low level action plan and cost.
        Will return Nones if a pot exploded throughout the execution of the action"""
        full_plan = []
        motion_goal_indices = (0, 0)
        total_cost = 0
        while not self.at_least_one_finished_hl_action(joint_hl_action, motion_goal_indices):
            curr_jm_goal = tuple(joint_hl_action[i].motion_goals[motion_goal_indices[i]] for i in range(2))
            (
                joint_motion_action_plans,
                end_pos_and_ors,
                plan_costs,
            ) = self.jmp.get_low_level_action_plan(curr_state.players_pos_and_or, curr_jm_goal)
            curr_state = self.jmp.derive_state(curr_state, end_pos_and_ors, joint_motion_action_plans)
            motion_goal_indices = self._advance_motion_goal_indices(motion_goal_indices, plan_costs)
            total_cost += min(plan_costs)
            full_plan.extend(joint_motion_action_plans)
        return full_plan, curr_state, total_cost

    def at_least_one_finished_hl_action(self, joint_hl_action, motion_goal_indices):
        """Returns whether either agent has reached the end of the motion goal list it was supposed
        to perform to finish it's high level action"""
        return any([len(joint_hl_action[i].motion_goals) == motion_goal_indices[i] for i in range(2)])

    def get_low_level_action_plan(self, start_state, h_fn, debug=False):
        """
        Get a plan of joint-actions executable in the environment that will lead to a goal number of deliveries
        by performaing an A* search in high-level action space

        Args:
            state (OvercookedState): starting state

        Returns:
            full_joint_action_plan (list): joint actions to reach goal
            cost (int): a cost in number of timesteps to reach the goal
        """
        full_joint_low_level_action_plan = []
        hl_plan, cost = self.get_hl_plan(start_state, h_fn)
        curr_state = start_state
        prev_h = h_fn(start_state, debug=False)
        total_cost = 0
        for joint_hl_action, curr_goal_state in hl_plan:
            assert all([type(a) is HighLevelAction for a in joint_hl_action])
            hl_action_plan, curr_state, hl_action_cost = self.perform_hl_action(joint_hl_action, curr_state)
            full_joint_low_level_action_plan.extend(hl_action_plan)
            total_cost += hl_action_cost
            assert curr_state == curr_goal_state

            curr_h = h_fn(curr_state, debug=False)
            self.mlp.check_heuristic_consistency(curr_h, prev_h, total_cost)
            prev_h = curr_h
        assert total_cost == cost == len(full_joint_low_level_action_plan), "{} vs {} vs {}".format(
            total_cost, cost, len(full_joint_low_level_action_plan)
        )
        return full_joint_low_level_action_plan, cost

    def get_hl_plan(self, start_state, h_fn, debug=False):
        expand_fn = lambda state: self.get_successor_states(state)
        goal_fn = lambda state: state.num_orders_remaining == 0
        heuristic_fn = lambda state: h_fn(state)

        search_problem = SearchTree(start_state, goal_fn, expand_fn, heuristic_fn, debug=debug)
        hl_plan, cost = search_problem.A_star_graph_search(info=True)
        return hl_plan[1:], cost

    def _advance_motion_goal_indices(self, curr_plan_indices, plan_lengths):
        """Advance indices for agents current motion goals
        based on who finished their motion goal this round"""
        idx0, idx1 = curr_plan_indices
        if plan_lengths[0] == plan_lengths[1]:
            return idx0 + 1, idx1 + 1

        who_finished = np.argmin(plan_lengths)
        if who_finished == 0:
            return idx0 + 1, idx1
        elif who_finished == 1:
            return idx0, idx1 + 1


class Heuristic:
    def __init__(self, mp):
        self.motion_planner = mp
        self.mdp = mp.mdp
        self.heuristic_cost_dict = self._calculate_heuristic_costs()

    def hard_heuristic(self, state, goal_deliveries, time=0, debug=False):
        # NOTE: does not support tomatoes – currently deprecated as harder heuristic
        # does not seem worth the additional computational time

        """
        From a state, we can calculate exactly how many:
        - soup deliveries we need
        - dishes to pots we need
        - onion to pots we need

        We then determine if there are any soups/dishes/onions
        in transit (on counters or on players) than can be
        brought to their destinations faster than starting off from
        a dispenser of the same type. If so, we consider fulfilling
        all demand from these positions.

        After all in-transit objects are considered, we consider the
        costs required to fulfill all the rest of the demand, that is
        given by:
        - pot-delivery trips
        - dish-pot trips
        - onion-pot trips

        The total cost is obtained by determining an optimistic time
        cost for each of these trip types
        """
        forward_cost = 0

        # Obtaining useful quantities
        objects_dict = state.unowned_objects_by_type
        player_objects = state.player_objects_by_type
        pot_states_dict = self.mdp.get_pot_states(state)
        min_pot_delivery_cost = self.heuristic_cost_dict["pot-delivery"]
        min_dish_to_pot_cost = self.heuristic_cost_dict["dish-pot"]
        min_onion_to_pot_cost = self.heuristic_cost_dict["onion-pot"]

        pot_locations = self.mdp.get_pot_locations()
        full_soups_in_pots = (
            pot_states_dict["onion"]["cooking"]
            + pot_states_dict["tomato"]["cooking"]
            + pot_states_dict["onion"]["ready"]
            + pot_states_dict["tomato"]["ready"]
        )
        partially_full_soups = pot_states_dict["onion"]["partially_full"] + pot_states_dict["tomato"]["partially_full"]
        num_onions_in_partially_full_pots = sum([state.get_object(loc).state[1] for loc in partially_full_soups])

        # Calculating costs
        num_deliveries_to_go = goal_deliveries - state.num_delivered

        # SOUP COSTS
        total_num_soups_needed = max([0, num_deliveries_to_go])

        soups_on_counters = [soup_obj for soup_obj in objects_dict["soup"] if soup_obj.position not in pot_locations]
        soups_in_transit = player_objects["soup"] + soups_on_counters
        soup_delivery_locations = self.mdp.get_serving_locations()

        (
            num_soups_better_than_pot,
            total_better_than_pot_soup_cost,
        ) = self.get_costs_better_than_dispenser(
            soups_in_transit,
            soup_delivery_locations,
            min_pot_delivery_cost,
            total_num_soups_needed,
            state,
        )

        min_pot_to_delivery_trips = max([0, total_num_soups_needed - num_soups_better_than_pot])
        pot_to_delivery_costs = min_pot_delivery_cost * min_pot_to_delivery_trips

        forward_cost += total_better_than_pot_soup_cost
        forward_cost += pot_to_delivery_costs

        # DISH COSTS
        total_num_dishes_needed = max([0, min_pot_to_delivery_trips])
        dishes_on_counters = objects_dict["dish"]
        dishes_in_transit = player_objects["dish"] + dishes_on_counters

        (
            num_dishes_better_than_disp,
            total_better_than_disp_dish_cost,
        ) = self.get_costs_better_than_dispenser(
            dishes_in_transit,
            pot_locations,
            min_dish_to_pot_cost,
            total_num_dishes_needed,
            state,
        )

        min_dish_to_pot_trips = max([0, min_pot_to_delivery_trips - num_dishes_better_than_disp])
        dish_to_pot_costs = min_dish_to_pot_cost * min_dish_to_pot_trips

        forward_cost += total_better_than_disp_dish_cost
        forward_cost += dish_to_pot_costs

        # ONION COSTS
        num_pots_to_be_filled = min_pot_to_delivery_trips - len(full_soups_in_pots)
        total_num_onions_needed = num_pots_to_be_filled * 3 - num_onions_in_partially_full_pots
        onions_on_counters = objects_dict["onion"]
        onions_in_transit = player_objects["onion"] + onions_on_counters

        (
            num_onions_better_than_disp,
            total_better_than_disp_onion_cost,
        ) = self.get_costs_better_than_dispenser(
            onions_in_transit,
            pot_locations,
            min_onion_to_pot_cost,
            total_num_onions_needed,
            state,
        )

        min_onion_to_pot_trips = max([0, total_num_onions_needed - num_onions_better_than_disp])
        onion_to_pot_costs = min_onion_to_pot_cost * min_onion_to_pot_trips

        forward_cost += total_better_than_disp_onion_cost
        forward_cost += onion_to_pot_costs

        # Going to closest feature costs
        # NOTE: as implemented makes heuristic inconsistent
        # for player in state.players:
        #     if not player.has_object():
        #         counter_objects = soups_on_counters + dishes_on_counters + onions_on_counters
        #         possible_features = counter_objects + pot_locations + self.mdp.get_dish_dispenser_locations() + self.mdp.get_onion_dispenser_locations()
        #         forward_cost += self.action_manager.min_cost_to_feature(player.pos_and_or, possible_features)

        heuristic_cost = forward_cost / 2

        if debug:
            env = OvercookedEnv(self.mdp)
            env.state = state
            print("\n" + "#" * 35)
            print(f"Current state: (ml timestep {time})\n")

            print(
                "# in transit: \t\t Soups {} \t Dishes {} \t Onions {}".format(
                    len(soups_in_transit),
                    len(dishes_in_transit),
                    len(onions_in_transit),
                )
            )

            # NOTE Possible improvement: consider cost of dish delivery too when considering if a
            # transit soup is better than dispenser equivalent
            print(
                "# better than disp: \t Soups {} \t Dishes {} \t Onions {}".format(
                    num_soups_better_than_pot,
                    num_dishes_better_than_disp,
                    num_onions_better_than_disp,
                )
            )

            print(
                "# of trips: \t\t pot-del {} \t dish-pot {} \t onion-pot {}".format(
                    min_pot_to_delivery_trips,
                    min_dish_to_pot_trips,
                    min_onion_to_pot_trips,
                )
            )

            print(
                "Trip costs: \t\t pot-del {} \t dish-pot {} \t onion-pot {}".format(
                    pot_to_delivery_costs, dish_to_pot_costs, onion_to_pot_costs
                )
            )

            print(str(env) + f"HEURISTIC: {heuristic_cost}")

        return heuristic_cost

    def get_costs_better_than_dispenser(self, possible_objects, target_locations, baseline_cost, num_needed, state):
        """
        Computes the number of objects whose minimum cost to any of the target locations is smaller than
        the baseline cost (clipping it if greater than the number needed). It also calculates a lower
        bound on the cost of using such objects.
        """
        costs_from_transit_locations = []
        for obj in possible_objects:
            obj_pos = obj.position
            if obj_pos in state.player_positions:
                # If object is being carried by a player
                player = [p for p in state.players if p.position == obj_pos][0]
                # NOTE: not sure if this -1 is justified.
                # Made things work better in practice for greedy heuristic based agents.
                # For now this function is just used from there. Consider removing later if
                # greedy heuristic agents end up not being used.
                min_cost = self.motion_planner.min_cost_to_feature(player.pos_and_or, target_locations) - 1
            else:
                # If object is on a counter
                min_cost = self.motion_planner.min_cost_between_features([obj_pos], target_locations)
            costs_from_transit_locations.append(min_cost)

        costs_better_than_dispenser = [cost for cost in costs_from_transit_locations if cost <= baseline_cost]
        better_than_dispenser_total_cost = sum(np.sort(costs_better_than_dispenser)[:num_needed])
        return len(costs_better_than_dispenser), better_than_dispenser_total_cost

    def _calculate_heuristic_costs(self, debug=False):
        """Pre-computes the costs between common trip types for this mdp"""
        pot_locations = self.mdp.get_pot_locations()
        delivery_locations = self.mdp.get_serving_locations()
        dish_locations = self.mdp.get_dish_dispenser_locations()
        onion_locations = self.mdp.get_onion_dispenser_locations()
        tomato_locations = self.mdp.get_tomato_dispenser_locations()

        heuristic_cost_dict = {
            "pot-delivery": self.motion_planner.min_cost_between_features(
                pot_locations, delivery_locations, manhattan_if_fail=True
            ),
            "dish-pot": self.motion_planner.min_cost_between_features(
                dish_locations, pot_locations, manhattan_if_fail=True
            ),
        }

        onion_pot_cost = self.motion_planner.min_cost_between_features(
            onion_locations, pot_locations, manhattan_if_fail=True
        )
        tomato_pot_cost = self.motion_planner.min_cost_between_features(
            tomato_locations, pot_locations, manhattan_if_fail=True
        )

        if debug:
            print("Heuristic cost dict", heuristic_cost_dict)
        assert onion_pot_cost != np.inf or tomato_pot_cost != np.inf
        if onion_pot_cost != np.inf:
            heuristic_cost_dict["onion-pot"] = onion_pot_cost
        if tomato_pot_cost != np.inf:
            heuristic_cost_dict["tomato-pot"] = tomato_pot_cost

        return heuristic_cost_dict

    def simple_heuristic(self, state, time=0, debug=False):
        """Simpler heuristic that tends to run faster than current one"""
        # NOTE: State should be modified to have an order list w.r.t. which
        # one can calculate the heuristic
        assert state.order_list is not None

        objects_dict = state.unowned_objects_by_type
        player_objects = state.player_objects_by_type
        pot_states_dict = self.mdp.get_pot_states(state)
        num_deliveries_to_go = state.num_orders_remaining

        full_soups_in_pots = (
            pot_states_dict["onion"]["cooking"]
            + pot_states_dict["tomato"]["cooking"]
            + pot_states_dict["onion"]["ready"]
            + pot_states_dict["tomato"]["ready"]
        )
        partially_full_onion_soups = pot_states_dict["onion"]["partially_full"]
        partially_full_tomato_soups = pot_states_dict["tomato"]["partially_full"]
        num_onions_in_partially_full_pots = sum([state.get_object(loc).state[1] for loc in partially_full_onion_soups])
        num_tomatoes_in_partially_full_pots = sum(
            [state.get_object(loc).state[1] for loc in partially_full_tomato_soups]
        )

        soups_in_transit = player_objects["soup"]
        dishes_in_transit = objects_dict["dish"] + player_objects["dish"]
        onions_in_transit = objects_dict["onion"] + player_objects["onion"]
        tomatoes_in_transit = objects_dict["tomato"] + player_objects["tomato"]

        num_pot_to_delivery = max([0, num_deliveries_to_go - len(soups_in_transit)])
        num_dish_to_pot = max([0, num_pot_to_delivery - len(dishes_in_transit)])

        num_pots_to_be_filled = num_pot_to_delivery - len(full_soups_in_pots)
        num_onions_needed_for_pots = (
            num_pots_to_be_filled * 3 - len(onions_in_transit) - num_onions_in_partially_full_pots
        )
        num_tomatoes_needed_for_pots = (
            num_pots_to_be_filled * 3 - len(tomatoes_in_transit) - num_tomatoes_in_partially_full_pots
        )
        num_onion_to_pot = max([0, num_onions_needed_for_pots])
        num_tomato_to_pot = max([0, num_tomatoes_needed_for_pots])

        pot_to_delivery_costs = self.heuristic_cost_dict["pot-delivery"] * num_pot_to_delivery
        dish_to_pot_costs = self.heuristic_cost_dict["dish-pot"] * num_dish_to_pot

        items_to_pot_costs = []
        if "onion-pot" in self.heuristic_cost_dict.keys():
            onion_to_pot_costs = self.heuristic_cost_dict["onion-pot"] * num_onion_to_pot
            items_to_pot_costs.append(onion_to_pot_costs)
        if "tomato-pot" in self.heuristic_cost_dict.keys():
            tomato_to_pot_costs = self.heuristic_cost_dict["tomato-pot"] * num_tomato_to_pot
            items_to_pot_costs.append(tomato_to_pot_costs)

        # NOTE: doesn't take into account that a combination of the two might actually be more advantageous.
        # Might cause heuristic to be inadmissable in some edge cases.
        items_to_pot_cost = min(items_to_pot_costs)

        heuristic_cost = (pot_to_delivery_costs + dish_to_pot_costs + items_to_pot_cost) / 2

        if debug:
            env = OvercookedEnv(self.mdp)
            env.state = state
            print("\n" + "#" * 35)
            print(f"Current state: (ml timestep {time})\n")

            print(
                "# in transit: \t\t Soups {} \t Dishes {} \t Onions {}".format(
                    len(soups_in_transit),
                    len(dishes_in_transit),
                    len(onions_in_transit),
                )
            )

            print(
                "Trip costs: \t\t pot-del {} \t dish-pot {} \t onion-pot {}".format(
                    pot_to_delivery_costs, dish_to_pot_costs, onion_to_pot_costs
                )
            )

            print(str(env) + f"HEURISTIC: {heuristic_cost}")

        return heuristic_cost
