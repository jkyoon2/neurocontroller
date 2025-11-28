"""
Observation Decoder: Decode Overcooked lossless_state_encoding to game state.

This module decodes the observation tensor (H, W, C) back to:
- Player positions and orientations
- Held objects
- Pot states (ingredients, cooking status)
- Dish and ingredient locations
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


# Overcooked constants
DIRECTION_TO_NAME = {
    0: "north",
    1: "south",
    2: "east",
    3: "west"
}

# Action order from Action.ALL_ACTIONS = Direction.INDEX_TO_DIRECTION + [STAY, INTERACT]
# = [NORTH, SOUTH, EAST, WEST, STAY, INTERACT]
ACTION_NAMES = ["north", "south", "east", "west", "stay", "interact"]


class ObservationDecoder:
    """Decode Overcooked observation to game state."""
    
    def __init__(self, num_agents: int = 2):
        """Initialize decoder.
        
        Args:
            num_agents: Number of agents (2 for standard Overcooked)
        """
        self.num_agents = num_agents
        
        # Layer indices for new dynamics (25 channels for 2 agents)
        # ego_player: 0-9 (location + 4 orientations + held object type × 4)
        # partner_player: 10-19
        # base_map: 20-25 (pot, counter, onion_disp, tomato_disp, dish_disp, serve)
        # variable_map: depends on num layers
        
        # Exact layer mapping from lossless_state_encoding
        # ordered_player_features (10): ego_loc(0), partner_loc(1), 
        #   ego_orientation(2-5), partner_orientation(6-9)
        # base_map_features (6): pot_loc(10), counter_loc(11), onion_disp(12), 
        #   tomato_disp(13), dish_disp(14), serve(15)
        # variable_map_features (8): onions_in_pot(16), tomatoes_in_pot(17), 
        #   onions_in_soup(18), tomatoes_in_soup(19), soup_cook_time_remaining(20),
        #   dishes(21), onions(22), tomatoes(23)
        # urgency_features (1): urgency(24)
        # Total: 25 channels
        self.layer_idx = {
            'ego_loc': 0,
            'partner_loc': 1,
            'ego_orientation_0': 2,  # north
            'ego_orientation_1': 3,  # south
            'ego_orientation_2': 4,  # east
            'ego_orientation_3': 5,  # west
            'partner_orientation_0': 6,  # north
            'partner_orientation_1': 7,  # south
            'partner_orientation_2': 8,  # east
            'partner_orientation_3': 9,  # west
            'pot_loc': 10,
            'counter_loc': 11,
            'onion_disp_loc': 12,
            'tomato_disp_loc': 13,
            'dish_disp_loc': 14,
            'serve_loc': 15,
            'onions_in_pot': 16,
            'tomatoes_in_pot': 17,
            'onions_in_soup': 18,
            'tomatoes_in_soup': 19,
            'soup_cook_time_remaining': 20,
            'dishes': 21,
            'onions': 22,
            'tomatoes': 23,
            'urgency': 24,
        }
    
    def decode(
        self,
        obs: np.ndarray,
        timestep: int,
        actions: Optional[np.ndarray] = None,
        rewards: Optional[np.ndarray] = None,
        previous_actions: Optional[np.ndarray] = None
    ) -> Dict:
        """Decode observation to game state.
        
        Args:
            obs: Observation array, shape (H, W, C)
            timestep: Current timestep
            actions: Current actions
            rewards: Current rewards
            previous_actions: Previous actions
            
        Returns:
            Game state dictionary
        """
        H, W, C = obs.shape
        
        # Decode players
        players = self._decode_players(obs, actions, previous_actions)
        
        # Decode objects (pots, dishes, onions)
        objects = self._decode_objects(obs)
        
        # Calculate score (ensure JSON serializable)
        if rewards is not None:
            if isinstance(rewards, np.ndarray):
                score = float(rewards.sum())
            else:
                score = float(rewards)
        else:
            score = 0.0
        
        state = {
            "timestep": int(timestep),  # Ensure int, not np.int64
            "score": score,
            "orders": [
                {"ingredients": ["onion", "onion", "onion"]}  # Default order
            ],
            "players": players,
            "objects": objects
        }
        
        return state
    
    def _decode_players(
        self,
        obs: np.ndarray,
        actions: Optional[np.ndarray],
        previous_actions: Optional[np.ndarray]
    ) -> List[Dict]:
        """Decode player information from observation.
        
        Args:
            obs: Observation array, shape (H, W, C)
            actions: Current actions
            previous_actions: Previous actions
            
        Returns:
            List of player dictionaries
        """
        players = []
        H, W, C = obs.shape
        
        # Decode ego player (agent 0 from its perspective)
        ego_pos, ego_orient = self._find_player_from_layers(
            obs, loc_channel=self.layer_idx['ego_loc'], 
            orient_start_channel=self.layer_idx['ego_orientation_0']
        )
        
        # Decode partner player
        partner_pos, partner_orient = self._find_player_from_layers(
            obs, loc_channel=self.layer_idx['partner_loc'],
            orient_start_channel=self.layer_idx['partner_orientation_0']
        )
        
        # Decode held objects (simplified - check specific layers)
        ego_held = self._decode_held_object(obs, player_pos=ego_pos, player_idx=0)
        partner_held = self._decode_held_object(obs, player_pos=partner_pos, player_idx=1)
        
        # Convert actions
        current_action = self._action_to_name(actions)
        prev_action = self._action_to_name(previous_actions)
        
        players.append({
            "id": 0,
            "position": {"x": int(ego_pos[0]) if ego_pos else 1, "y": int(ego_pos[1]) if ego_pos else 2},
            "orientation": str(ego_orient) if ego_orient else "north",
            "heldObject": ego_held,
            "action": {
                "current": str(current_action),
                "previous": str(prev_action)
            }
        })
        
        players.append({
            "id": 1,
            "position": {"x": int(partner_pos[0]) if partner_pos else 5, "y": int(partner_pos[1]) if partner_pos else 2},
            "orientation": str(partner_orient) if partner_orient else "north",
            "heldObject": partner_held,
            "action": {
                "current": str(current_action),
                "previous": str(prev_action)
            }
        })
        
        return players
    
    def _find_player_from_layers(
        self,
        obs: np.ndarray,
        loc_channel: int,
        orient_start_channel: int
    ) -> Tuple[Optional[Tuple[int, int]], Optional[str]]:
        """Find player position and orientation from observation layers.
        
        Args:
            obs: Observation array
            loc_channel: Channel index for player location
            orient_start_channel: Starting channel index for orientations (4 consecutive)
            
        Returns:
            (position, orientation) tuple
        """
        H, W, C = obs.shape
        
        # Find player location
        # Note: observation values are 255 (present) or 0 (absent) due to uint8 encoding
        if loc_channel < C:
            player_loc_layer = obs[:, :, loc_channel]
            positions = np.argwhere(player_loc_layer > 0)
            
            if len(positions) > 0:
                y, x = positions[0]  # Note: numpy indexing is (row, col) = (y, x)
                position = (x, y)
            else:
                position = None
        else:
            position = None
        
        # Find orientation (4 consecutive layers)
        orientation = None
        for i in range(4):
            channel = orient_start_channel + i
            if channel < C:
                orient_layer = obs[:, :, channel]
                if orient_layer.max() > 0:
                    orientation = DIRECTION_TO_NAME.get(i, "north")
                    break
        
        return position, orientation
    
    def _decode_held_object(
        self,
        obs: np.ndarray,
        player_pos: Optional[Tuple[int, int]],
        player_idx: int
    ) -> Optional[Dict]:
        """Decode held object for a player.
        
        Player held objects are encoded in observation at the player's position.
        Check onions, tomatoes, dishes, soup channels at player position.
        
        Args:
            obs: Observation array, shape (H, W, C)
            player_pos: Player position (x, y)
            player_idx: Player index (0 or 1)
            
        Returns:
            Held object dict {"name": str, "position": {x, y}} or None
        """
        if player_pos is None:
            return None
        
        H, W, C = obs.shape
        x, y = player_pos
        
        # Check if position is valid
        if not (0 <= y < H and 0 <= x < W):
            return None
        
        # Check each object type channel at player position
        # Order matters: check soup first (most specific), then others
        
        # 1. Check for soup (cooked or ready)
        # Soup can be held at player position
        onions_in_soup_channel = self.layer_idx.get('onions_in_soup')
        tomatoes_in_soup_channel = self.layer_idx.get('tomatoes_in_soup')
        
        if onions_in_soup_channel and onions_in_soup_channel < C:
            soup_value = obs[y, x, onions_in_soup_channel]
            if soup_value > 0:
                # Player is holding soup
                return {
                    "name": "soup",
                    "position": {"x": int(x), "y": int(y)}
                }
        
        # 2. Check for dish
        dishes_channel = self.layer_idx.get('dishes')
        if dishes_channel and dishes_channel < C:
            dish_value = obs[y, x, dishes_channel]
            if dish_value > 0:
                return {
                    "name": "dish",
                    "position": {"x": int(x), "y": int(y)}
                }
        
        # 3. Check for onion
        onions_channel = self.layer_idx.get('onions')
        if onions_channel and onions_channel < C:
            onion_value = obs[y, x, onions_channel]
            if onion_value > 0:
                return {
                    "name": "onion",
                    "position": {"x": int(x), "y": int(y)}
                }
        
        # 4. Check for tomato
        tomatoes_channel = self.layer_idx.get('tomatoes')
        if tomatoes_channel and tomatoes_channel < C:
            tomato_value = obs[y, x, tomatoes_channel]
            if tomato_value > 0:
                return {
                    "name": "tomato",
                    "position": {"x": int(x), "y": int(y)}
                }
        
        # No object held
        return None
    
    def _decode_objects(self, obs: np.ndarray) -> List[Dict]:
        """Decode objects (pots, dishes, onions, tomatoes) from observation.
        
        Args:
            obs: Observation array
            
        Returns:
            List of object dictionaries
        """
        objects = []
        H, W, C = obs.shape
        
        # Decode pots with ingredients
        pot_objects = self._decode_pots(obs)
        objects.extend(pot_objects)
        
        # Decode dishes on counters
        dish_objects = self._decode_dishes(obs)
        objects.extend(dish_objects)
        
        # Decode onions on counters
        onion_objects = self._decode_onions(obs)
        objects.extend(onion_objects)
        
        # Decode tomatoes on counters
        tomato_objects = self._decode_tomatoes(obs)
        objects.extend(tomato_objects)
        
        return objects
    
    def _decode_pots(self, obs: np.ndarray) -> List[Dict]:
        """Decode pot states from observation.
        
        Args:
            obs: Observation array
            
        Returns:
            List of pot/soup objects
        """
        pots = []
        H, W, C = obs.shape
        
        # Find pot locations (channel 10)
        pot_loc_channel = self.layer_idx['pot_loc']
        onions_in_pot_channel = self.layer_idx['onions_in_pot']
        tomatoes_in_pot_channel = self.layer_idx['tomatoes_in_pot']
        onions_in_soup_channel = self.layer_idx['onions_in_soup']
        tomatoes_in_soup_channel = self.layer_idx['tomatoes_in_soup']
        cook_time_channel = self.layer_idx['soup_cook_time_remaining']
        
        if pot_loc_channel < C:
            pot_layer = obs[:, :, pot_loc_channel]
            pot_positions = np.argwhere(pot_layer > 0)
            
            # For each pot, check if it has ingredients
            for pos in pot_positions:
                y, x = pos
                
                # Get ingredient counts
                # Note: values are stored as 0-255 range, need to normalize back to 0-3
                # Original encoding uses 0, 1, 2, 3 but stored as uint8 (0-255)
                # So we normalize: 255 → 1, 510 → 2, 765 → 3
                # But since max value is 255, we interpret raw values directly if ≤ 3,
                # or normalize by 255 if > 3
                
                def decode_count(value):
                    """Decode ingredient count from observation value.
                    
                    The observation is multiplied by 255 in Overcooked_Env.py line 883:
                    self.featurize_fn = lambda state: [
                        self.featurize_fn_mapping[f](state)[i] * (255 if f == "ppo" else 1)
                        ...
                    ]
                    
                    So: 0 → 0, 1 → 255, 2 → 510, 3 → 765
                    """
                    if value == 0:
                        return 0
                    else:
                        # Divide by 255 to get back original count
                        return min(3, max(0, int(round(value / 255.0))))
                
                num_onions_idle = decode_count(obs[y, x, onions_in_pot_channel]) if onions_in_pot_channel < C else 0
                num_tomatoes_idle = decode_count(obs[y, x, tomatoes_in_pot_channel]) if tomatoes_in_pot_channel < C else 0
                num_onions_cooking = decode_count(obs[y, x, onions_in_soup_channel]) if onions_in_soup_channel < C else 0
                num_tomatoes_cooking = decode_count(obs[y, x, tomatoes_in_soup_channel]) if tomatoes_in_soup_channel < C else 0
                cook_time_remaining = int(obs[y, x, cook_time_channel]) if cook_time_channel < C else 0
                
                # Pot is either idle (can add ingredients) or cooking
                is_cooking = (num_onions_cooking > 0 or num_tomatoes_cooking > 0)
                
                if is_cooking:
                    num_onions = num_onions_cooking
                    num_tomatoes = num_tomatoes_cooking
                else:
                    num_onions = num_onions_idle
                    num_tomatoes = num_tomatoes_idle
                
                if num_onions > 0 or num_tomatoes > 0:
                    ingredients = ["onion"] * num_onions + ["tomato"] * num_tomatoes
                    
                    pot = {
                        "name": "soup",
                        "position": {"x": int(x), "y": int(y)},
                        "ingredients": ingredients,
                        "isCooking": is_cooking,
                        "isReady": is_cooking and cook_time_remaining == 0,
                        "cookTime": 20,
                        "cookingTick": max(0, 20 - cook_time_remaining) if is_cooking else 0
                    }
                    pots.append(pot)
        
        return pots
    
    def _decode_dishes(self, obs: np.ndarray) -> List[Dict]:
        """Decode dish objects from observation.
        
        Args:
            obs: Observation array
            
        Returns:
            List of dish objects
        """
        dishes = []
        H, W, C = obs.shape
        
        # Find dish layer (channel 21)
        dish_channel = self.layer_idx['dishes']
        
        if dish_channel < C:
            dish_layer = obs[:, :, dish_channel]
            dish_positions = np.argwhere(dish_layer > 0)
            
            for pos in dish_positions:
                y, x = pos
                dishes.append({
                    "name": "dish",
                    "position": {"x": int(x), "y": int(y)}
                })
        
        return dishes
    
    def _decode_onions(self, obs: np.ndarray) -> List[Dict]:
        """Decode onion objects from observation.
        
        Args:
            obs: Observation array
            
        Returns:
            List of onion objects
        """
        onions = []
        H, W, C = obs.shape
        
        # Find onion layer (channel 22)
        onion_channel = self.layer_idx['onions']
        
        if onion_channel < C:
            onion_layer = obs[:, :, onion_channel]
            onion_positions = np.argwhere(onion_layer > 0)
            
            for pos in onion_positions:
                y, x = pos
                onions.append({
                    "name": "onion",
                    "position": {"x": int(x), "y": int(y)}
                })
        
        return onions
    
    def _decode_tomatoes(self, obs: np.ndarray) -> List[Dict]:
        """Decode tomato objects from observation.
        
        Args:
            obs: Observation array
            
        Returns:
            List of tomato objects
        """
        tomatoes = []
        H, W, C = obs.shape
        
        # Find tomato layer (channel 23)
        tomato_channel = self.layer_idx['tomatoes']
        
        if tomato_channel < C:
            tomato_layer = obs[:, :, tomato_channel]
            tomato_positions = np.argwhere(tomato_layer > 0)
            
            for pos in tomato_positions:
                y, x = pos
                tomatoes.append({
                    "name": "tomato",
                    "position": {"x": int(x), "y": int(y)}
                })
        
        return tomatoes
    
    def _action_to_name(self, actions: Optional[np.ndarray]) -> str:
        """Convert action index to name.
        
        Args:
            actions: Action array or None
            
        Returns:
            Action name string
        """
        if actions is None:
            return "stay"
        
        if isinstance(actions, np.ndarray):
            action_idx = int(actions.flatten()[0])
        else:
            action_idx = int(actions)
        
        return ACTION_NAMES[action_idx] if action_idx < len(ACTION_NAMES) else "stay"

