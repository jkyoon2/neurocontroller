"""
Layout Parser: Parse Overcooked layout files to extract static information.

This module reads .layout files from overcooked_new and extracts:
- Grid structure
- Start positions
- Orders
- Game parameters
"""

import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


class LayoutParser:
    """Parser for Overcooked .layout files."""
    
    def __init__(self, layouts_dir: Optional[str] = None):
        """Initialize LayoutParser.
        
        Args:
            layouts_dir: Directory containing .layout files.
                        If None, uses default overcooked_new layouts directory.
        """
        if layouts_dir is None:
            # Default to overcooked_new layouts directory
            self.layouts_dir = Path(
                "zsceval/envs/overcooked_new/src/overcooked_ai_py/data/layouts"
            )
        else:
            self.layouts_dir = Path(layouts_dir)
        
        logger.debug(f"LayoutParser initialized with directory: {self.layouts_dir}")
    
    def parse_layout(self, layout_name: str) -> Dict[str, Any]:
        """Parse a layout file and return static info dictionary.
        
        Args:
            layout_name: Name of the layout (without .layout extension)
            
        Returns:
            Dictionary with static game information
        """
        layout_path = self.layouts_dir / f"{layout_name}.layout"
        
        if not layout_path.exists():
            logger.error(f"Layout file not found: {layout_path}")
            return self._get_default_static_info(layout_name)
        
        try:
            with open(layout_path, 'r') as f:
                file_content = f.read()
            
            # .layout 파일은 Python dict 형식이므로 ast.literal_eval 사용
            try:
                layout_data = ast.literal_eval(file_content)
            except (SyntaxError, ValueError):
                # Fallback: JSON으로 시도
                layout_data = json.loads(file_content)
            
            # Parse grid
            grid_str = layout_data.get("grid", "")
            grid, width, height, start_positions = self._parse_grid(grid_str)
            
            # Parse orders
            all_orders = layout_data.get("start_all_orders", [])
            
            # Build static info
            static_info = {
                "layoutName": layout_name,
                "width": width,
                "height": height,
                "grid": grid,
                "startPlayerPositions": start_positions,
                "allOrders": all_orders,
                "cookTime": 20,  # Default values - can be customized
                "soupIngredientCapacity": 3,
                "deliveryReward": 20
            }
            
            logger.debug(f"Successfully parsed layout: {layout_name} ({width}x{height})")
            return static_info
            
        except (SyntaxError, ValueError, json.JSONDecodeError) as e:
            logger.error(f"Failed to decode layout {layout_path}: {e}")
            return self._get_default_static_info(layout_name)
        except Exception as e:
            logger.error(f"Failed to parse layout {layout_path}: {e}")
            return self._get_default_static_info(layout_name)
    
    def _parse_grid(self, grid_str: str) -> Tuple[List[List[str]], int, int, List[Dict]]:
        """Parse grid string to extract grid structure and start positions.
        
        Args:
            grid_str: Multi-line grid string from layout file
            
        Returns:
            Tuple of (grid, width, height, start_positions)
        """
        # Split by lines and strip whitespace
        lines = [line.strip() for line in grid_str.strip().split('\n')]
        
        # Build grid as 2D list
        grid = []
        start_positions = []
        player_id = 0
        
        for y, line in enumerate(lines):
            row = []
            for x, char in enumerate(line):
                # Map characters to grid symbols
                if char == 'X':
                    row.append('X')  # Wall
                elif char == 'P':
                    row.append('P')  # Pot
                elif char == 'O':
                    row.append('O')  # Onion dispenser
                elif char == 'T':
                    row.append('T')  # Tomato dispenser
                elif char == 'D':
                    row.append('D')  # Dish dispenser
                elif char == 'S':
                    row.append('S')  # Serving location
                elif char == ' ':
                    row.append(' ')  # Empty space
                elif char.isdigit():
                    # Player start position
                    row.append(' ')  # Mark as empty space in grid
                    start_positions.append({
                        "id": player_id,
                        "position": {"x": x, "y": y}
                    })
                    player_id += 1
                else:
                    # Unknown character, treat as empty
                    row.append(' ')
            
            grid.append(row)
        
        height = len(grid)
        width = len(grid[0]) if grid else 0
        
        # If no explicit player positions, use default
        if not start_positions:
            # Default positions (center-ish)
            start_positions = [
                {"id": 0, "position": {"x": width // 2, "y": height // 2}},
                {"id": 1, "position": {"x": width // 2 + 1, "y": height // 2}}
            ]
        
        return grid, width, height, start_positions
    
    def _get_default_static_info(self, layout_name: str) -> Dict[str, Any]:
        """Return default static info if layout parsing fails.
        
        Args:
            layout_name: Name of the layout
            
        Returns:
            Default static info dictionary (cramped_room 기준)
        """
        logger.warning(f"Using default static info for layout: {layout_name}")
        
        return {
            "layoutName": layout_name,
            "width": 5,
            "height": 4,
            "grid": [
                ["X", "X", "P", "X", "X"],
                ["O", " ", " ", "2", "O"],
                ["X", "1", " ", " ", "X"],
                ["X", "D", "X", "S", "X"]
            ],
            "startPlayerPositions": [
                {"id": 0, "position": {"x": 1, "y": 2}},
                {"id": 1, "position": {"x": 3, "y": 1}}
            ],
            "allOrders": [
                {"ingredients": ["onion", "onion", "onion"]}
            ],
            "cookTime": 20,
            "soupIngredientCapacity": 3,
            "deliveryReward": 20
        }
    
    def list_available_layouts(self) -> List[str]:
        """List all available layout files.
        
        Returns:
            List of layout names (without .layout extension)
        """
        if not self.layouts_dir.exists():
            logger.warning(f"Layouts directory not found: {self.layouts_dir}")
            return []
        
        layout_files = sorted(self.layouts_dir.glob("*.layout"))
        return [f.stem for f in layout_files]

