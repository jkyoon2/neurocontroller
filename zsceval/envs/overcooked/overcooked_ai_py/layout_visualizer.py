#!/usr/bin/env python3
"""
Overcooked Layout Visualizer
레이아웃 파일들을 시각적으로 보여주는 스크립트
"""

import os
import json
import glob
import argparse
import itertools
from pathlib import Path
from zsceval.envs.overcooked.overcooked_ai_py.utils import load_dict_from_file
from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
from zsceval.envs.overcooked.overcooked_ai_py.visualization.state_visualizer import StateVisualizer

# ANSI 색상 코드
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    GRAY = '\033[90m'
    BOLD = '\033[1m'
    END = '\033[0m'

# 레이아웃 심볼과 색상 매핑
LAYOUT_SYMBOLS = {
    'X': ('█', Colors.GRAY),      # 벽/카운터
    ' ': ('·', Colors.WHITE),     # 빈 공간
    'O': ('O', Colors.YELLOW),    # 양파 디스펜서
    'T': ('T', Colors.RED),       # 토마토 디스펜서
    'P': ('P', Colors.BLUE),      # 냄비
    'D': ('D', Colors.CYAN),      # 접시 디스펜서
    'S': ('S', Colors.GREEN),     # 서빙 위치
    '1': ('1', Colors.PURPLE),    # 플레이어 1
    '2': ('2', Colors.PURPLE),    # 플레이어 2
    '3': ('3', Colors.PURPLE),    # 플레이어 3
    '4': ('4', Colors.PURPLE),    # 플레이어 4
    '5': ('5', Colors.PURPLE),    # 플레이어 5
    '6': ('6', Colors.PURPLE),    # 플레이어 6
    '7': ('7', Colors.PURPLE),    # 플레이어 7
    '8': ('8', Colors.PURPLE),    # 플레이어 8
    '9': ('9', Colors.PURPLE),    # 플레이어 9
}

# 사용할 수 있는 셰프 모자 색상 (스프라이트에 존재하는 색상만)
AVAILABLE_CHEF_COLORS = ["blue", "green", "orange", "purple", "red"]

def load_layout_file(filepath):
    """레이아웃 파일을 로드합니다 (프로젝트와 동일한 방식으로 eval 사용)."""
    try:
        return load_dict_from_file(filepath)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def parse_grid_lines(grid_str):
    grid = grid_str.strip()
    return [line.strip() for line in grid.split('\n') if line.strip()]

def visualize_layout(layout_data, layout_name):
    """레이아웃을 시각화합니다 (터미널 텍스트)."""
    if not layout_data or 'grid' not in layout_data:
        print(f"Invalid layout data for {layout_name}")
        return
    
    lines = parse_grid_lines(layout_data['grid'])
    
    print(f"\n{Colors.BOLD}{Colors.CYAN}=== {layout_name} ==={Colors.END}")
    print(f"{Colors.GRAY}Size: {len(lines[0])}x{len(lines)}{Colors.END}")
    
    # 레이아웃 정보 출력
    if 'cook_time' in layout_data:
        print(f"{Colors.GRAY}Cook time: {layout_data['cook_time']}{Colors.END}")
    if 'num_items_for_soup' in layout_data:
        print(f"{Colors.GRAY}Items per soup: {layout_data['num_items_for_soup']}{Colors.END}")
    if 'delivery_reward' in layout_data:
        print(f"{Colors.GRAY}Delivery reward: {layout_data['delivery_reward']}{Colors.END}")
    
    print(f"\n{Colors.BOLD}Layout:{Colors.END}")
    
    # 그리드 시각화
    for i, line in enumerate(lines):
        print(f"{Colors.GRAY}{i:2d}{Colors.END} ", end="")
        for char in line:
            if char in LAYOUT_SYMBOLS:
                symbol, color = LAYOUT_SYMBOLS[char]
                print(f"{color}{symbol}{Colors.END}", end="")
            else:
                print(f"{Colors.RED}?{Colors.END}", end="")
        print()
    
    # 범례 출력
    joined_grid = "\n".join(lines)
    print(f"\n{Colors.BOLD}Legend:{Colors.END}")
    for symbol, (display, color) in LAYOUT_SYMBOLS.items():
        if symbol in joined_grid:
            print(f"  {color}{display}{Colors.END} = {symbol}")

def list_all_layouts(layouts_dir):
    """모든 레이아웃 파일을 찾습니다."""
    layout_files = glob.glob(os.path.join(layouts_dir, "*.layout"))
    return sorted(layout_files)

def render_layout_to_image(layout_name, layout_data, output_path, show_players=False, tile_size=75):
    """아이콘 기반 스프라이트로 레이아웃 이미지를 저장합니다."""
    lines = parse_grid_lines(layout_data['grid'])
    # OvercookedGridworld가 플레이어 숫자를 제거하고 terrain/시작 위치를 만들어줌
    mdp = OvercookedGridworld.from_grid(lines, base_layout_params={"layout_name": layout_name, **{k:v for k,v in layout_data.items() if k != 'grid'}})
    if show_players:
        state = mdp.get_standard_start_state()
    else:
        state = OvercookedState(players=[], objects={}, order_list=mdp.start_order_list)
    # 플레이어 수만큼 색상 리스트를 생성 (부족하면 순환 재사용)
    num_players = len(state.players)
    player_colors = [AVAILABLE_CHEF_COLORS[i % len(AVAILABLE_CHEF_COLORS)] for i in range(num_players)]

    visualizer = StateVisualizer(
        tile_size=tile_size,
        is_rendering_hud=False,
        is_rendering_cooking_timer=False,
        is_rendering_action_probs=False,
        player_colors=player_colors,
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    saved_path = visualizer.display_rendered_state(
        state=state,
        grid=mdp.terrain_mtx,
        img_path=output_path,
        ipython_display=False,
        window_display=False,
    )
    print(f"{Colors.GREEN}Saved image:{Colors.END} {saved_path}")

def main():
    """메인 함수"""
    # Argument parser 설정
    parser = argparse.ArgumentParser(description='Overcooked Layout Visualizer')
    parser.add_argument('--layout', '-l', type=str, help='특정 레이아웃 파일명 (예: simple, random0_hard)')
    parser.add_argument('--list', action='store_true', help='사용 가능한 모든 레이아웃 목록 표시')
    parser.add_argument('--save', action='store_true', help='이미지를 기본 경로로 저장 (visualizations/<name>.png)')
    parser.add_argument('--output', '-o', type=str, help='이미지 저장 경로 (예: /path/to/out.png)')
    parser.add_argument('--tile-size', type=int, default=75, help='타일 크기 (기본 75)')
    parser.add_argument('--hide-players', action='store_true', help='이미지에서 플레이어 표시 숨기기')
    args = parser.parse_args()
    
    # 레이아웃 디렉토리 경로
    layouts_dir = "zsceval/envs/overcooked/overcooked_ai_py/data/layouts"
    default_out_dir = "zsceval/envs/overcooked/overcooked_ai_py/data/layouts/visualization"
    
    if not os.path.exists(layouts_dir):
        print(f"Layouts directory not found: {layouts_dir}")
        return
    
    layout_files = list_all_layouts(layouts_dir)
    
    if not layout_files:
        print("No layout files found!")
        return
    
    # 사용 가능한 레이아웃 목록 표시
    if args.list:
        print(f"{Colors.BOLD}{Colors.GREEN}Available layouts:{Colors.END}")
        for i, filepath in enumerate(layout_files, 1):
            filename = os.path.basename(filepath).replace('.layout', '')
            print(f"  {i:2d}. {filename}")
        return
    
    # 특정 레이아웃 시각화
    if args.layout:
        layout_name = args.layout
        layout_file = os.path.join(layouts_dir, f"{layout_name}.layout")
        
        if not os.path.exists(layout_file):
            print(f"{Colors.RED}Layout file not found: {layout_file}{Colors.END}")
            print(f"{Colors.YELLOW}Available layouts:{Colors.END}")
            for filepath in layout_files:
                filename = os.path.basename(filepath).replace('.layout', '')
                print(f"  - {filename}")
            return
        
        layout_data = load_layout_file(layout_file)
        visualize_layout(layout_data, layout_name)
        
        # 이미지 저장 로직
        out_path = args.output
        if args.save and not out_path:
            os.makedirs(default_out_dir, exist_ok=True)
            out_path = os.path.join(default_out_dir, f"{layout_name}.png")
        if out_path:
            render_layout_to_image(
                layout_name,
                layout_data,
                out_path,
                show_players=(not args.hide_players),
                tile_size=args.tile_size,
            )
        return
    
    # 기본: 모든 레이아웃 시각화 (텍스트) 및 필요 시 이미지 저장
    print(f"{Colors.BOLD}{Colors.GREEN}Found {len(layout_files)} layout files:{Colors.END}")
    for i, filepath in enumerate(layout_files, 1):
        filename = os.path.basename(filepath)
        print(f"  {i:2d}. {filename}")
    
    print(f"\n{Colors.BOLD}{Colors.YELLOW}Visualizing all layouts:{Colors.END}")
    
    for filepath in layout_files:
        layout_name = os.path.basename(filepath).replace('.layout', '')
        layout_data = load_layout_file(filepath)
        visualize_layout(layout_data, layout_name)
        if args.save or args.output:
            out_path = args.output or os.path.join(default_out_dir, f"{layout_name}.png")
            render_layout_to_image(
                layout_name,
                layout_data,
                out_path,
                show_players=(not args.hide_players),
                tile_size=args.tile_size,
            )
    
    print(f"\n{Colors.BOLD}{Colors.GREEN}Visualization complete!{Colors.END}")

if __name__ == "__main__":
    main()
