"""Read real v3 episodes and decode their first/last frames without loading a model."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.types import ModalityConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-path', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=3, help='Number of episodes to decode; 0 means all')
    args = parser.parse_args()
    if args.episodes < 0:
        parser.error('--episodes must be nonnegative')
    metadata = json.loads((args.dataset_path / 'meta/modality.json').read_text())
    configs = {key: ModalityConfig(delta_indices=[0], modality_keys=list(metadata[key]))
               for key in ['state', 'action', 'video']}
    configs['language'] = ModalityConfig(delta_indices=[0], modality_keys=['annotation.human.task_description'])
    loader = LeRobotEpisodeLoader(args.dataset_path, configs)
    count = min(args.episodes or len(loader), len(loader))
    for index in range(count):
        frame = loader.load_episode(index, select_frames=lambda length: np.unique([0, length-1]))
        print(f'Episode {loader.episodes_metadata[index]["episode_index"]}: {len(frame)} rows', flush=True)
        for key in configs['video'].modality_keys:
            first, last = frame[f'video.{key}'].iloc[0], frame[f'video.{key}'].iloc[-1]
            if first is None or last is None or not first.size or not last.size:
                raise ValueError(f'Empty decoded video frame: {key}')
            print(f'  {key}: first={first.shape}, last={last.shape}', flush=True)
    print(f'PASS: loaded {count} episodes and decoded boundary frames for every configured video.')
    print('This checks loading only; action semantics and task descriptions still require review before training.')


if __name__ == '__main__':
    main()
