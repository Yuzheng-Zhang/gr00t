import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.types import ModalityConfig


class TestV3Loader(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'meta/episodes/chunk-000').mkdir(parents=True)
        (self.root / 'data/chunk-000').mkdir(parents=True)
        self.info = dict(codebase_version='v3.0', total_episodes=2, chunks_size=1000,
                         fps=10, data_path='data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
                         video_path='videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4',
                         features={'observation.images.cam': {'dtype': 'video'}})
        self.write_json('info.json', self.info)
        self.write_json('stats.json', {})
        self.write_json('modality.json', {
            'state': {'arm': {'start': 0, 'end': 2}},
            'action': {'arm': {'start': 0, 'end': 2}},
            'video': {'cam': {'original_key': 'observation.images.cam'}},
            'annotation': {'human.task_description': {'original_key': 'task_index'}}})
        # Two episodes share one parquet AND one video; ids are not row positions.
        self.records = []
        for ep, start, offset in [(3, 100, 0.0), (8, 103, 0.3)]:
            self.records.append({'episode_index': ep, 'length': 3, 'tasks': ['pick'],
                'data/chunk_index': 0, 'data/file_index': 0,
                'dataset_from_index': start, 'dataset_to_index': start + 3,
                'videos/observation.images.cam/chunk_index': 0,
                'videos/observation.images.cam/file_index': 0,
                'videos/observation.images.cam/from_timestamp': offset,
                'videos/observation.images.cam/to_timestamp': offset + 0.3})
        self.save_records()
        pd.DataFrame({'task': ['pick'], 'task_index': [7]}).set_index('task').to_parquet(self.root / 'meta/tasks.parquet')
        self.rows = pd.DataFrame({'index': range(100, 106), 'episode_index': [3]*3+[8]*3,
            'frame_index': [0,1,2]*2, 'timestamp': [0,.1,.2]*2, 'task_index': [7]*6,
            'observation.state': [[i, i+1] for i in range(6)],
            'action': [[i+10, i+11] for i in range(6)]})
        self.rows.to_parquet(self.root / 'data/chunk-000/file-000.parquet')
        self.config = {k: ModalityConfig(delta_indices=[0], modality_keys=[v]) for k,v in
            [('state','arm'), ('action','arm'), ('video','cam'), ('language','annotation.human.task_description')]}

    def write_json(self, name, value):
        (self.root / 'meta' / name).write_text(json.dumps(value))

    def save_records(self):
        pd.DataFrame(self.records).to_parquet(self.root / 'meta/episodes/chunk-000/file-000.parquet')

    def loader(self):
        return LeRobotEpisodeLoader(self.root, self.config)

    def test_shared_parquet_isolated_and_language_mapped(self):
        loader = self.loader()
        frame = loader._load_parquet_data(8)
        self.assertEqual(len(frame), 3)
        np.testing.assert_array_equal(frame['action.arm'].iloc[0], [13,14])
        self.assertEqual(frame['language.annotation.human.task_description'].tolist(), ['pick']*3)

    def test_shared_video_offset_and_sparse_decode(self):
        with patch('gr00t.data.dataset.lerobot_episode_loader.get_frames_by_timestamps',
                   return_value=np.zeros((2,2,2,3), dtype=np.uint8)) as decode:
            frame = self.loader().load_episode(1, select_frames=lambda n: np.array([0,2]))
            self.assertIsNone(frame['video.cam'].iloc[1])
            self.assertTrue(decode.call_args.args[0].endswith('file-000.mp4'))
            np.testing.assert_allclose(decode.call_args.args[1], [.3,.5])

    def test_missing_rows_rejected(self):
        self.rows.iloc[:-1].to_parquet(self.root / 'data/chunk-000/file-000.parquet')
        with self.assertRaisesRegex(ValueError, 'row count'):
            self.loader()._load_parquet_data(8)

    def test_global_index_mismatch_rejected(self):
        self.rows.loc[5, 'index'] = 999
        self.rows.to_parquet(self.root / 'data/chunk-000/file-000.parquet')
        with self.assertRaisesRegex(ValueError, 'global rows'):
            self.loader()._load_parquet_data(8)

    def test_duplicate_episode_rejected(self):
        self.records[1]['episode_index'] = 3
        self.save_records()
        with self.assertRaisesRegex(ValueError, 'Duplicate episode'):
            self.loader()

    def test_video_interval_cannot_cross_episode(self):
        self.records[1]['videos/observation.images.cam/to_timestamp'] = .4
        self.save_records()
        with self.assertRaisesRegex(ValueError, 'too short'):
            self.loader()._load_video_data(8, np.array([2]))

    def test_v2_keeps_index_decoder(self):
        self.info.update(codebase_version='v2.1', total_episodes=1,
            data_path='data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
            video_path='videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4')
        self.write_json('info.json', self.info)
        (self.root/'meta/episodes.jsonl').write_text(json.dumps({'episode_index':3,'length':3,'tasks':['pick']})+'\n')
        (self.root/'meta/tasks.jsonl').write_text(json.dumps({'task_index':7,'task':'pick'})+'\n')
        self.rows.iloc[:3].to_parquet(self.root/'data/chunk-000/episode_000003.parquet')
        with patch('gr00t.data.dataset.lerobot_episode_loader.get_frames_by_indices',
                   return_value=np.zeros((3,2,2,3), dtype=np.uint8)) as decode:
            self.assertEqual(len(self.loader()[0]), 3)
            np.testing.assert_array_equal(decode.call_args.args[1], [0,1,2])


if __name__ == '__main__':
    unittest.main()
