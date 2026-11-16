# RealMan v3 loader preparation

This example supplies a field mapping for the inspected Ball—Placement dataset.
It is not a complete training configuration. Action units and absolute versus
incremental control semantics have not yet been established.

The mapping keeps all 58 state/action dimensions in their original order:
left arm (7), left hand (22), right arm (7), right hand (22). All five video keys
are mapped. The tactile force tensor remains in the source dataset but is not
added as a model input by this mapping. Training input selection is a separate
configuration step.

After reviewing the mapping, copy modality.json to the dataset's meta directory
without overwriting an existing mapping. The task description in the source
currently describes data collection, not the actual ball placement task; review
and correct task annotations before multitask training.

The loader detects codebase_version v3.0 and reads episode/task Parquet metadata.
It filters shared data files by episode id and validates global row indices,
episode lengths, frame indices, and timestamps. Video timestamps include the
per-view offset in the shared video file. Missing or inconsistent metadata fails
explicitly. V2 behavior is preserved. V3 mask loading is not implemented.

Validation performed before application: seven CPU unit tests; all 50 episodes
and 36,299 rows of the user's dataset; all 250 video paths/time intervals; video
container frame counts versus episode lengths. Real video decoding, statistics
pipeline integration, and end-to-end training remain to be tested once the full
environment is installed. The offline checks bypassed package-level startup to
use the completed pandas/pyarrow wheel cache; they did not replace those libraries
or the loader implementation with stubs. Video decoder calls were mocked for
loader tests; ffprobe was used separately for real video container metadata.

Once dependencies are installed, run from the repository root:

    python -m unittest discover -s tests/gr00t/data -p test_lerobot_v3.py -v
