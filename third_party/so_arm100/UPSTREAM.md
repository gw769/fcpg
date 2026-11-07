# SO-ARM100 upstream

The minimal SO-101 MuJoCo model in this directory comes from
[`TheRobotStudio/SO-ARM100`](https://github.com/TheRobotStudio/SO-ARM100).
It is distributed under the included Apache License 2.0.

Only the files required by this project's SO-101 forward-kinematics and
synthetic-camera experiments are vendored: the new-calibration MJCF, its scene,
and referenced visual meshes. Printable parts, media, and unrelated robot
descriptions from the local upstream checkout are intentionally excluded.
