# Gaussian splat viewer (raylib)

A minimal 3D Gaussian Splatting viewer: one C++ file that runs the whole render pipeline on the CPU,
commented step by step (STEP 1–8 in `splat_viewer.cpp`). It is written to be read, not to be fast.

## Get the code

The code lives in a GitHub gist, which is a small git repository, so it clones like any other (no GitHub login needed):

    git clone https://gist.github.com/fc7d1595f439192277ad04d7be1a248a.git splat_viewer
    cd splat_viewer

Or with the GitHub CLI: `gh gist clone fc7d1595f439192277ad04d7be1a248a splat_viewer`.

A gist cannot contain folders, so all files sit at the top level. Run the commands below from inside `splat_viewer/`.
`git pull` there picks up later updates to the gist.

## Build

Needs raylib 5.5 and CMake ≥ 3.16. [happly](https://github.com/nmwsharp/happly) (PLY reader) is fetched by CMake.

    cmake -S . -B build && cmake --build build

`CMakeLists.txt` links OpenGL/X11 explicitly because it targets a static raylib on Linux. On other setups you may
only need `raylib`.

## Run

    ./build/splat_viewer               # procedural demo scene
    ./build/splat_viewer scene.ply     # standard 3DGS point_cloud.ply (with spherical harmonics)
    ./build/splat_viewer scene.splat   # compact .splat (antimatter15 web viewer format)

| Input | Action |
|---|---|
| Left drag / right or middle drag / wheel | orbit / pan / zoom |
| `1` `2` `3` | view: Gaussians / 2-sigma ellipse outlines / centers only |
| `S` | depth sorting on/off (shows why sorting matters) |
| `H` | spherical-harmonics (view-dependent) color on/off |
| `+` `-` | shrink / grow every splat |
| `F` | flip Y (COLMAP scenes are Y-down) |
| `R` | reset camera |

## The pipeline, as laid out in the code

1. **Splat**: position, 3 sigmas, rotation quaternion, color, opacity.
2. **Load**: `.ply` raw values go through the training activations (`exp` scale, `sigmoid` opacity, SH DC color).
3. **3D covariance** once per splat: `Σ = R S Sᵀ Rᵀ`.
4. **Orbit camera**: right/up/forward basis and pinhole projection.
5. **Project**: `Σ' = J W Σ Wᵀ Jᵀ` plus a 0.3 px² low-pass filter. **5b**: view-dependent color from SH degrees 0–3.
6. **Ellipse**: eigen-decomposition of the 2×2 `Σ'` gives the axes and the 1-sigma radii.
7. **Sort** back to front with `std::sort`. This is simple but not optimal: a 16-bit counting/radix sort is about 10× faster.
8. **Draw** one quad per splat; the fragment shader evaluates `opacity · exp(-½|uv|²)` and alpha blending composites.

## Example scenes (not included, CMake can download them)

    cmake -S . -B build -DSPLAT_VIEWER_EXAMPLES=ON    # ~200 MB into examples/, checked by SHA-256
    ./build/splat_viewer examples/train/train.ply

Files already in `examples/` are skipped, so re-running configure does not download them again.

- `nike/nike.splat` (9 MB), `plush/plush.splat` (9 MB): https://huggingface.co/cakewalk/splat-data. No license is
  stated for these files, so they are for local learning only.
- `train/train.ply` (184 MB, full 3DGS `.ply` with SH degree 3):
  `FO_dataset/train/point_cloud/iteration_7000/point_cloud.ply` from
  https://huggingface.co/datasets/Voxel51/gaussian_splatting (dataset card: Apache-2.0). The scene comes from
  Tanks and Temples and was trained with Inria's gaussian-splatting code, which have their own terms.

## Limits

Everything is on the CPU. On the train scene (580k visible splats) a frame takes about 25 ms to project and shade
(with SH), 40 ms to sort and 35 ms to submit through raylib's immediate mode, roughly 10 FPS. The next steps for
speed are a counting/radix sort (see the note at STEP 7) and moving projection into a vertex shader. Compressed
formats (`.spz`, compressed `.ply`) are not supported.
