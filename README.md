# Gaussian splat viewer (raylib)

A minimal 3D Gaussian Splatting viewer: one C++ file that runs the whole render pipeline on the CPU,
commented step by step (STEP 1–8 in `splat_viewer.cpp`). It is written to be read, not to be fast.

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
7. **Sort** back to front with a 16-bit counting sort.
8. **Draw** one quad per splat; the fragment shader evaluates `opacity · exp(-½|uv|²)` and alpha blending composites.

## Example scenes (not included)

- `nike.splat`, `plush.splat`: https://huggingface.co/cakewalk/splat-data (no license stated for these files).
- `train`: `FO_dataset/train/point_cloud/iteration_7000/point_cloud.ply` from
  https://huggingface.co/datasets/Voxel51/gaussian_splatting (184 MB, full 3DGS `.ply` with SH degree 3).

## Limits

Everything is on the CPU. At about 1M splats a frame takes roughly 18 ms to project, 8 ms to sort and 130 ms to
submit through raylib's immediate mode, so large captures are slow. The next step for speed is moving projection
into a vertex shader. Compressed formats (`.spz`, compressed `.ply`) are not supported.
