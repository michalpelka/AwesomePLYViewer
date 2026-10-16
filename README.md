# Gaussian splat viewer (raylib), GPU version

A minimal 3D Gaussian Splatting viewer: one C++ file, commented step by step (STEP 1–9 in `splat_viewer.cpp`).
STEPs 1–8 run the whole render pipeline on the CPU, so it can be read and stepped through in a debugger. STEP 9
moves the per-splat work into a vertex shader; it is the default renderer, and `G` switches back to the CPU path.

This is the faster version of the CPU-only viewer at https://gist.github.com/michalpelka/fc7d1595f439192277ad04d7be1a248a,
which is the shorter read if you only want to learn the pipeline.

## Get the code

The code lives in a GitHub gist, which is a small git repository, so it clones like any other (no GitHub login needed):

    git clone https://gist.github.com/90d61e34e235524e20deec411371bc9e.git splat_viewer_gpu
    cd splat_viewer_gpu

Or with the GitHub CLI: `gh gist clone 90d61e34e235524e20deec411371bc9e splat_viewer_gpu`.

A gist cannot contain folders, so all files sit at the top level. Run the commands below from inside `splat_viewer_gpu/`.
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
| `G` | renderer: GPU (default) / CPU reference |
| `R` | reset camera |

## The pipeline, as laid out in the code

1. **Splat**: position, 3 sigmas, rotation quaternion, color, opacity.
2. **Load**: `.ply` raw values go through the training activations (`exp` scale, `sigmoid` opacity, SH DC color).
3. **3D covariance** once per splat: `Σ = R S Sᵀ Rᵀ`.
4. **Orbit camera**: right/up/forward basis and pinhole projection.
5. **Project**: `Σ' = J W Σ Wᵀ Jᵀ` plus a 0.3 px² low-pass filter. **5b**: view-dependent color from SH degrees 0–3.
6. **Ellipse**: eigen-decomposition of the 2×2 `Σ'` gives the axes and the 1-sigma radii.
7. **Sort** back to front with a 16-bit counting sort: about 4 ms for 580k splats, where `std::sort` takes about 40 ms.
8. **Draw** one quad per splat; the fragment shader evaluates `opacity · exp(-½|uv|²)` and alpha blending composites.
9. **GPU renderer**: all splat data is uploaded once into a float texture. Per frame the CPU only culls, sorts and
   uploads the draw order (4 bytes per splat), and one instanced draw call runs steps 5, 5b and 6 in the vertex shader.

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

On the train scene (742k splats, 580k visible) the GPU renderer takes about 8.5 ms per frame: 4.4 ms to cull and
3.6 ms to sort on the CPU, with the GPU work finishing within that time. The CPU reference path takes about 68 ms. What remains is
CPU culling and sorting, single-threaded. The splat texture takes 304 bytes per splat in GPU memory (about 225 MB
for the train scene). Compressed formats (`.spz`, compressed `.ply`) are not supported.
