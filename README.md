# Gaussian splat viewer (raylib), GPU version

A minimal 3D Gaussian Splatting viewer: one C++ file, commented step by step (STEP 1–10 in `splat_viewer.cpp`).
STEPs 1–8 run the whole render pipeline on the CPU, so it can be read and stepped through in a debugger. STEP 9
moves the per-splat projection into a vertex shader, and STEP 10 moves culling and sorting into compute shaders, so
the CPU does no per-splat work at all. STEP 10 is the default renderer; `G` switches to the other two.

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

Needs CMake ≥ 3.16. CMake fetches [raylib](https://github.com/raysan5/raylib) 5.5 and
[happly](https://github.com/nmwsharp/happly) (PLY reader), and builds raylib for OpenGL 4.3, because the GPU sort
(STEP 10) needs its compute shaders and a usual raylib install is built for OpenGL 3.3.

    cmake -S . -B build && cmake --build build

On Linux, building raylib needs the X11 and OpenGL development packages, on Debian/Ubuntu:
`sudo apt install libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libgl1-mesa-dev`.

For a GPU or driver without OpenGL 4.3, configure with `-DOPENGL_VERSION=3.3`: the viewer then culls and sorts on
the CPU. macOS gets this automatically, since its OpenGL stops at 4.1.

## Run

    ./build/splat_viewer               # procedural demo scene
    ./build/splat_viewer scene.ply     # standard 3DGS point_cloud.ply (with spherical harmonics)
    ./build/splat_viewer scene.splat   # compact .splat (antimatter15 web viewer format)

To open another scene without restarting, drop a `.ply` or `.splat` file on the window.

| Input | Action |
|---|---|
| Drop a file on the window | open it in place of the current scene |
| `W` `A` `S` `D` | fly forward / left / back / right (forward is where the camera looks) |
| `Q` `E` | fly straight down / up |
| Left or right drag | look around |
| Middle drag | pan |
| Wheel | fly speed (shown in the HUD) |
| Hold `Shift` / `Ctrl` | fly 4x faster / slower |
| `1` `2` `3` | view: Gaussians / 2-sigma ellipse outlines / centers only |
| `O` | depth sorting on/off (shows why sorting matters) |
| `H` | spherical-harmonics (view-dependent) color on/off |
| `+` `-` | shrink / grow every splat |
| `Z` | swap Z up (scans, point clouds) and Z front (COLMAP, -Y up) |
| `F` | cycle the up axis: Z front (COLMAP, -Y up) / Z up / +Y up |
| `G` | renderer: GPU, sorted on GPU (default) / GPU, sorted on CPU / CPU reference |
| `R` | reset camera |

## The pipeline, as laid out in the code

1. **Splat**: position, 3 sigmas, rotation quaternion, color, opacity.
2. **Load**: `.ply` raw values go through the training activations (`exp` scale, `sigmoid` opacity, SH DC color).
3. **3D covariance** once per splat: `Σ = R S Sᵀ Rᵀ`.
4. **Fly camera**: position, yaw and pitch give the right/up/forward basis; pinhole projection.
5. **Project**: `Σ' = J W Σ Wᵀ Jᵀ` plus a 0.3 px² low-pass filter. **5b**: view-dependent color from SH degrees 0–3.
6. **Ellipse**: eigen-decomposition of the 2×2 `Σ'` gives the axes and the 1-sigma radii.
7. **Sort** back to front with a 16-bit counting sort: about 4 ms for 580k splats, where `std::sort` takes about 40 ms.
8. **Draw** one quad per splat; the fragment shader evaluates `opacity · exp(-½|uv|²)` and alpha blending composites.
9. **GPU renderer**: all splat data is uploaded once into four textures: position, opacity and covariance as 32-bit
   floats, and the SH coefficients as half floats, one texture per color channel. Per frame the CPU only culls, sorts
   and uploads the draw order (4 bytes per splat), and one instanced draw call runs steps 5, 5b and 6 in the vertex
   shader.
10. **GPU cull and sort**: compute shaders cull the splats and sort them with the same 16-bit depth key as step 7,
   using a radix sort (two passes of 8 bits). The sort is stable like the CPU one, so splats at equal depth do not
   swap places from frame to frame. It writes the order straight into the buffer step 9 draws from; only the
   visible count comes back to the CPU. Needs OpenGL 4.3.

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

Measured on an RTX 4080, without vsync. On the train scene (742k splats) a frame takes about 1.6 ms with the GPU
sort: 0.4 ms to cull and sort and about 1 ms to draw, both on the GPU. With the CPU sort it takes about 8.5 ms
(4.4 ms to cull and 3.6 ms to sort, single-threaded), and the CPU reference path takes about 68 ms.

The splat textures take 144 bytes per splat in GPU memory with SH degree 3 (about 100 MB for the train scene, 72 bytes
per splat without SH), and the GPU sort adds 12 bytes per splat. The textures hold up to 67M splats with SH degree 3
and 89M without; larger scenes fall back to the CPU renderer.

The train scene tiled 25 times (18.5M splats, a 4.6 GB `.ply`) loads in 14 s and needs 10.5 GB of RAM while loading
and 2.8 GB of GPU memory. With the GPU sort it runs at about 12 FPS: 10–12 ms to cull and sort, and about 75 ms to
draw 18.4M overlapping splats, so the draw itself is now the limit. With the CPU sort it runs at about 4 FPS, as
culling and sorting take about 100 ms each. Compressed formats (`.spz`, compressed `.ply`) are not supported.
