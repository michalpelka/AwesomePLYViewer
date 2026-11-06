# Gaussian splat viewer (raylib), GPU version

A minimal 3D Gaussian Splatting viewer: one C++ file, commented step by step (STEP 1–10 in `splat_viewer.cpp`), and
`camera_link.cpp` that lets several viewers share their camera.
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

To install the viewer, run `cmake --install build`. It puts `splat_viewer` in `/usr/local/bin` by default; add
`--prefix ~/.local` to install it in `~/.local/bin` instead. Only the executable is installed: the shaders are built
into it, and raylib is linked statically. (libzmq, below, stays a shared library of the system.)

On Linux, building raylib needs the X11 and OpenGL development packages, on Debian/Ubuntu:
`sudo apt install libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libgl1-mesa-dev`.

Viewers open on one computer share their camera through [ZeroMQ](https://zeromq.org), which CMake looks for with
pkg-config: `sudo apt install libzmq3-dev` on Debian/Ubuntu, `brew install zeromq` on macOS. Without it, or with
`-DSPLAT_VIEWER_LINK=OFF`, the viewer is built without the camera link.

For a GPU or driver without OpenGL 4.3, configure with `-DOPENGL_VERSION=3.3`: the viewer then culls and sorts on
the CPU. macOS gets this automatically, since its OpenGL stops at 4.1.

## Run

    ./build/splat_viewer               # procedural demo scene
    ./build/splat_viewer scene.ply     # standard 3DGS point_cloud.ply (with spherical harmonics)
    ./build/splat_viewer scene.splat   # compact .splat (antimatter15 web viewer format)
    ./build/splat_viewer a.ply b.ply   # one viewer per file, flying together (below)

To open a scene without restarting, drop a `.ply` or `.splat` file on the window.

Several viewers open at once share one camera and view: fly in any of them, or pick a view with `1` … `7`, and the
others follow. This lets you compare two scenes, say two training iterations of one capture, or one scene in two
views (`M`) or renderers, side by side. Give the viewer several files and it opens one viewer for each:

    ./build/splat_viewer iteration_7000.ply iteration_30000.ply

Dropping files works the same way. The first file dropped on the demo scene takes its place, and each file dropped
after that, on any viewer, opens in a new viewer that flies with the others. Of several files dropped at once, the
first goes in place of the demo scene, and each of the others gets a new viewer. Each window's title names its file
and the folder it is in, as in `iteration_7000/point_cloud.ply`.

The first viewer becomes the server, on TCP ports 25150 and 25151 of 127.0.0.1, and the ones after it connect to it
as clients. When the server quits, a client takes its place within 2 s. The camera is shared in the file's own
coordinates, so viewers whose scenes were turned to different up axes (`Z`, `F`) still look at the same part of
it. `L` unlinks a viewer, and links it again; a viewer that joins takes the camera and view the others have. An
unlinked viewer loads a dropped file in place of its scene, as the demo scene does. `M` keeps the camera linked but
lets a viewer pick views of its own: its picks no longer switch the others, nor theirs it; `M` again takes the view
picked last.

| Input | Action |
|---|---|
| Drop a file on the window | open it: in place of the demo scene, or else in a new viewer that flies with this one |
| `W` `A` `S` `D` | fly forward / left / back / right (forward is where the camera looks) |
| `Q` `E` | fly straight down / up |
| Left or right drag | look around |
| Middle drag | pan |
| Wheel | fly speed (shown in the HUD) |
| Hold `Shift` / `Ctrl` | fly 4x faster / slower |
| `1` … `7` | view: Gaussians / 2-sigma ellipse outlines / centers only / solid 2-sigma ellipsoids / Gaussians colored by how far off the linearized projection is (blue: exact, red: far off) / points with normal arrows (shortest axis, faded for round splats) / Gaussians evaluated per pixel ray, without the linearization (as 3DGUT) |
| `O` | depth sorting on/off (shows why sorting matters) |
| `H` | spherical-harmonics (view-dependent) color on/off |
| `+` `-` | shrink / grow every splat |
| `Z` | swap Z up (scans, point clouds; files open this way) and Z front (COLMAP, -Y up; most 3DGS scenes) |
| `F` | cycle the up axis: Z front (COLMAP, -Y up) / Z up / +Y up |
| `G` | renderer: GPU, sorted on GPU (default) / GPU, sorted on CPU / CPU reference |
| `[` `]` | the sigma of the ellipses (view 2) and ellipsoids (views 4, 5): 0.5 to 5, 2 by default |
| `B` | flat quads: the fragment shader only fills each splat's quad with its color, to show what the vertex shader drew |
| `C` | save all views from this camera as one picture, `views_<date>_<time>.png` in the folder the viewer was started from |
| `R` | reset camera |
| `L` | link the camera and view with the other viewers on this computer on/off (on by default) |
| `M` | share the view (`1` … `7`) with the linked viewers on/off (on by default); the camera stays linked |

## The pipeline, as laid out in the code

1. **Splat**: position, 3 sigmas, rotation quaternion, color, opacity.
2. **Load**: `.ply` raw values go through the training activations (`exp` scale, `sigmoid` opacity, SH DC color).
3. **3D covariance** once per splat: `Σ = R S Sᵀ Rᵀ`.
4. **Fly camera**: position, yaw and pitch give the right/up/forward basis; pinhole projection.
5. **Project**: `Σ' = J W Σ Wᵀ Jᵀ` plus a 0.3 px² low-pass filter. **5b**: view-dependent color from SH degrees 0–3.
6. **Ellipse**: eigen-decomposition of the 2×2 `Σ'` gives the axes and the 1-sigma radii. **6b**: the exact
   outline of the 2-sigma ellipsoid under perspective, from its projected dual quadric; the linearized ellipse of
   view 2 is off for splats that are large and close to the camera or the screen edge. **6c** (view 4): inside that
   outline the fragment shader casts each pixel's ray at the 2-sigma ellipsoid and shades the hit, drawing every
   splat as a solid ellipsoid. **6d** (view 5): the Hellinger distance between the linearized 2D Gaussian and the
   exact one colors each splat, so artifacts of the linearization (smears near the camera) show up red.
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

Files already in `examples/` are skipped, so re-running configure does not download them again. The three scenes
come from COLMAP, with -Y up: files open Z up, so press `Z` to turn them the right way up.

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
