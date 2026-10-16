// ============================================================================
//  Minimal Gaussian Splat viewer (raylib)
//
//  Goal: be easy to read, not fast. Every stage of the 3D Gaussian Splatting
//  render pipeline is done on the CPU in plain, C-style C++ so you can step
//  through it:
//
//    STEP 1  What a splat is                         (struct Splat)
//    STEP 2  Get splats: load a 3DGS .ply or .splat, or build a demo scene
//    STEP 3  Precompute each splat's 3D covariance   Σ = R S Sᵀ Rᵀ
//    STEP 4  Orbit camera -> view basis (right, up, forward)
//    STEP 5  Project every splat to the screen       Σ' = J W Σ Wᵀ Jᵀ
//    STEP 5b Color for this view: spherical harmonics (H turns them off)
//    STEP 6  Turn the 2D covariance into an ellipse  (eigen-decomposition)
//    STEP 7  Sort back to front                      (counting sort on depth)
//    STEP 8  Draw one quad per splat, the fragment shader evaluates the
//            Gaussian and alpha blending composites them
//
//  Usage:  ./splat_viewer              -> procedural demo scene
//          ./splat_viewer scene.ply    -> standard 3DGS output .ply (read with happly)
//          ./splat_viewer scene.splat  -> compact .splat (web viewer format)
// ============================================================================

#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"

#include "happly.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// ----------------------------------------------------------------------------
// STEP 1: What a splat is
//
// A splat is a 3D Gaussian "blob": a fuzzy ellipsoid with a color.
//   position : center of the ellipsoid
//   scale    : standard deviation (sigma) along the ellipsoid's 3 local axes
//   rotation : orientation of those local axes
//   color    : RGB, plus A = opacity at the very center of the blob
// That is enough for the basic render. Trained .ply files also store
// view-dependent color as spherical harmonics (struct SphericalHarmonics,
// evaluated in STEP 5b). With those off, `color` is used from every angle.
// ----------------------------------------------------------------------------
struct Splat {
    Vector3 position;
    Vector3 scale;
    Quaternion rotation;  // raylib quaternion: {x, y, z, w}
    Color color;          // a = opacity
    float cov[6];         // derived in STEP 3: Σ as xx, xy, xz, yy, yz, zz
};

// View-dependent color, kept next to the splats (same index). Only .ply files
// carry it; for .splat files and the demo scene `degree` stays 0.
struct SphericalHarmonics {
    int degree = 0;               // 0 = base color only, up to 3
    int perSplat = 1;             // (degree+1)² coefficients per splat: 1, 4, 9 or 16
    std::vector<Vector3> coeffs;  // RGB per coefficient; [i*perSplat + 0] is f_dc
};

// Per-frame result of projecting one splat to the screen (STEP 5 + 6).
struct ProjectedSplat {
    Vector2 center;  // pixel position of the splat center
    Vector2 axis1;   // ellipse major axis, length = 1 sigma, in pixels
    Vector2 axis2;   // ellipse minor axis, length = 1 sigma, in pixels
    float depth;     // distance along the camera's forward axis
    Color color;     // color seen from this camera (STEP 5b), a = opacity
};

enum ViewMode { VIEW_GAUSSIANS = 0, VIEW_ELLIPSES, VIEW_CENTERS };

// ----------------------------------------------------------------------------
// STEP 2a: Load a .ply produced by 3D Gaussian Splatting training
//
// A PLY file is an ASCII header that declares "elements" and their
// "properties", followed by the data. A 3DGS file has one element, "vertex",
// with one entry per splat:
//     element vertex 1063091
//     property float x            <- position
//     property float nx           <- normals, unused (always 0)
//     property float f_dc_0       <- color (SH coefficient 0, per channel)
//     property float f_rest_0     <- 45 view-dependent color coefficients (STEP 5b)
//     property float opacity      <- opacity, stored BEFORE sigmoid
//     property float scale_0      <- scale, stored as log(sigma)
//     property float rot_0        <- quaternion, w first
//     ...
// happly parses the header and the data, then hands each property back as a
// column: a std::vector<float> with one value per splat.
// Training stores "raw" values that are easier to optimise. We convert them
// to real quantities with the same activation functions training uses.
// ----------------------------------------------------------------------------
static float Sigmoid(float x) { return 1.0f/(1.0f + expf(-x)); }

static std::vector<Splat> LoadSplatsFromPly(const char *path, SphericalHarmonics *sh)
{
    std::vector<Splat> splats;
    try {
        happly::PLYData ply(path);  // reads the whole file; throws std::runtime_error on problems
        happly::Element &vertex = ply.getElement("vertex");
        auto column = [&](const std::string &name) { return vertex.getProperty<float>(name); };

        std::vector<float> x = column("x"), y = column("y"), z = column("z");
        std::vector<float> dc0 = column("f_dc_0"), dc1 = column("f_dc_1"), dc2 = column("f_dc_2");
        std::vector<float> opacity = column("opacity");
        std::vector<float> s0 = column("scale_0"), s1 = column("scale_1"), s2 = column("scale_2");
        std::vector<float> r0 = column("rot_0"), r1 = column("rot_1"), r2 = column("rot_2"), r3 = column("rot_3");

        const float SH_C0 = 0.28209479177387814f;  // constant of the 0th spherical harmonic
        splats.resize(x.size());
        for (size_t i = 0; i < splats.size(); i++) {
            Splat &sp = splats[i];
            sp.position = Vector3{ x[i], y[i], z[i] };

            // color = 0.5 + C0 * f_dc   (the view-independent part of the SH color)
            sp.color.r = (unsigned char)(Clamp(0.5f + SH_C0*dc0[i], 0.0f, 1.0f)*255.0f);
            sp.color.g = (unsigned char)(Clamp(0.5f + SH_C0*dc1[i], 0.0f, 1.0f)*255.0f);
            sp.color.b = (unsigned char)(Clamp(0.5f + SH_C0*dc2[i], 0.0f, 1.0f)*255.0f);
            // opacity = sigmoid(raw)  -> always in (0, 1)
            sp.color.a = (unsigned char)(Sigmoid(opacity[i])*255.0f);
            // sigma = exp(raw)        -> always positive
            sp.scale = Vector3{ expf(s0[i]), expf(s1[i]), expf(s2[i]) };
            // the file stores (w, x, y, z), raylib wants {x, y, z, w}; training does not
            // keep it unit length, so normalize
            sp.rotation = QuaternionNormalize(Quaternion{ r1[i], r2[i], r3[i], r0[i] });
        }

        // Spherical harmonics. f_rest_* holds degrees 1..3 channel by channel:
        // first all red coefficients, then all green, then all blue. Regroup
        // them as one RGB triple per coefficient, after f_dc as coefficient 0.
        int restCount = 0;
        while (vertex.hasProperty("f_rest_" + std::to_string(restCount))) restCount++;
        int perChannel = restCount/3;  // 3, 8 or 15 for degree 1, 2 or 3
        sh->degree = (perChannel >= 15) ? 3 : (perChannel >= 8) ? 2 : (perChannel >= 3) ? 1 : 0;
        sh->perSplat = (sh->degree + 1)*(sh->degree + 1);
        if (sh->degree > 0) {
            sh->coeffs.resize(splats.size()*sh->perSplat);
            for (size_t i = 0; i < splats.size(); i++) sh->coeffs[i*sh->perSplat] = Vector3{ dc0[i], dc1[i], dc2[i] };
            for (int k = 1; k < sh->perSplat; k++) {
                std::vector<float> r = column("f_rest_" + std::to_string(k - 1));
                std::vector<float> g = column("f_rest_" + std::to_string(perChannel + k - 1));
                std::vector<float> b = column("f_rest_" + std::to_string(2*perChannel + k - 1));
                for (size_t i = 0; i < splats.size(); i++) sh->coeffs[i*sh->perSplat + k] = Vector3{ r[i], g[i], b[i] };
            }
        }
    } catch (const std::exception &e) {
        TraceLog(LOG_ERROR, "Cannot load %s: %s", path, e.what());
        TraceLog(LOG_ERROR, "A Gaussian Splatting .ply needs x y z, f_dc_0..2, opacity, scale_0..2, rot_0..3");
        return {};
    }

    TraceLog(LOG_INFO, "Loaded %d splats from %s (spherical harmonics degree %d)", (int)splats.size(), path, sh->degree);
    return splats;
}

// ----------------------------------------------------------------------------
// STEP 2b: Load a .splat file (the compact format of many web viewers)
//
// There is no header, just one 32-byte record per splat:
//     float x, y, z          12 bytes  position
//     float sx, sy, sz       12 bytes  sigma, exp() already applied
//     uint8 r, g, b, a        4 bytes  color and opacity, already activated
//     uint8 qw, qx, qy, qz    4 bytes  rotation, stored as q*128 + 128
// It holds the same information as the .ply, but the activation functions
// were applied before saving, color and rotation are squeezed into bytes, and
// the 45 view-dependent color coefficients are dropped (so H has nothing to
// switch off). That makes it about 8x smaller (32 bytes instead of 248).
// ----------------------------------------------------------------------------
static std::vector<Splat> LoadSplatsFromSplatFile(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) { TraceLog(LOG_ERROR, "Cannot open %s", path); return {}; }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (size % 32 != 0) TraceLog(LOG_WARNING, "%s: size is not a multiple of 32 bytes", path);

    std::vector<Splat> splats;
    splats.reserve((size_t)(size/32));
    unsigned char record[32];
    while (fread(record, sizeof(record), 1, f) == 1) {
        float v[6];
        memcpy(v, record, sizeof(v));
        const unsigned char *rgba = record + 24;
        const unsigned char *q = record + 28;

        Splat sp = {};
        sp.position = Vector3{ v[0], v[1], v[2] };
        sp.scale = Vector3{ v[3], v[4], v[5] };
        sp.color = Color{ rgba[0], rgba[1], rgba[2], rgba[3] };
        // undo the byte packing: (byte - 128)/128 gives back -1..1; file order is w, x, y, z
        sp.rotation = QuaternionNormalize(Quaternion{ (q[1] - 128)/128.0f, (q[2] - 128)/128.0f,
                                                      (q[3] - 128)/128.0f, (q[0] - 128)/128.0f });
        splats.push_back(sp);
    }
    fclose(f);

    TraceLog(LOG_INFO, "Loaded %d splats from %s", (int)splats.size(), path);
    return splats;
}

// ----------------------------------------------------------------------------
// STEP 2c: Demo scene, built in code so you can see exactly what goes in.
//   - three big see-through ellipsoids  -> one splat = one soft ellipse,
//                                          and why draw order matters
//   - a sphere of small flat discs      -> many splats form a surface
//   - a checkered floor of flat discs
//   - a helix of long thin "needles"    -> scale + rotation = anisotropy
// ----------------------------------------------------------------------------
static Splat MakeSplat(Vector3 pos, Vector3 scale, Quaternion rot, Color color)
{
    Splat s = {};
    s.position = pos;
    s.scale = scale;
    s.rotation = rot;
    s.color = color;
    return s;
}

static std::vector<Splat> BuildDemoScene()
{
    const int sphereCount = 3000, floorSide = 60, helixCount = 160;
    std::vector<Splat> splats;

    // Three large, semi-transparent, flattened ellipsoids at different depths.
    splats.push_back(MakeSplat(Vector3{ 1.75f, 1.35f, -1.65f }, Vector3{ 0.40f, 0.25f, 0.04f },
                               QuaternionFromEuler(0.0f, 0.6f, 0.4f), Color{ 230, 40, 40, 220 }));
    splats.push_back(MakeSplat(Vector3{ 2.15f, 1.05f, -1.45f }, Vector3{ 0.40f, 0.25f, 0.04f },
                               QuaternionFromEuler(0.0f, 0.6f, -0.4f), Color{ 40, 220, 40, 220 }));
    splats.push_back(MakeSplat(Vector3{ 2.60f, 1.35f, -1.25f }, Vector3{ 0.40f, 0.25f, 0.04f },
                               QuaternionFromEuler(0.0f, 0.6f, 1.2f), Color{ 40, 80, 240, 220 }));

    // Sphere: points spread evenly with a Fibonacci spiral. Each splat is a flat
    // disc (tiny z scale) whose local z axis is turned to the surface normal.
    Vector3 sphereCenter = { 0.0f, 1.2f, 0.0f };
    for (int i = 0; i < sphereCount; i++) {
        float y = 1.0f - 2.0f*(i + 0.5f)/sphereCount;
        float r = sqrtf(1.0f - y*y);
        float phi = i*2.39996323f;  // golden angle
        Vector3 normal = { r*cosf(phi), y, r*sinf(phi) };
        Color c = { (unsigned char)(127 + 127*normal.x), (unsigned char)(127 + 127*normal.y),
                    (unsigned char)(127 + 127*normal.z), 230 };
        splats.push_back(MakeSplat(Vector3Add(sphereCenter, normal), Vector3{ 0.04f, 0.04f, 0.004f },
                                   QuaternionFromVector3ToVector3(Vector3{ 0, 0, 1 }, normal), c));
    }

    // Floor: a grid of flat discs lying in the XZ plane, checkered in 0.5 m tiles.
    Quaternion faceUp = QuaternionFromVector3ToVector3(Vector3{ 0, 0, 1 }, Vector3{ 0, 1, 0 });
    for (int iz = 0; iz < floorSide; iz++) {
        for (int ix = 0; ix < floorSide; ix++) {
            float x = -3.0f + 6.0f*ix/(floorSide - 1);
            float z = -3.0f + 6.0f*iz/(floorSide - 1);
            int checker = ((int)floorf(x*2.0f) + (int)floorf(z*2.0f)) & 1;
            Color c = checker ? Color{ 200, 200, 200, 230 } : Color{ 70, 70, 80, 230 };
            splats.push_back(MakeSplat(Vector3{ x, 0.0f, z }, Vector3{ 0.06f, 0.06f, 0.005f }, faceUp, c));
        }
    }

    // Helix: long thin splats whose long (local x) axis follows the helix tangent.
    for (int i = 0; i < helixCount; i++) {
        float t = (float)i/(helixCount - 1);
        float angle = t*6.0f*PI;
        Vector3 pos = { 1.7f*cosf(angle), 0.2f + 2.2f*t, 1.7f*sinf(angle) };
        Vector3 tangent = Vector3Normalize(Vector3{ -1.7f*sinf(angle)*6.0f*PI, 2.2f, 1.7f*cosf(angle)*6.0f*PI });
        Color c = { (unsigned char)(255*(1 - t) + 230*t), (unsigned char)(220*(1 - t) + 40*t), (unsigned char)(40 + 200*t), 240 };
        splats.push_back(MakeSplat(pos, Vector3{ 0.10f, 0.012f, 0.012f },
                                   QuaternionFromVector3ToVector3(Vector3{ 1, 0, 0 }, tangent), c));
    }

    return splats;
}

// ----------------------------------------------------------------------------
// STEP 3: 3D covariance  Σ = R S Sᵀ Rᵀ
//
// A Gaussian is fully described by its center and a 3x3 covariance matrix Σ.
// Build it from rotation R and the diagonal scale matrix S = diag(sx, sy, sz):
//     M = R S     (column j of R, stretched by sigma j)
//     Σ = M Mᵀ
// Σ is symmetric, so we keep just 6 numbers. Σ does not depend on the camera,
// so it is computed once, not every frame.
// ----------------------------------------------------------------------------
static void QuatToMat3(Quaternion q, float R[3][3])
{
    float x = q.x, y = q.y, z = q.z, w = q.w;
    R[0][0] = 1 - 2*(y*y + z*z); R[0][1] = 2*(x*y - w*z);     R[0][2] = 2*(x*z + w*y);
    R[1][0] = 2*(x*y + w*z);     R[1][1] = 1 - 2*(x*x + z*z); R[1][2] = 2*(y*z - w*x);
    R[2][0] = 2*(x*z - w*y);     R[2][1] = 2*(y*z + w*x);     R[2][2] = 1 - 2*(x*x + y*y);
}

static void ComputeCovariance3D(Splat *s)
{
    float R[3][3];
    QuatToMat3(s->rotation, R);
    float sc[3] = { s->scale.x, s->scale.y, s->scale.z };

    float M[3][3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) M[i][j] = R[i][j]*sc[j];

    float S[3][3];  // Σ = M Mᵀ
    for (int i = 0; i < 3; i++)
        for (int k = 0; k < 3; k++) S[i][k] = M[i][0]*M[k][0] + M[i][1]*M[k][1] + M[i][2]*M[k][2];

    s->cov[0] = S[0][0]; s->cov[1] = S[0][1]; s->cov[2] = S[0][2];
    s->cov[3] = S[1][1]; s->cov[4] = S[1][2]; s->cov[5] = S[2][2];
}

// Turn the whole scene 180 degrees about X. 3DGS scenes usually come from
// COLMAP, where +Y points down; raylib's camera expects +Y up.
static Quaternion SceneFlip() { return QuaternionFromAxisAngle(Vector3{ 1, 0, 0 }, PI); }

static void FlipSceneUpsideDown(std::vector<Splat> &splats)
{
    Quaternion flip = SceneFlip();
    for (Splat &s : splats) {
        s.position = Vector3RotateByQuaternion(s.position, flip);
        s.rotation = QuaternionNormalize(QuaternionMultiply(flip, s.rotation));
        ComputeCovariance3D(&s);
    }
}

// ----------------------------------------------------------------------------
// STEP 4: Camera
//
// A simple orbit camera: it sits on a sphere around `target`. From it we build
// an orthonormal basis. Expressing a point in this basis gives its
// camera-space coordinates:
//     x = how far right   y = how far up   z = depth (how far in front)
// The three basis vectors are the rows of the view rotation W used in STEP 5.
// ----------------------------------------------------------------------------
struct OrbitCamera {
    Vector3 target;
    float distance, yaw, pitch;
    float fovy;  // vertical field of view in radians
};

struct ViewParams {
    Vector3 eye, right, up, forward;
    float focal;      // focal length in pixels
    float cx, cy;     // principal point = screen center
    float tanHalfFovX, tanHalfFovY;
};

static ViewParams MakeView(const OrbitCamera *cam, int width, int height)
{
    ViewParams v;
    Vector3 offset = { cosf(cam->pitch)*sinf(cam->yaw), sinf(cam->pitch), cosf(cam->pitch)*cosf(cam->yaw) };
    v.eye = Vector3Add(cam->target, Vector3Scale(offset, cam->distance));
    v.forward = Vector3Normalize(Vector3Subtract(cam->target, v.eye));
    v.right = Vector3Normalize(Vector3CrossProduct(v.forward, Vector3{ 0, 1, 0 }));
    v.up = Vector3CrossProduct(v.right, v.forward);

    // Pinhole camera: a point at camera coords (x, y, z) lands on pixel
    //     u = cx + focal * x / z,   v = cy - focal * y / z
    // (minus because screen y grows downward). Choosing focal from the
    // field of view makes the top/bottom of the screen be at ±fovy/2.
    v.focal = 0.5f*height/tanf(0.5f*cam->fovy);
    v.cx = 0.5f*width;
    v.cy = 0.5f*height;
    v.tanHalfFovY = tanf(0.5f*cam->fovy);
    v.tanHalfFovX = v.tanHalfFovY*width/height;
    return v;
}

// ----------------------------------------------------------------------------
// STEP 5 + 6: Project one splat into a 2D screen-space ellipse
//
// A 3D Gaussian seen through a camera is (approximately) a 2D Gaussian on
// the screen. Its 2D covariance is
//     Σ' = J W Σ Wᵀ Jᵀ
//   W : rotates world -> camera axes (rows = right, up, forward)
//   J : Jacobian of the perspective projection at the splat center, i.e. the
//       best *linear* approximation of "divide by depth" near that point.
//       Perspective is not linear, so this is the one approximation that
//       makes splatting work (the "EWA splatting" trick).
// Then the 2x2 Σ' is turned into an ellipse with an eigen-decomposition:
//   eigenvectors = ellipse axis directions, sqrt(eigenvalues) = 1 sigma radii.
// Returns false if the splat is not visible.
// ----------------------------------------------------------------------------
static bool ProjectSplat(const Splat *s, const ViewParams *v, float scaleMod, ProjectedSplat *out)
{
    // --- camera-space position: dot with each basis vector ---
    Vector3 d = Vector3Subtract(s->position, v->eye);
    float tx = Vector3DotProduct(d, v->right);
    float ty = Vector3DotProduct(d, v->up);
    float tz = Vector3DotProduct(d, v->forward);
    if (tz < 0.2f) return false;  // behind the camera or too close (near plane)

    // Skip splats whose center is far outside the view. 1.3x the screen gives
    // big splats just off screen a chance to still poke in.
    float limX = 1.3f*v->tanHalfFovX, limY = 1.3f*v->tanHalfFovY;
    if (fabsf(tx/tz) > limX || fabsf(ty/tz) > limY) return false;

    // --- W: world -> camera rotation (rows are the camera axes) ---
    float W[3][3] = {
        { v->right.x,   v->right.y,   v->right.z   },
        { v->up.x,      v->up.y,      v->up.z      },
        { v->forward.x, v->forward.y, v->forward.z },
    };

    // --- J: derivatives of (u, v) with respect to (tx, ty, tz) ---
    //   u = cx + f*tx/tz  ->  du/dtx = f/tz,  du/dty = 0,      du/dtz = -f*tx/tz²
    //   v = cy - f*ty/tz  ->  dv/dtx = 0,     dv/dty = -f/tz,  dv/dtz =  f*ty/tz²
    float f = v->focal;
    float J[2][3] = {
        { f/tz, 0.0f,  -f*tx/(tz*tz) },
        { 0.0f, -f/tz,  f*ty/(tz*tz) },
    };

    // --- T = J W  (2x3): maps a small world-space offset to a pixel offset ---
    float T[2][3];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 3; j++) T[i][j] = J[i][0]*W[0][j] + J[i][1]*W[1][j] + J[i][2]*W[2][j];

    // --- Σ' = T Σ Tᵀ  (2x2). The scale modifier k multiplies sigma, so Σ by k² ---
    const float *c = s->cov;
    float k2 = scaleMod*scaleMod;
    float Sigma[3][3] = {
        { c[0]*k2, c[1]*k2, c[2]*k2 },
        { c[1]*k2, c[3]*k2, c[4]*k2 },
        { c[2]*k2, c[4]*k2, c[5]*k2 },
    };
    float TS[2][3];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 3; j++) TS[i][j] = T[i][0]*Sigma[0][j] + T[i][1]*Sigma[1][j] + T[i][2]*Sigma[2][j];
    float a = TS[0][0]*T[0][0] + TS[0][1]*T[0][1] + TS[0][2]*T[0][2];  // Σ'xx
    float b = TS[0][0]*T[1][0] + TS[0][1]*T[1][1] + TS[0][2]*T[1][2];  // Σ'xy
    float e = TS[1][0]*T[1][0] + TS[1][1]*T[1][1] + TS[1][2]*T[1][2];  // Σ'yy

    // Low-pass filter: add 0.3 px² so every splat covers at least about one
    // pixel. Without it, tiny or edge-on splats flicker and alias.
    a += 0.3f;
    e += 0.3f;

    // --- STEP 6: eigen-decomposition of the symmetric 2x2 [[a b] [b e]] ---
    float mid = 0.5f*(a + e);
    float radius = sqrtf(0.25f*(a - e)*(a - e) + b*b);
    float lambda1 = mid + radius;              // larger eigenvalue
    float lambda2 = fmaxf(mid - radius, 0.1f); // smaller one (clamped >= 0)
    Vector2 dir1 = (fabsf(b) > 1e-6f) ? Vector2Normalize(Vector2{ b, lambda1 - a })
                                      : ((a >= e) ? Vector2{ 1, 0 } : Vector2{ 0, 1 });
    Vector2 dir2 = { -dir1.y, dir1.x };        // perpendicular

    // Cap the size so a splat right in front of the lens can't cover the whole
    // screen thousands of times over.
    float r1 = fminf(sqrtf(lambda1), 1024.0f);
    float r2 = fminf(sqrtf(lambda2), 1024.0f);

    out->center = Vector2{ v->cx + f*tx/tz, v->cy - f*ty/tz };
    out->axis1 = Vector2Scale(dir1, r1);
    out->axis2 = Vector2Scale(dir2, r2);
    out->depth = tz;
    return true;
}

// ----------------------------------------------------------------------------
// STEP 5b: View-dependent color from spherical harmonics (SH)
//
// SH do for directions what a Fourier series does for a periodic signal:
// fixed basis functions Y_k(dir), each multiplied by a learned RGB
// coefficient, add up to a color that changes with the viewing direction:
//     color(dir) = 0.5 + Σ_k coeff_k * Y_k(dir)
// Degree 0 is one constant (the base color, f_dc). Degree 1 adds 3 functions
// that vary linearly with the direction, degree 2 adds 5, degree 3 adds 7:
// 1 + 3 + 5 + 7 = 16 coefficients per color channel. This is how trained
// splats get shine and reflections. Constants and signs are the real SH
// basis used by the 3DGS reference renderer.
// `dir` is the unit vector from the camera to the splat center, in the
// coordinate frame the file was trained in.
// ----------------------------------------------------------------------------
static Color EvalSphericalHarmonics(const Vector3 *coeffs, int degree, Vector3 dir, unsigned char alpha)
{
    const float C0 = 0.28209479177387814f;
    const float C1 = 0.4886025119029199f;
    const float C2[5] = { 1.0925484305920792f, -1.0925484305920792f, 0.31539156525252005f,
                          -1.0925484305920792f, 0.5462742152960396f };
    const float C3[7] = { -0.5900435899266435f, 2.890611442640554f, -0.4570457994644658f, 0.3731763325901154f,
                          -0.4570457994644658f, 1.445305721320277f, -0.5900435899266435f };
    float x = dir.x, y = dir.y, z = dir.z;
    float xx = x*x, yy = y*y, zz = z*z;

    // Value of every basis function for this direction.
    float Y[16];
    Y[0] = C0;
    if (degree >= 1) {
        Y[1] = -C1*y;  Y[2] = C1*z;  Y[3] = -C1*x;
    }
    if (degree >= 2) {
        Y[4] = C2[0]*x*y;  Y[5] = C2[1]*y*z;  Y[6] = C2[2]*(2*zz - xx - yy);
        Y[7] = C2[3]*x*z;  Y[8] = C2[4]*(xx - yy);
    }
    if (degree >= 3) {
        Y[9]  = C3[0]*y*(3*xx - yy);       Y[10] = C3[1]*x*y*z;
        Y[11] = C3[2]*y*(4*zz - xx - yy);  Y[12] = C3[3]*z*(2*zz - 3*xx - 3*yy);
        Y[13] = C3[4]*x*(4*zz - xx - yy);  Y[14] = C3[5]*z*(xx - yy);
        Y[15] = C3[6]*x*(xx - 3*yy);
    }

    // Weighted sum, then clamp: only the total has to be a valid color.
    Vector3 rgb = { 0.5f, 0.5f, 0.5f };
    int n = (degree + 1)*(degree + 1);
    for (int k = 0; k < n; k++) rgb = Vector3Add(rgb, Vector3Scale(coeffs[k], Y[k]));
    return Color{ (unsigned char)(Clamp(rgb.x, 0.0f, 1.0f)*255.0f), (unsigned char)(Clamp(rgb.y, 0.0f, 1.0f)*255.0f),
                  (unsigned char)(Clamp(rgb.z, 0.0f, 1.0f)*255.0f), alpha };
}

// ----------------------------------------------------------------------------
// STEP 7: Sort back to front
//
// Splats are semi-transparent, so they must be blended in depth order:
// farthest first, nearest last ("painter's algorithm"). The 3DGS paper blends
// front to back. Back to front with ordinary alpha blending gives the same
// image and is what the GPU's blend unit does natively.
//
// A counting sort is used instead of std::sort because it is O(N) and stays
// fast for millions of splats. Depth is quantized to 16 bits (65536 buckets);
// splats in the same bucket keep their original relative order.
// ----------------------------------------------------------------------------
#define SORT_BUCKETS 65536

static void SortBackToFront(const ProjectedSplat *proj, const int *visible, int visibleCount,
                            int *sorted, uint16_t *keys, unsigned int *bucketStart)
{
    if (visibleCount == 0) return;

    float minDepth = INFINITY, maxDepth = -INFINITY;
    for (int i = 0; i < visibleCount; i++) {
        float d = proj[visible[i]].depth;
        minDepth = fminf(minDepth, d);
        maxDepth = fmaxf(maxDepth, d);
    }
    float range = fmaxf(maxDepth - minDepth, 1e-6f);

    // key 0 = farthest, key 65535 = nearest, so ascending key = back to front
    memset(bucketStart, 0, sizeof(unsigned int)*SORT_BUCKETS);
    for (int i = 0; i < visibleCount; i++) {
        keys[i] = (uint16_t)((maxDepth - proj[visible[i]].depth)/range*(SORT_BUCKETS - 1));
        bucketStart[keys[i]]++;                               // 1) count per bucket
    }
    unsigned int sum = 0;
    for (int k = 0; k < SORT_BUCKETS; k++) {                  // 2) prefix sum -> where each bucket starts
        unsigned int c = bucketStart[k];
        bucketStart[k] = sum;
        sum += c;
    }
    for (int i = 0; i < visibleCount; i++) sorted[bucketStart[keys[i]]++] = visible[i];  // 3) scatter
}

// ----------------------------------------------------------------------------
// STEP 8: The fragment shader
//
// Each splat is drawn as a rectangle spanning ±3 sigma along both ellipse
// axes. The texture coordinates of the corners are (±3, ±3), so inside the
// rectangle the interpolated texcoord is "where am I, measured in sigmas
// along each axis". The Gaussian is then simply  exp(-0.5 * |p|²).
// Final alpha = splat opacity * Gaussian falloff. The GPU blends it over what
// is already on screen:  out = color*alpha + previous*(1 - alpha).
// The default raylib vertex shader passes fragTexCoord and fragColor through.
// ----------------------------------------------------------------------------
static const char *SPLAT_FS =
    "#version 330\n"
    "in vec2 fragTexCoord;   // position inside the splat, in sigmas\n"
    "in vec4 fragColor;      // rgb = splat color, a = opacity\n"
    "uniform int mode;       // 0 gaussian, 1 ellipse outline, 2 solid dot\n"
    "out vec4 finalColor;\n"
    "void main() {\n"
    "    float r2 = dot(fragTexCoord, fragTexCoord);\n"
    "    if (r2 > 9.0) discard;   // outside 3 sigma: cut off\n"
    "    float alpha;\n"
    "    if (mode == 0) {\n"
    "        alpha = fragColor.a * exp(-0.5 * r2);\n"
    "    } else if (mode == 1) {\n"
    "        float r = sqrt(r2);\n"
    "        float px = fwidth(r);   // how many sigmas one pixel spans here\n"
    "        alpha = 1.0 - smoothstep(0.0, 1.5 * px, abs(r - 2.0));  // ring at 2 sigma\n"
    "    } else {\n"
    "        alpha = 1.0;\n"
    "    }\n"
    "    if (alpha < 1.0/255.0) discard;\n"
    "    finalColor = vec4(fragColor.rgb, alpha);\n"
    "}\n";

static void DrawSplats(const ProjectedSplat *proj, const int *order, int count,
                       ViewMode mode, Shader shader, int modeLoc)
{
    int shaderMode = (mode == VIEW_GAUSSIANS) ? 0 : (mode == VIEW_ELLIPSES) ? 1 : 2;
    SetShaderValue(shader, modeLoc, &shaderMode, SHADER_UNIFORM_INT);

    BeginShaderMode(shader);
    rlBegin(RL_QUADS);
    for (int i = 0; i < count; i++) {
        int idx = order[i];
        const ProjectedSplat *p = &proj[idx];
        Color c = p->color;

        // Rectangle corners: center ± 3*axis1 ± 3*axis2.
        // In "centers" mode, draw a fixed 2.5 px dot instead of the ellipse.
        Vector2 a1 = Vector2Scale(p->axis1, 3.0f);
        Vector2 a2 = Vector2Scale(p->axis2, 3.0f);
        if (mode == VIEW_CENTERS) {
            a1 = Vector2{ 2.5f, 0.0f };
            a2 = Vector2{ 0.0f, 2.5f };
            c.a = 255;
        }

        // raylib culls back faces, so the corners must go counter-clockwise on
        // screen. axis2 is always axis1 turned by +90 degrees, so this one order
        // works for every splat.
        rlColor4ub(c.r, c.g, c.b, c.a);
        rlTexCoord2f(-3, -3); rlVertex2f(p->center.x - a1.x - a2.x, p->center.y - a1.y - a2.y);
        rlTexCoord2f(-3,  3); rlVertex2f(p->center.x - a1.x + a2.x, p->center.y - a1.y + a2.y);
        rlTexCoord2f( 3,  3); rlVertex2f(p->center.x + a1.x + a2.x, p->center.y + a1.y + a2.y);
        rlTexCoord2f( 3, -3); rlVertex2f(p->center.x + a1.x - a2.x, p->center.y + a1.y - a2.y);
    }
    rlEnd();
    EndShaderMode();
}

// ----------------------------------------------------------------------------
// Helpers to aim the camera at a loaded scene: the median is robust to the
// far-away background splats that real captures always contain.
// ----------------------------------------------------------------------------
static float Median(std::vector<float> &values)
{
    std::nth_element(values.begin(), values.begin() + values.size()/2, values.end());
    return values[values.size()/2];
}

static void FrameScene(const std::vector<Splat> &splats, OrbitCamera *cam)
{
    std::vector<float> tmp(splats.size());
    for (size_t i = 0; i < splats.size(); i++) tmp[i] = splats[i].position.x;
    cam->target.x = Median(tmp);
    for (size_t i = 0; i < splats.size(); i++) tmp[i] = splats[i].position.y;
    cam->target.y = Median(tmp);
    for (size_t i = 0; i < splats.size(); i++) tmp[i] = splats[i].position.z;
    cam->target.z = Median(tmp);
    for (size_t i = 0; i < splats.size(); i++) tmp[i] = Vector3Distance(splats[i].position, cam->target);
    cam->distance = fmaxf(3.0f*Median(tmp), 0.5f);
}

// ----------------------------------------------------------------------------
// Main: load, then each frame  input -> project -> sort -> draw
// ----------------------------------------------------------------------------
int main(int argc, char **argv)
{
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_VSYNC_HINT);
    InitWindow(1280, 800, "Gaussian splat viewer");

    // STEP 2: get splats
    bool fromFile = (argc > 1);
    std::vector<Splat> splats;
    SphericalHarmonics sh;
    if (!fromFile) splats = BuildDemoScene();
    else if (IsFileExtension(argv[1], ".splat")) splats = LoadSplatsFromSplatFile(argv[1]);
    else splats = LoadSplatsFromPly(argv[1], &sh);
    if (splats.empty()) { CloseWindow(); return 1; }
    int count = (int)splats.size();

    // STEP 3: covariance once per splat
    for (Splat &s : splats) ComputeCovariance3D(&s);
    bool flipped = false;
    if (fromFile) { FlipSceneUpsideDown(splats); flipped = true; }

    OrbitCamera cam = { .target = { 0.0f, 1.0f, 0.0f }, .distance = 6.5f, .yaw = 0.6f, .pitch = 0.35f, .fovy = 60.0f*DEG2RAD };
    if (fromFile) FrameScene(splats, &cam);
    OrbitCamera initialCam = cam;

    // Per-frame working memory, allocated once.
    std::vector<ProjectedSplat> proj(count);
    std::vector<int> visible(count), sorted(count);
    std::vector<uint16_t> keys(count);
    std::vector<unsigned int> buckets(SORT_BUCKETS);

    Shader shader = LoadShaderFromMemory(NULL, SPLAT_FS);
    int modeLoc = GetShaderLocation(shader, "mode");

    ViewMode mode = VIEW_GAUSSIANS;
    bool sortEnabled = true;
    bool shEnabled = (sh.degree > 0);
    float scaleMod = 1.0f;

    while (!WindowShouldClose()) {
        // ---- input ----
        Vector2 md = GetMouseDelta();
        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
            cam.yaw -= md.x*0.005f;
            cam.pitch = Clamp(cam.pitch + md.y*0.005f, -1.55f, 1.55f);
        }
        ViewParams view = MakeView(&cam, GetScreenWidth(), GetScreenHeight());
        if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT) || IsMouseButtonDown(MOUSE_BUTTON_MIDDLE)) {
            // one pixel at the target's depth = distance / focal world units
            float k = cam.distance/view.focal;
            cam.target = Vector3Add(cam.target, Vector3Scale(view.right, -md.x*k));
            cam.target = Vector3Add(cam.target, Vector3Scale(view.up, md.y*k));
        }
        cam.distance *= powf(0.9f, GetMouseWheelMove());
        if (IsKeyPressed(KEY_ONE)) mode = VIEW_GAUSSIANS;
        if (IsKeyPressed(KEY_TWO)) mode = VIEW_ELLIPSES;
        if (IsKeyPressed(KEY_THREE)) mode = VIEW_CENTERS;
        if (IsKeyPressed(KEY_S)) sortEnabled = !sortEnabled;
        if (IsKeyPressed(KEY_H) && sh.degree > 0) shEnabled = !shEnabled;
        if (IsKeyDown(KEY_EQUAL) || IsKeyDown(KEY_KP_ADD)) scaleMod = fminf(scaleMod*1.02f, 4.0f);
        if (IsKeyDown(KEY_MINUS) || IsKeyDown(KEY_KP_SUBTRACT)) scaleMod = fmaxf(scaleMod/1.02f, 0.01f);
        if (IsKeyPressed(KEY_F)) { FlipSceneUpsideDown(splats); flipped = !flipped; }
        if (IsKeyPressed(KEY_R)) { cam = initialCam; scaleMod = 1.0f; }
        view = MakeView(&cam, GetScreenWidth(), GetScreenHeight());

        // ---- STEP 5 + 6: project every splat, STEP 5b: its color from this view ----
        // The SH were trained in the file's frame. If the scene was flipped, turn
        // the viewing direction back into that frame before evaluating them.
        Quaternion toFileFrame = flipped ? QuaternionInvert(SceneFlip()) : QuaternionIdentity();
        double t0 = GetTime();
        int visibleCount = 0;
        for (int i = 0; i < count; i++) {
            if (!ProjectSplat(&splats[i], &view, scaleMod, &proj[i])) continue;
            proj[i].color = splats[i].color;
            if (shEnabled) {
                Vector3 dir = Vector3Normalize(Vector3Subtract(splats[i].position, view.eye));
                dir = Vector3RotateByQuaternion(dir, toFileFrame);
                proj[i].color = EvalSphericalHarmonics(&sh.coeffs[(size_t)i*sh.perSplat], sh.degree, dir, splats[i].color.a);
            }
            visible[visibleCount++] = i;
        }

        // ---- STEP 7: sort ----
        double t1 = GetTime();
        const int *order = visible.data();  // unsorted = file order
        if (sortEnabled) {
            SortBackToFront(proj.data(), visible.data(), visibleCount, sorted.data(), keys.data(), buckets.data());
            order = sorted.data();
        }

        // ---- STEP 8: draw ----
        double t2 = GetTime();
        BeginDrawing();
        ClearBackground(Color{ 20, 20, 26, 255 });
        DrawSplats(proj.data(), order, visibleCount, mode, shader, modeLoc);
        double t3 = GetTime();

        // ---- HUD ----
        static const char *modeNames[] = { "Gaussians", "Ellipses (2 sigma)", "Centers" };
        DrawRectangle(8, 8, 380, 242, Fade(BLACK, 0.6f));
        DrawText(TextFormat("%d FPS   %d / %d splats visible", GetFPS(), visibleCount, count), 16, 16, 18, RAYWHITE);
        DrawText(TextFormat("project %.1f ms  sort %.1f ms  draw %.1f ms",
                            (t1 - t0)*1000.0, (t2 - t1)*1000.0, (t3 - t2)*1000.0), 16, 38, 16, LIGHTGRAY);
        DrawText(TextFormat("[1/2/3] view:   %s", modeNames[mode]), 16, 66, 16, RAYWHITE);
        DrawText(TextFormat("[S] sort:       %s", sortEnabled ? "on (back to front)" : "OFF (file order)"), 16, 86, 16,
                 sortEnabled ? RAYWHITE : ORANGE);
        DrawText((sh.degree == 0) ? "[H] SH color:   none in file"
                 : shEnabled      ? TextFormat("[H] SH color:   on (degree %d)", sh.degree)
                                  : "[H] SH color:   OFF (base color)", 16, 106, 16,
                 (sh.degree > 0 && !shEnabled) ? ORANGE : RAYWHITE);
        DrawText(TextFormat("[+/-] scale:    %.2fx", scaleMod), 16, 126, 16, RAYWHITE);
        DrawText(TextFormat("[F] flip Y:     %s", flipped ? "on" : "off"), 16, 146, 16, RAYWHITE);
        DrawText("[R] reset camera", 16, 166, 16, RAYWHITE);
        DrawText("LMB orbit  RMB/MMB pan  wheel zoom", 16, 194, 16, LIGHTGRAY);
        DrawText("F12 screenshot  Esc quit", 16, 214, 16, LIGHTGRAY);
        EndDrawing();
    }

    UnloadShader(shader);
    CloseWindow();
    return 0;
}
