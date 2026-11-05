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
//    STEP 9  The same on the GPU: splats uploaded once, the vertex shader
//            runs STEPs 5, 5b and 6
//    STEP 10 Cull and sort on the GPU too, with compute shaders (OpenGL 4.3):
//            the CPU does no per-splat work at all. This is the default
//            renderer where it is available; G switches between the three.
//
//  Usage:  ./splat_viewer              -> procedural demo scene
//          ./splat_viewer scene.ply    -> standard 3DGS output .ply (read with happly)
//          ./splat_viewer scene.splat  -> compact .splat (web viewer format)
//  A .ply or .splat dropped on the window replaces the scene.
// ============================================================================

#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"
#include "external/glad.h"  // raylib's OpenGL loader: for the few calls STEP 10 needs that rlgl does not wrap

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
// A .ply without these properties is a plain point cloud (STEP 2d).
// ----------------------------------------------------------------------------
static float Sigmoid(float x) { return 1.0f/(1.0f + expf(-x)); }

static std::vector<Splat> SplatsFromPointCloud(happly::Element &vertex);  // STEP 2d, below

static std::vector<Splat> LoadSplatsFromPly(const char *path, SphericalHarmonics *sh, bool *pointCloud)
{
    std::vector<Splat> splats;
    try {
        happly::PLYData ply(path);  // reads the whole file; throws std::runtime_error on problems
        happly::Element &vertex = ply.getElement("vertex");
        auto column = [&](const std::string &name) { return vertex.getProperty<float>(name); };

        *pointCloud = !(vertex.hasProperty("f_dc_0") && vertex.hasProperty("opacity") &&
                        vertex.hasProperty("scale_0") && vertex.hasProperty("rot_0"));
        if (*pointCloud) {
            splats = SplatsFromPointCloud(vertex);
            TraceLog(LOG_INFO, "Loaded %d points from %s (a point cloud: no Gaussian properties)", (int)splats.size(),
                     path);
            return splats;
        }

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
        TraceLog(LOG_ERROR, "A point cloud .ply needs x y z; a Gaussian Splatting .ply also f_dc_0..2, opacity, "
                            "scale_0..2, rot_0..3");
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
// STEP 2d: A point cloud .ply: laser scans, photogrammetry, COLMAP's sparse
// points, whatever CloudCompare exports
//
// It has positions and maybe colors, but none of the Gaussian properties of
// STEP 2a. Every point becomes a round (the same sigma in every direction),
// opaque splat about as wide as the gaps between points, so in the Gaussians
// view (1) the points blend into closed surfaces. Point clouds open in the
// centers view (3), though: one dot per point, as point cloud viewers draw them.
// Color: red/green/blue if the file has them, else intensity as gray, else a
// color ramp over height (Z).
// Scans are often stored in map coordinates (UTM: thousands of kilometers),
// where a 32-bit float only resolves about 0.5 m. So positions are read as
// doubles and the cloud is moved to have its center at the origin.
// ----------------------------------------------------------------------------

// Any numeric property, as doubles. happly only converts within one family
// (uchar to ushort, float to double), and point clouds use all of them.
static std::vector<double> NumericColumn(happly::Element &vertex, const std::string &name)
{
    if (vertex.hasPropertyType<float>(name) || vertex.hasPropertyType<double>(name)) {
        return vertex.getProperty<double>(name);
    }
    std::vector<double> values;
    values.reserve(vertex.count);
    if (vertex.hasPropertyType<uint8_t>(name) || vertex.hasPropertyType<uint16_t>(name) ||
        vertex.hasPropertyType<uint32_t>(name)) {
        for (uint32_t v : vertex.getProperty<uint32_t>(name)) values.push_back(v);
    } else {
        for (int32_t v : vertex.getProperty<int32_t>(name)) values.push_back(v);
    }
    return values;
}

// Factor that brings a color channel to 0..255: files store bytes (0..255),
// 16-bit values (0..65535) or floats (0..1).
static float ColorScale(happly::Element &vertex, const std::string &name)
{
    if (vertex.hasPropertyType<uint16_t>(name)) return 255.0f/65535.0f;
    if (vertex.hasPropertyType<float>(name) || vertex.hasPropertyType<double>(name)) return 255.0f;
    return 1.0f;
}

// The value a fraction p of the values lie below, from up to 100k evenly
// spread samples (exact enough for a color range, and fast).
static double Percentile(const std::vector<double> &values, double p)
{
    size_t step = std::max<size_t>(1, values.size()/100000);
    std::vector<double> sample;
    for (size_t i = 0; i < values.size(); i += step) sample.push_back(values[i]);
    size_t k = std::min(sample.size() - 1, (size_t)(p*sample.size()));
    std::nth_element(sample.begin(), sample.begin() + k, sample.end());
    return sample[k];
}

// t from 0 to 1: blue, cyan, green, yellow, red
static Color RampColor(float t)
{
    const Vector3 stops[5] = { { 0.15f, 0.25f, 0.90f }, { 0.10f, 0.75f, 0.90f }, { 0.20f, 0.80f, 0.30f },
                               { 0.95f, 0.85f, 0.20f }, { 0.90f, 0.20f, 0.15f } };
    t = Clamp(t, 0.0f, 1.0f)*4.0f;
    int i = std::min((int)t, 3);
    Vector3 c = Vector3Lerp(stops[i], stops[i + 1], t - i);
    return Color{ (unsigned char)(c.x*255.0f), (unsigned char)(c.y*255.0f), (unsigned char)(c.z*255.0f), 255 };
}

// Typical distance from a point to its nearest neighbor. An exact search is
// slow for millions of points, so 500 sample points are each compared with a
// subset of up to 20000 points spread over the file. In a subset of m of the n
// points the gaps are sqrt(n/m) times wider when the points lie on surfaces
// (scans, photogrammetry), so the median gap is scaled back by sqrt(m/n).
static float EstimatePointSpacing(const std::vector<Splat> &points)
{
    size_t n = points.size(), m = std::min<size_t>(n, 20000), queries = std::min<size_t>(n, 500);
    std::vector<float> gaps;
    for (size_t q = 0; q < queries; q++) {
        size_t qi = q*n/queries;
        float best = INFINITY;
        for (size_t j = 0; j < m; j++) {
            float d2 = Vector3DistanceSqr(points[qi].position, points[j*n/m].position);
            if (d2 > 0.0f && d2 < best) best = d2;  // 0: the point itself, or a duplicate
        }
        if (best < INFINITY) gaps.push_back(sqrtf(best));
    }
    if (gaps.empty()) return 0.01f;
    std::nth_element(gaps.begin(), gaps.begin() + gaps.size()/2, gaps.end());
    return gaps[gaps.size()/2]*sqrtf((float)m/n);
}

static std::vector<Splat> SplatsFromPointCloud(happly::Element &vertex)
{
    size_t count = vertex.count;
    std::vector<Splat> points(count);
    if (count == 0) return points;

    // Positions, one axis at a time, moved by the center of their bounding box
    const char *axes[3] = { "x", "y", "z" };
    double center[3];
    std::vector<double> height;  // Z as stored, for the color ramp
    for (int a = 0; a < 3; a++) {
        std::vector<double> v = NumericColumn(vertex, axes[a]);
        auto [lo, hi] = std::minmax_element(v.begin(), v.end());
        center[a] = 0.5*(*lo + *hi);
        for (size_t i = 0; i < count; i++) {
            float value = (float)(v[i] - center[a]);
            Vector3 &p = points[i].position;
            if (a == 0) p.x = value; else if (a == 1) p.y = value; else p.z = value;
        }
        if (a == 2) height = std::move(v);
    }
    TraceLog(LOG_INFO, "Point cloud: moved by (%.3f, %.3f, %.3f) to put its center at the origin", -center[0],
             -center[1], -center[2]);

    // Colors
    std::string intensity;  // CloudCompare calls it scalar_Intensity
    for (const std::string &name : vertex.getPropertyNames()) {
        std::string lower = name;
        for (char &c : lower) c = (char)tolower(c);
        if (lower == "intensity" || lower == "scalar_intensity") intensity = name;
    }
    bool rgb = vertex.hasProperty("red") && vertex.hasProperty("green") && vertex.hasProperty("blue");
    if (rgb) {
        const char *channels[3] = { "red", "green", "blue" };
        for (int c = 0; c < 3; c++) {
            std::vector<double> v = NumericColumn(vertex, channels[c]);
            float scale = ColorScale(vertex, channels[c]);
            for (size_t i = 0; i < count; i++) {
                unsigned char value = (unsigned char)Clamp((float)v[i]*scale + 0.5f, 0.0f, 255.0f);
                Color &col = points[i].color;
                if (c == 0) col.r = value; else if (c == 1) col.g = value; else col.b = value;
            }
        }
    } else if (!intensity.empty()) {
        std::vector<double> v = NumericColumn(vertex, intensity);
        double lo = Percentile(v, 0.01), hi = Percentile(v, 0.99);  // ignore the brightest and darkest 1%
        for (size_t i = 0; i < count; i++) {
            unsigned char gray = (unsigned char)(Clamp((float)((v[i] - lo)/fmax(hi - lo, 1e-9)), 0.0f, 1.0f)*255.0f);
            points[i].color = Color{ gray, gray, gray, 255 };
        }
    } else {
        double lo = Percentile(height, 0.02), hi = Percentile(height, 0.98);
        for (size_t i = 0; i < count; i++) points[i].color = RampColor((float)((height[i] - lo)/fmax(hi - lo, 1e-9)));
    }
    TraceLog(LOG_INFO, "Point cloud: colors from %s", rgb ? "red/green/blue"
                                                      : !intensity.empty() ? intensity.c_str() : "height (Z)");

    // Round, opaque splats; sigma half the gap between points
    float sigma = 0.5f*EstimatePointSpacing(points);
    TraceLog(LOG_INFO, "Point cloud: points about %.4f apart, splat sigma %.4f", 2.0f*sigma, sigma);
    for (Splat &p : points) {
        p.scale = Vector3{ sigma, sigma, sigma };
        p.rotation = QuaternionIdentity();
        p.color.a = 255;
    }
    return points;
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

// Which way is up in the file. raylib's orbit camera keeps +Y up, so the
// scene is turned to bring the file's up direction there. 3DGS scenes usually
// come from COLMAP, where +Y points down; scans and most point clouds have +Z
// up. F cycles through these.
enum UpAxis { UP_MINUS_Y = 0, UP_PLUS_Z, UP_PLUS_Y, UP_AXIS_COUNT };
static const char *upAxisNames[] = { "-Y (COLMAP)", "+Z (scans)", "+Y" };
static_assert(sizeof(upAxisNames)/sizeof(upAxisNames[0]) == UP_AXIS_COUNT, "one name per up axis");

// The turn that brings the file's up direction to +Y
static Quaternion UpAxisTurn(UpAxis up)
{
    if (up == UP_MINUS_Y) return QuaternionFromAxisAngle(Vector3{ 1, 0, 0 }, PI);     // 180 degrees about X
    if (up == UP_PLUS_Z) return QuaternionFromAxisAngle(Vector3{ 1, 0, 0 }, -PI/2);  // +Z to +Y
    return QuaternionIdentity();                                                     // UP_PLUS_Y
}

static void TurnScene(std::vector<Splat> &splats, Quaternion turn)
{
    for (Splat &s : splats) {
        s.position = Vector3RotateByQuaternion(s.position, turn);
        s.rotation = QuaternionNormalize(QuaternionMultiply(turn, s.rotation));
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
// Camera-space position of a splat center (a dot product with each camera
// axis), and whether the splat is worth drawing. Shared by both CPU paths;
// the GPU cull (STEP 10) makes the same test with the same two numbers.
#define NEAR_PLANE 0.2f   // closer than this, or behind the camera: culled
#define CULL_MARGIN 1.3f  // centers up to 1.3x the screen out still count, so big splats
                          // just off screen get a chance to poke in
static bool ToCameraSpace(Vector3 p, const ViewParams *v, float *tx, float *ty, float *tz)
{
    Vector3 d = Vector3Subtract(p, v->eye);
    *tx = Vector3DotProduct(d, v->right);
    *ty = Vector3DotProduct(d, v->up);
    *tz = Vector3DotProduct(d, v->forward);
    if (*tz < NEAR_PLANE) return false;

    // Skip splats whose center is far outside the view.
    float limX = CULL_MARGIN*v->tanHalfFovX, limY = CULL_MARGIN*v->tanHalfFovY;
    return fabsf(*tx / *tz) <= limX && fabsf(*ty / *tz) <= limY;
}

static bool ProjectSplat(const Splat *s, const ViewParams *v, float scaleMod, ProjectedSplat *out)
{
    // --- camera-space position, cull ---
    float tx, ty, tz;
    if (!ToCameraSpace(s->position, v, &tx, &ty, &tz)) return false;

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
// std::sort would be the simplest (the main branch uses it), but it is
// O(N log N) and every comparison fetches a depth from a random place in
// memory: about 40 ms for the 580k visible splats of the train scene.
// This branch uses a counting sort instead, about 4 ms for the same scene:
// depth is quantized to 16 bits (65536 buckets), then three linear passes
// put every splat in place. Splats in the same bucket keep their file order,
// which is invisible on screen.
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

// dotSize: diameter in pixels of the dots of the centers view
static void DrawSplats(const ProjectedSplat *proj, const int *order, int count,
                       ViewMode mode, float dotSize, Shader shader, int modeLoc)
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
        // In "centers" mode, draw a round dot of dotSize pixels instead of the ellipse.
        Vector2 a1 = Vector2Scale(p->axis1, 3.0f);
        Vector2 a2 = Vector2Scale(p->axis2, 3.0f);
        if (mode == VIEW_CENTERS) {
            a1 = Vector2{ 0.5f*dotSize, 0.0f };
            a2 = Vector2{ 0.0f, 0.5f*dotSize };
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
// STEP 9: The same pipeline on the GPU (G switches renderers)
//
// The CPU renderer above runs STEPs 5, 5b and 6 for every splat on the CPU
// and sends 4 vertices (144 bytes) per splat to the GPU every frame, through
// ~12M raylib function calls. The GPU renderer instead:
//   - uploads all splat data ONCE, into four textures. In each texture every
//     splat takes the same number of texels, one splat after the other,
//     wrapping into the next row: texel i*texelsPerSplat + k is splat i's k-th.
//         geometry  RGBA32F, 3 texels per splat:
//                     position xyz, opacity | covariance Σ xx xy xz yy | yz zz
//         shRed     RGBA16F (half floats), 1 to 4 texels per splat: the SH
//         shGreen   coefficients of one color channel, 4 per texel, f_dc
//         shBlue    first (the .ply also stores them channel by channel)
//     Half floats are precise enough for color and halve the memory: with SH
//     degree 3 a splat takes 48 + 3*32 = 144 bytes. One texture per channel
//     keeps each texture small enough for the GPU's size limit.
//   - per frame lets the CPU only cull, sort, and upload the draw order: one
//     int per visible splat (its index), 4 bytes instead of 144
//   - draws everything with ONE instanced draw call: the same 6-vertex quad,
//     once per splat. The vertex shader reads "its" splat from the textures
//     and runs STEPs 5, 5b and 6 itself. The fragment shader is unchanged.
// ----------------------------------------------------------------------------
static const char *SPLAT_VS = R"(#version 330
layout(location = 0) in vec2 corner;       // quad corner (+-1, +-1), the same 6 for every splat
layout(location = 1) in vec4 indexBytes;   // per instance: which splat to draw (in sorted order),
                                           // as the 4 bytes of an int, lowest first

uniform sampler2D geometry;                // the splat textures described above
uniform sampler2D shRed, shGreen, shBlue;
uniform int shTexels;                      // texels per splat in each SH texture
uniform int textureWidth;
uniform vec3 eye, camRight, camUp, camForward;
uniform float focal;
uniform vec2 screenSize;
uniform float scaleMod;
uniform int shDegree;                      // 0 = base color only
uniform mat4 toFileFrame;                  // world direction -> frame the SH were trained in
uniform int mode;                          // 0 gaussian, 1 ellipse outline, 2 solid dot
uniform float dotSize;                     // dot diameter in pixels (mode 2)

out vec2 fragTexCoord;                     // same outputs as raylib's default vertex shader,
out vec4 fragColor;                        // so the fragment shader is shared with the CPU path

// Texel k of splat s, in a texture that holds n texels per splat.
vec4 Fetch(sampler2D tex, int n, int s, int k)
{
    int i = s*n + k;
    return texelFetch(tex, ivec2(i % textureWidth, i / textureWidth), 0);
}

// STEP 5b: the same basis functions as EvalSphericalHarmonics(), grouped 4 to
// a vec4 like the coefficients in the SH textures. The unused ones stay 0.
vec3 EvalSH(int s, vec3 dir)
{
    float x = dir.x, y = dir.y, z = dir.z;
    float xx = x*x, yy = y*y, zz = z*z;
    vec4 Y[4] = vec4[4](vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0));  // Y[0].x is Y_0, Y[3].w is Y_15
    Y[0].x = 0.28209479177387814;
    if (shDegree >= 1) {
        Y[0].yzw = vec3(-0.4886025119029199*y, 0.4886025119029199*z, -0.4886025119029199*x);
    }
    if (shDegree >= 2) {
        Y[1] = vec4(1.0925484305920792*x*y, -1.0925484305920792*y*z, 0.31539156525252005*(2.0*zz - xx - yy),
                    -1.0925484305920792*x*z);
        Y[2].x = 0.5462742152960396*(xx - yy);
    }
    if (shDegree >= 3) {
        Y[2].yzw = vec3(-0.5900435899266435*y*(3.0*xx - yy), 2.890611442640554*x*y*z,
                        -0.4570457994644658*y*(4.0*zz - xx - yy));
        Y[3] = vec4(0.3731763325901154*z*(2.0*zz - 3.0*xx - 3.0*yy), -0.4570457994644658*x*(4.0*zz - xx - yy),
                    1.445305721320277*z*(xx - yy), -0.5900435899266435*x*(xx - 3.0*yy));
    }
    // 0.5 + sum of coefficient * basis value: one dot product per texel and channel
    vec3 rgb = vec3(0.5);
    int texels = ((shDegree + 1)*(shDegree + 1) + 3)/4;  // the texels holding the coefficients in use
    for (int t = 0; t < texels; t++) {
        rgb += vec3(dot(Fetch(shRed, shTexels, s, t), Y[t]), dot(Fetch(shGreen, shTexels, s, t), Y[t]),
                    dot(Fetch(shBlue, shTexels, s, t), Y[t]));
    }
    return clamp(rgb, 0.0, 1.0);
}

void main()
{
    ivec4 bytes = ivec4(indexBytes);
    int s = bytes.x | (bytes.y << 8) | (bytes.z << 16) | (bytes.w << 24);
    vec4 t0 = Fetch(geometry, 3, s, 0), t1 = Fetch(geometry, 3, s, 1), t2 = Fetch(geometry, 3, s, 2);

    // STEP 5: camera space, T = J W written out row by row, Sigma' = T Sigma T^T
    vec3 d = t0.xyz - eye;
    float tx = dot(d, camRight), ty = dot(d, camUp), tz = dot(d, camForward);
    vec3 T0 = (focal/tz)*camRight - (focal*tx/(tz*tz))*camForward;
    vec3 T1 = -(focal/tz)*camUp + (focal*ty/(tz*tz))*camForward;
    mat3 Sigma = mat3(t1.x, t1.y, t1.z,
                      t1.y, t1.w, t2.x,
                      t1.z, t2.x, t2.y)*(scaleMod*scaleMod);
    float a = dot(T0, Sigma*T0) + 0.3;  // + the same low-pass filter as on the CPU
    float b = dot(T0, Sigma*T1);
    float e = dot(T1, Sigma*T1) + 0.3;

    // STEP 6: ellipse axes from the eigen-decomposition, rectangle at +-3 sigma
    float mid = 0.5*(a + e);
    float radius = sqrt(0.25*(a - e)*(a - e) + b*b);
    float lambda1 = mid + radius;
    float lambda2 = max(mid - radius, 0.1);
    vec2 dir1 = (abs(b) > 1e-6) ? normalize(vec2(b, lambda1 - a)) : ((a >= e) ? vec2(1.0, 0.0) : vec2(0.0, 1.0));
    vec2 dir2 = vec2(-dir1.y, dir1.x);
    vec2 a1 = 3.0*dir1*min(sqrt(lambda1), 1024.0);
    vec2 a2 = 3.0*dir2*min(sqrt(lambda2), 1024.0);

    // STEP 5b: color seen from this camera
    vec4 color = vec4(EvalSH(s, mat3(toFileFrame)*normalize(d)), t0.w);
    if (mode == 2) { a1 = vec2(0.5*dotSize, 0.0); a2 = vec2(0.0, 0.5*dotSize); color.a = 1.0; }

    vec2 center = vec2(0.5*screenSize.x + focal*tx/tz, 0.5*screenSize.y - focal*ty/tz);
    vec2 pixel = center + corner.x*a1 + corner.y*a2;
    fragTexCoord = 3.0*corner;
    fragColor = color;
    // pixels -> clip space; y flips because pixel rows go down and clip y goes up
    gl_Position = vec4(2.0*pixel.x/screenSize.x - 1.0, 1.0 - 2.0*pixel.y/screenSize.y, 0.0, 1.0);
}
)";

// The splat textures are TEXTURE_WIDTH texels wide and as many rows tall as
// they need, at most TEXTURE_WIDTH: 16384 x 16384 is supported by every
// desktop GPU of the last decade. That is room for 16384²/3 = 89M splats of
// geometry, and for 67M splats with SH degree 3 (4 texels per channel).
#define TEXTURE_WIDTH 16384
#define GEOMETRY_TEXELS 3

struct GpuRenderer {
    Shader shader = {};
    unsigned int geometryTexture = 0;
    unsigned int shTextures[3] = {};   // red, green, blue
    int shTexels = 0;                  // texels per splat in each SH texture
    int geometryRows = 0, shRows = 0;  // texture heights
    unsigned int vao = 0, cornerVbo = 0, orderVbo = 0;
    int locEye, locRight, locUp, locForward, locFocal, locScreen, locScale, locShDegree, locToFile, locMode, locDotSize;
};

// Rows of TEXTURE_WIDTH texels needed to give `count` splats `texelsPerSplat` texels each.
static int TextureRows(size_t count, int texelsPerSplat)
{
    return (int)((count*texelsPerSplat + TEXTURE_WIDTH - 1)/TEXTURE_WIDTH);
}

// 32-bit float -> 16-bit half float (1 sign, 5 exponent, 10 mantissa bits),
// rounded to nearest. Values beyond the half range are clamped to ±65504;
// values below 2^-14 (6e-5) and NaN become 0.
static uint16_t FloatToHalf(float f)
{
    if (!(fabsf(f) >= 6.103515625e-5f)) return 0;
    f = Clamp(f, -65504.0f, 65504.0f);
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000;
    uint32_t exponent = ((bits >> 23) & 0xff) - 127 + 15;   // re-bias the exponent from 8 to 5 bits
    uint32_t mantissa = (bits & 0x7fffff) + 0x1000;         // + half of the 13 dropped bits = round
    if (mantissa & 0x800000) { mantissa = 0; exponent++; }  // the rounding carried into the exponent
    return (uint16_t)(sign | (exponent << 10) | (mantissa >> 13));
}

// Geometry texture: GEOMETRY_TEXELS RGBA32F texels per splat (layout above).
static std::vector<float> PackGeometryTexels(const std::vector<Splat> &splats, int rows)
{
    std::vector<float> texels((size_t)TEXTURE_WIDTH*rows*4, 0.0f);
    for (size_t i = 0; i < splats.size(); i++) {
        const Splat &s = splats[i];
        float *t = &texels[i*GEOMETRY_TEXELS*4];
        t[0] = s.position.x; t[1] = s.position.y; t[2] = s.position.z; t[3] = s.color.a/255.0f;
        memcpy(t + 4, s.cov, sizeof(s.cov));  // texels 1 and 2 (the last 2 floats stay 0)
    }
    return texels;
}

// SH texture of one color channel (0 red, 1 green, 2 blue): shTexels RGBA16F
// texels per splat, 4 coefficients in each. Unused slots stay 0.
static std::vector<uint16_t> PackShTexels(const std::vector<Splat> &splats, const SphericalHarmonics &sh,
                                          int channel, int shTexels, int rows)
{
    const float SH_C0 = 0.28209479177387814f;
    std::vector<uint16_t> texels((size_t)TEXTURE_WIDTH*rows*4, 0);  // all bits 0 is the half float 0
    for (size_t i = 0; i < splats.size(); i++) {
        uint16_t *t = &texels[i*shTexels*4];
        if (sh.degree > 0) {
            for (int k = 0; k < sh.perSplat; k++) {
                Vector3 c = sh.coeffs[i*sh.perSplat + k];
                t[k] = FloatToHalf((channel == 0) ? c.x : (channel == 1) ? c.y : c.z);
            }
        } else {
            // No SH in the file: store the base color as coefficient 0 (color = 0.5 + C0*c0).
            Color c = splats[i].color;
            unsigned char v = (channel == 0) ? c.r : (channel == 1) ? c.g : c.b;
            t[0] = FloatToHalf((v/255.0f - 0.5f)/SH_C0);
        }
    }
    return texels;
}

static bool LoadGpuRenderer(GpuRenderer *g, const std::vector<Splat> &splats, const SphericalHarmonics &sh)
{
    g->shTexels = (sh.perSplat + 3)/4;  // 1, 4, 9 or 16 coefficients -> 1, 1, 3 or 4 texels
    g->geometryRows = TextureRows(splats.size(), GEOMETRY_TEXELS);
    g->shRows = TextureRows(splats.size(), g->shTexels);
    if (g->geometryRows > TEXTURE_WIDTH || g->shRows > TEXTURE_WIDTH) {
        TraceLog(LOG_WARNING, "Too many splats for the GPU textures, using the CPU renderer");
        return false;
    }

    // Pack and upload one texture at a time, so only one packed copy is in memory.
    {
        std::vector<float> texels = PackGeometryTexels(splats, g->geometryRows);
        g->geometryTexture = rlLoadTexture(texels.data(), TEXTURE_WIDTH, g->geometryRows,
                                           RL_PIXELFORMAT_UNCOMPRESSED_R32G32B32A32, 1);
    }
    for (int c = 0; c < 3; c++) {
        std::vector<uint16_t> texels = PackShTexels(splats, sh, c, g->shTexels, g->shRows);
        g->shTextures[c] = rlLoadTexture(texels.data(), TEXTURE_WIDTH, g->shRows, RL_PIXELFORMAT_UNCOMPRESSED_R16G16B16A16, 1);
    }
    TraceLog(LOG_INFO, "GPU renderer: %.0f MB of splat textures",
             TEXTURE_WIDTH*(16.0*g->geometryRows + 3*8.0*g->shRows)/(1024.0*1024.0));

    g->shader = LoadShaderFromMemory(SPLAT_VS, SPLAT_FS);
    const char *samplers[4] = { "geometry", "shRed", "shGreen", "shBlue" };
    for (int i = 0; i < 4; i++) {
        int unit = 1 + i;  // the splat textures go to texture units 1-4, raylib uses unit 0
        SetShaderValue(g->shader, GetShaderLocation(g->shader, samplers[i]), &unit, SHADER_UNIFORM_INT);
    }
    int width = TEXTURE_WIDTH;
    SetShaderValue(g->shader, GetShaderLocation(g->shader, "textureWidth"), &width, SHADER_UNIFORM_INT);
    SetShaderValue(g->shader, GetShaderLocation(g->shader, "shTexels"), &g->shTexels, SHADER_UNIFORM_INT);
    g->locEye = GetShaderLocation(g->shader, "eye");
    g->locRight = GetShaderLocation(g->shader, "camRight");
    g->locUp = GetShaderLocation(g->shader, "camUp");
    g->locForward = GetShaderLocation(g->shader, "camForward");
    g->locFocal = GetShaderLocation(g->shader, "focal");
    g->locScreen = GetShaderLocation(g->shader, "screenSize");
    g->locScale = GetShaderLocation(g->shader, "scaleMod");
    g->locShDegree = GetShaderLocation(g->shader, "shDegree");
    g->locToFile = GetShaderLocation(g->shader, "toFileFrame");
    g->locMode = GetShaderLocation(g->shader, "mode");
    g->locDotSize = GetShaderLocation(g->shader, "dotSize");

    // Vertex array: attribute 0 = the 6 quad corners, shared by every instance;
    // attribute 1 = the splat index, advancing once per instance (divisor 1).
    // raylib hands every vertex attribute to the shader as floats, and a float
    // holds whole numbers exactly only up to 2^24 (16.7M). So each int index
    // goes as its 4 bytes (RL_UNSIGNED_BYTE, each exact as a float) and the
    // shader puts them back together.
    static const float corners[12] = { -1, -1,  -1, 1,  1, 1,    -1, -1,  1, 1,  1, -1 };  // CCW on screen
    g->vao = rlLoadVertexArray();
    rlEnableVertexArray(g->vao);
    g->cornerVbo = rlLoadVertexBuffer(corners, sizeof(corners), false);
    rlSetVertexAttribute(0, 2, RL_FLOAT, false, 0, 0);
    rlEnableVertexAttribute(0);
    g->orderVbo = rlLoadVertexBuffer(NULL, (int)(splats.size()*sizeof(int)), true);
    rlSetVertexAttribute(1, 4, RL_UNSIGNED_BYTE, false, 0, 0);
    rlSetVertexAttributeDivisor(1, 1);
    rlEnableVertexAttribute(1);
    rlDisableVertexArray();
    return true;
}

// Splats changed on the CPU (the F flip): send the new geometry to the GPU
// copy. The SH textures stay as they are: SH live in the file's frame.
static void UpdateGpuGeometry(GpuRenderer *g, const std::vector<Splat> &splats)
{
    std::vector<float> texels = PackGeometryTexels(splats, g->geometryRows);
    rlUpdateTexture(g->geometryTexture, 0, 0, TEXTURE_WIDTH, g->geometryRows, RL_PIXELFORMAT_UNCOMPRESSED_R32G32B32A32,
                    texels.data());
}

// order: the draw order, one int per instance, uploaded as it is. NULL when
// it is already in the buffer (the GPU sort of STEP 10 writes it there).
static void DrawSplatsGpu(GpuRenderer *g, const int *order, int count, const ViewParams *v, Vector2 screen,
                          float scaleMod, int shDegree, Quaternion toFileFrame, ViewMode mode, float dotSize)
{
    if (count == 0) return;

    if (order) rlUpdateVertexBuffer(g->orderVbo, order, count*(int)sizeof(int), 0);

    int shaderMode = (mode == VIEW_GAUSSIANS) ? 0 : (mode == VIEW_ELLIPSES) ? 1 : 2;
    SetShaderValue(g->shader, g->locEye, &v->eye, SHADER_UNIFORM_VEC3);
    SetShaderValue(g->shader, g->locRight, &v->right, SHADER_UNIFORM_VEC3);
    SetShaderValue(g->shader, g->locUp, &v->up, SHADER_UNIFORM_VEC3);
    SetShaderValue(g->shader, g->locForward, &v->forward, SHADER_UNIFORM_VEC3);
    SetShaderValue(g->shader, g->locFocal, &v->focal, SHADER_UNIFORM_FLOAT);
    SetShaderValue(g->shader, g->locScreen, &screen, SHADER_UNIFORM_VEC2);
    SetShaderValue(g->shader, g->locScale, &scaleMod, SHADER_UNIFORM_FLOAT);
    SetShaderValue(g->shader, g->locShDegree, &shDegree, SHADER_UNIFORM_INT);
    SetShaderValue(g->shader, g->locMode, &shaderMode, SHADER_UNIFORM_INT);
    SetShaderValue(g->shader, g->locDotSize, &dotSize, SHADER_UNIFORM_FLOAT);
    SetShaderValueMatrix(g->shader, g->locToFile, QuaternionToMatrix(toFileFrame));

    rlDrawRenderBatchActive();  // draw anything raylib queued before us first
    rlEnableShader(g->shader.id);
    unsigned int textures[4] = { g->geometryTexture, g->shTextures[0], g->shTextures[1], g->shTextures[2] };
    for (int i = 0; i < 4; i++) {
        rlActiveTextureSlot(1 + i);
        rlEnableTexture(textures[i]);
    }
    rlEnableVertexArray(g->vao);
    rlDrawVertexArrayInstanced(0, 6, count);
    rlDisableVertexArray();
    for (int i = 0; i < 4; i++) {
        rlActiveTextureSlot(1 + i);
        rlDisableTexture();
    }
    rlActiveTextureSlot(0);
    rlDisableShader();
}

static void UnloadGpuRenderer(GpuRenderer *g)
{
    rlUnloadVertexArray(g->vao);
    rlUnloadVertexBuffer(g->cornerVbo);
    rlUnloadVertexBuffer(g->orderVbo);
    rlUnloadTexture(g->geometryTexture);
    for (unsigned int id : g->shTextures) rlUnloadTexture(id);
    UnloadShader(g->shader);
}

// ----------------------------------------------------------------------------
// STEP 10: Cull and sort on the GPU too (compute shaders, OpenGL 4.3)
//
// With STEP 9 the CPU still handles every splat each frame: it culls them,
// sorts them (STEP 7) and uploads the order. With millions of splats that is
// what limits the frame rate. Compute shaders run the same work on the GPU,
// next to the splat data, and the order never leaves the GPU:
//   1. cull     one thread per splat: its camera depth, or -1 if culled; and
//               for the frame the depth range and the number of visible splats
//   2. sort     STEP 7's 16-bit depth key, sorted by a radix sort: one pass
//               for the low byte, then one for the high byte. Each pass splits
//               the splats into tiles of 4096 and runs three shaders:
//                 histogram  per tile: how many keys have each byte value
//                 scan       where each (byte value, tile) starts in the output
//                 scatter    every splat to its place
//               Splats with equal keys keep their order, as in the CPU sort, so
//               splats at the same depth do not swap places from frame to frame
//               (that would flicker). Culled splats get the largest key and end
//               up last. The second pass writes the splat indices straight into
//               the vertex buffer STEP 9 draws from.
//   3. draw     STEP 9's draw call, for as many splats as the cull counted:
//               the one number that comes back to the CPU each frame.
// rlgl wraps most of the calls; memory barriers and the timer come from glad.
// ----------------------------------------------------------------------------

// Shared by the four compute shaders: version, group size, buffers, the key.
static const char *SORT_COMMON = R"(#version 430
layout(local_size_x = 256) in;   // threads per work group
const uint TILE = 4096u;         // splats per tile

// Bound in CullAndSortGpu().
layout(std430, binding = 0) buffer Depths { float depth[]; };      // per splat, -1 = culled
layout(std430, binding = 1) buffer Frame { uint nearBits, farBits, visibleCount; };  // depths as float bits
layout(std430, binding = 2) buffer ByteCounts { uint byteCount[]; };  // [byte value*tiles + tile]
layout(std430, binding = 3) buffer ByteTotals { uint byteTotal[]; };  // [byte value], all tiles together
layout(std430, binding = 4) buffer KeysIn { uint keyIn[]; };          // the first pass's output,
layout(std430, binding = 5) buffer IndicesIn { uint indexIn[]; };     // read by the second pass
layout(std430, binding = 6) buffer KeysOut { uint keyOut[]; };
layout(std430, binding = 7) buffer IndicesOut { uint indexOut[]; };

uniform int splatCount;
uniform int tiles;
uniform int shift;        // 0: first pass, sorts by the key's low byte; 8: second pass, high byte
uniform int sortEnabled;  // 0 (S key): every visible splat gets key 0, so they stay in file order

// STEP 7's key: 0 = farthest ... 65534 = nearest, and 65535 = culled, after all others
uint DepthKey(uint i)
{
    if (depth[i] < 0.0) return 65535u;
    if (sortEnabled == 0) return 0u;
    float nearest = uintBitsToFloat(nearBits), farthest = uintBitsToFloat(farBits);
    float range = max(farthest - nearest, 1e-6);
    return min(uint((farthest - depth[i])/range*65535.0), 65534u);
}

// Key of element i of this pass's input. The first pass reads the splats in file order.
uint Key(uint i) { return (shift == 0) ? DepthKey(i) : keyIn[i]; }
)";

static const char *CULL_CS = R"(
uniform sampler2D geometry;  // STEP 9's texture: splat i's position is in texel 3*i
uniform int textureWidth;
uniform vec3 eye, camRight, camUp, camForward;
uniform float nearPlane;
uniform vec2 limit;          // ToCameraSpace()'s limits, CULL_MARGIN x tan(half field of view)

shared uint groupNear, groupFar, groupVisible;

void main()
{
    if (gl_LocalInvocationIndex == 0u) { groupNear = 0x7f800000u; groupFar = 0u; groupVisible = 0u; }
    barrier();

    // Fewer threads than splats: each thread takes every (number of threads)-th
    // splat. Depths are positive, and positive floats order like their bits read
    // as uints, so min and max can work on the bits: atomics exist for uints.
    uint threadNear = 0x7f800000u, threadFar = 0u, threadVisible = 0u;  // +infinity, 0, none
    for (uint i = gl_GlobalInvocationID.x; i < uint(splatCount); i += gl_NumWorkGroups.x*gl_WorkGroupSize.x) {
        int texel = int(i)*3;
        vec3 d = texelFetch(geometry, ivec2(texel % textureWidth, texel / textureWidth), 0).xyz - eye;
        float tx = dot(d, camRight), ty = dot(d, camUp), tz = dot(d, camForward);
        bool visible = tz >= nearPlane && abs(tx/tz) <= limit.x && abs(ty/tz) <= limit.y;
        depth[i] = visible ? tz : -1.0;
        if (visible) {
            threadNear = min(threadNear, floatBitsToUint(tz));
            threadFar = max(threadFar, floatBitsToUint(tz));
            threadVisible++;
        }
    }
    // Combine the threads of a group first, then the groups: a few thousand
    // atomics on the three frame numbers instead of millions.
    atomicMin(groupNear, threadNear);
    atomicMax(groupFar, threadFar);
    atomicAdd(groupVisible, threadVisible);
    barrier();
    if (gl_LocalInvocationIndex == 0u) {
        atomicMin(nearBits, groupNear);
        atomicMax(farBits, groupFar);
        atomicAdd(visibleCount, groupVisible);
    }
}
)";

// One group per tile: count the tile's keys per byte value.
static const char *HISTOGRAM_CS = R"(
shared uint counts[256];

void main()
{
    uint tile = gl_WorkGroupID.x, t = gl_LocalInvocationIndex;
    counts[t] = 0u;
    barrier();
    uint end = min((tile + 1u)*TILE, uint(splatCount));
    for (uint i = tile*TILE + t; i < end; i += 256u) atomicAdd(counts[(Key(i) >> shift) & 255u], 1u);
    barrier();
    byteCount[t*uint(tiles) + tile] = counts[t];  // thread t writes byte value t
}
)";

// One group per byte value: turn its count in each tile into where that
// tile's keys with this byte value start, counted from the byte value's
// start, and total them up.
static const char *SCAN_CS = R"(
shared uint stretchStart[256];

void main()
{
    uint row = gl_WorkGroupID.x*uint(tiles), t = gl_LocalInvocationIndex;
    uint per = (uint(tiles) + 255u)/256u;  // each thread takes a stretch of `per` tiles
    uint begin = min(t*per, uint(tiles)), end = min(begin + per, uint(tiles));
    uint sum = 0u;
    for (uint k = begin; k < end; k++) sum += byteCount[row + k];
    stretchStart[t] = sum;
    barrier();
    if (t == 0u) {  // stretch sums -> where each stretch starts
        uint run = 0u;
        for (uint k = 0u; k < 256u; k++) { uint c = stretchStart[k]; stretchStart[k] = run; run += c; }
        byteTotal[gl_WorkGroupID.x] = run;
    }
    barrier();
    uint run = stretchStart[t];
    for (uint k = begin; k < end; k++) { uint c = byteCount[row + k]; byteCount[row + k] = run; run += c; }
}
)";

// One group per tile: move every key (and its splat index) to its place.
static const char *SCATTER_CS = R"(
shared uint totals[256];
shared uint next[256];   // where the next key with each byte value goes
shared uint bytes[256];  // the byte value each thread holds, 256 = none

void main()
{
    uint tile = gl_WorkGroupID.x, t = gl_LocalInvocationIndex;
    // Byte value b starts after all keys with a smaller byte value, and this
    // tile's keys with it after those of the tiles before (the scan).
    totals[t] = byteTotal[t];
    barrier();
    if (t == 0u) {
        uint run = 0u;
        for (uint b = 0u; b < 256u; b++) { next[b] = run; run += totals[b]; }
    }
    barrier();
    next[t] += byteCount[t*uint(tiles) + tile];
    barrier();

    // 256 keys at a time, in input order. A key's place among the ones with the
    // same byte value is the number of them before it, so the sort is stable.
    uint end = min((tile + 1u)*TILE, uint(splatCount));
    for (uint first = tile*TILE; first < end; first += 256u) {
        uint i = first + t;
        uint key = (i < end) ? Key(i) : 0u;
        uint b = (i < end) ? (key >> shift) & 255u : 256u;
        bytes[t] = b;
        barrier();
        uint before = 0u, same = 0u;
        for (uint j = 0u; j < 256u; j++) {
            if (bytes[j] == b) { same++; if (j < t) before++; }
        }
        if (i < end) {
            uint dest = next[b] + before;
            if (shift == 0) keyOut[dest] = key;  // the second pass only writes the order
            indexOut[dest] = (shift == 0) ? i : indexIn[i];
        }
        barrier();
        if (i < end && before == same - 1u) next[b] += same;  // the last one moves its byte value on
        barrier();
    }
}
)";

#define SORT_TILE 4096        // splats per tile (TILE in the shaders)
#define SORT_GROUP_SIZE 256   // threads per work group (local_size_x in the shaders)
#define CULL_GROUPS_MAX 1024  // the cull's threads each take several splats beyond this

struct GpuSort {
    unsigned int cull = 0, histogram = 0, scan = 0, scatter = 0;  // compute shader programs
    unsigned int depths = 0, frame = 0, byteCounts = 0, byteTotals = 0, keys = 0, indices = 0;  // buffers
    int splatCount = 0, tiles = 0;
    unsigned int timer = 0;  // GPU timer query around the cull and sort
    double lastMs = 0.0;     // what it measured last frame
};

// One of the shaders above, after SORT_COMMON. 0 if it fails (raylib logs why).
static unsigned int LoadComputeProgram(const char *body)
{
    std::string code = std::string(SORT_COMMON) + body;
    unsigned int shader = rlCompileShader(code.c_str(), RL_COMPUTE_SHADER);
    if (shader == 0) return 0;
    unsigned int program = rlLoadComputeShaderProgram(shader);
    glDeleteShader(shader);  // the linked program does not need it
    return program;
}

// For the program enabled with rlEnableShader()
static void SetComputeUniform(unsigned int program, const char *name, const void *value, int type)
{
    rlSetUniform(rlGetLocationUniform(program, name), value, type, 1);
}

static void UnloadGpuSort(GpuSort *s)
{
    for (unsigned int program : { s->cull, s->histogram, s->scan, s->scatter }) rlUnloadShaderProgram(program);
    for (unsigned int buffer : { s->depths, s->frame, s->byteCounts, s->byteTotals, s->keys, s->indices }) {
        rlUnloadShaderBuffer(buffer);
    }
    glDeleteQueries(1, &s->timer);
}

static bool LoadGpuSort(GpuSort *s, int splatCount)
{
    if (rlGetVersion() != RL_OPENGL_43) {
        TraceLog(LOG_INFO, "GPU sort: raylib is not built for OpenGL 4.3, culling and sorting on the CPU");
        return false;
    }
    s->cull = LoadComputeProgram(CULL_CS);
    s->histogram = LoadComputeProgram(HISTOGRAM_CS);
    s->scan = LoadComputeProgram(SCAN_CS);
    s->scatter = LoadComputeProgram(SCATTER_CS);
    if (!s->cull || !s->histogram || !s->scan || !s->scatter) {
        TraceLog(LOG_WARNING, "GPU sort: a compute shader failed, culling and sorting on the CPU");
        UnloadGpuSort(s);
        return false;
    }

    s->splatCount = splatCount;
    s->tiles = (splatCount + SORT_TILE - 1)/SORT_TILE;
    unsigned int perSplat = splatCount*sizeof(unsigned int);  // floats are the same size
    s->depths = rlLoadShaderBuffer(perSplat, NULL, RL_DYNAMIC_COPY);
    s->frame = rlLoadShaderBuffer(4*sizeof(unsigned int), NULL, RL_DYNAMIC_COPY);
    s->byteCounts = rlLoadShaderBuffer(256*s->tiles*sizeof(unsigned int), NULL, RL_DYNAMIC_COPY);
    s->byteTotals = rlLoadShaderBuffer(256*sizeof(unsigned int), NULL, RL_DYNAMIC_COPY);
    s->keys = rlLoadShaderBuffer(perSplat, NULL, RL_DYNAMIC_COPY);
    s->indices = rlLoadShaderBuffer(perSplat, NULL, RL_DYNAMIC_COPY);
    glGenQueries(1, &s->timer);
    TraceLog(LOG_INFO, "GPU sort: %.0f MB of buffers", (3.0*perSplat + 256.0*s->tiles*4)/(1024.0*1024.0));

    // The uniforms that stay the same for the whole scene
    for (unsigned int program : { s->cull, s->histogram, s->scan, s->scatter }) {
        rlEnableShader(program);
        SetComputeUniform(program, "splatCount", &splatCount, RL_SHADER_UNIFORM_INT);
        SetComputeUniform(program, "tiles", &s->tiles, RL_SHADER_UNIFORM_INT);
    }
    int unit = 1, width = TEXTURE_WIDTH;  // the geometry texture goes to unit 1, as for STEP 9
    float nearPlane = NEAR_PLANE;
    rlEnableShader(s->cull);
    SetComputeUniform(s->cull, "geometry", &unit, RL_SHADER_UNIFORM_INT);
    SetComputeUniform(s->cull, "textureWidth", &width, RL_SHADER_UNIFORM_INT);
    SetComputeUniform(s->cull, "nearPlane", &nearPlane, RL_SHADER_UNIFORM_FLOAT);
    rlDisableShader();
    return true;
}

// Runs one compute shader over `groups` work groups, then waits (on the GPU)
// until what it wrote can be read the way `barrier` says.
static void Dispatch(unsigned int program, int groups, GLbitfield barrier)
{
    rlEnableShader(program);
    rlComputeShaderDispatch(groups, 1, 1);
    glMemoryBarrier(barrier);
}

// STEP 10 for one frame. Returns how many splats are visible; the first that
// many ints of the order buffer are their indices, back to front (file order
// with sorting off).
static int CullAndSortGpu(GpuSort *s, const GpuRenderer *g, const ViewParams *v, bool sortEnabled)
{
    unsigned int frameStart[4] = { 0x7f800000u, 0, 0, 0 };  // nearest +infinity, farthest 0, none visible
    rlUpdateShaderBuffer(s->frame, frameStart, sizeof(frameStart), 0);
    rlBindShaderBuffer(s->depths, 0);
    rlBindShaderBuffer(s->frame, 1);
    rlBindShaderBuffer(s->byteCounts, 2);
    rlBindShaderBuffer(s->byteTotals, 3);
    glBeginQuery(GL_TIME_ELAPSED, s->timer);

    // 1. cull, reading the positions from STEP 9's geometry texture
    Vector2 limit = { CULL_MARGIN*v->tanHalfFovX, CULL_MARGIN*v->tanHalfFovY };
    rlEnableShader(s->cull);
    SetComputeUniform(s->cull, "eye", &v->eye, RL_SHADER_UNIFORM_VEC3);
    SetComputeUniform(s->cull, "camRight", &v->right, RL_SHADER_UNIFORM_VEC3);
    SetComputeUniform(s->cull, "camUp", &v->up, RL_SHADER_UNIFORM_VEC3);
    SetComputeUniform(s->cull, "camForward", &v->forward, RL_SHADER_UNIFORM_VEC3);
    SetComputeUniform(s->cull, "limit", &limit, RL_SHADER_UNIFORM_VEC2);
    rlActiveTextureSlot(1);
    rlEnableTexture(g->geometryTexture);
    int cullGroups = std::min((s->splatCount + SORT_GROUP_SIZE - 1)/SORT_GROUP_SIZE, CULL_GROUPS_MAX);
    Dispatch(s->cull, cullGroups, GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
    rlDisableTexture();
    rlActiveTextureSlot(0);

    // 2. sort: low byte, then high byte. The first pass reads the depths and
    // writes keys and indices; the second reads those and writes the order.
    int sortFlag = sortEnabled ? 1 : 0;
    for (int shift = 0; shift <= 8; shift += 8) {
        bool last = (shift == 8);
        rlBindShaderBuffer(s->keys, 4);
        rlBindShaderBuffer(s->indices, 5);
        rlBindShaderBuffer(s->keys, 6);
        rlBindShaderBuffer(last ? g->orderVbo : s->indices, 7);
        for (unsigned int program : { s->histogram, s->scatter }) {
            rlEnableShader(program);
            SetComputeUniform(program, "shift", &shift, RL_SHADER_UNIFORM_INT);
            SetComputeUniform(program, "sortEnabled", &sortFlag, RL_SHADER_UNIFORM_INT);
        }
        Dispatch(s->histogram, s->tiles, GL_SHADER_STORAGE_BARRIER_BIT);
        Dispatch(s->scan, 256, GL_SHADER_STORAGE_BARRIER_BIT);
        // the order is read as a vertex attribute by the draw
        Dispatch(s->scatter, s->tiles, last ? GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT : GL_SHADER_STORAGE_BARRIER_BIT);
    }
    rlDisableShader();
    glEndQuery(GL_TIME_ELAPSED);

    // 3. the visible count for the draw call. Reading it makes the CPU wait
    // for the GPU, which the draw would have to do anyway.
    unsigned int frame[4];
    rlReadShaderBuffer(s->frame, frame, sizeof(frame), 0);
    GLuint64 nanoseconds = 0;
    glGetQueryObjectui64v(s->timer, GL_QUERY_RESULT, &nanoseconds);
    s->lastMs = nanoseconds/1e6;
    return (int)frame[2];
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
// A scene: the splats of one file and everything made for them. The viewer
// loads one at start, and a new one for every file dropped on the window.
// ----------------------------------------------------------------------------
struct Scene {
    std::string name;  // file name, shown in the HUD
    std::vector<Splat> splats;
    SphericalHarmonics sh;
    bool pointCloud = false;      // a plain point cloud .ply (STEP 2d)
    UpAxis upAxis = UP_PLUS_Y;    // the file's up direction, turned to +Y
    OrbitCamera initialCam = {};
    // per-frame working memory, allocated once
    std::vector<ProjectedSplat> proj;
    std::vector<int> visible, sorted;
    std::vector<uint16_t> keys;
    // STEP 9: the copy of the splats on the GPU
    GpuRenderer gpu;
    bool gpuReady = false;
    // STEP 10: where the GPU culls and sorts them (needs OpenGL 4.3)
    GpuSort gpuSort;
    bool gpuSortReady = false;
};

// path NULL = the demo scene. Returns false if the file gives no splats.
static bool LoadScene(const char *path, Scene *scene)
{
    // STEP 2: get splats
    if (!path) scene->splats = BuildDemoScene();
    else if (IsFileExtension(path, ".splat")) scene->splats = LoadSplatsFromSplatFile(path);
    else scene->splats = LoadSplatsFromPly(path, &scene->sh, &scene->pointCloud);
    if (scene->splats.empty()) return false;
    scene->name = path ? GetFileName(path) : "demo scene";

    // STEP 3: covariance once per splat
    for (Splat &s : scene->splats) ComputeCovariance3D(&s);
    // The demo is built +Y up; splat files are usually COLMAP's -Y, point clouds +Z.
    scene->upAxis = !path ? UP_PLUS_Y : scene->pointCloud ? UP_PLUS_Z : UP_MINUS_Y;
    if (scene->upAxis != UP_PLUS_Y) TurnScene(scene->splats, UpAxisTurn(scene->upAxis));

    scene->initialCam = { .target = { 0.0f, 1.0f, 0.0f }, .distance = 6.5f, .yaw = 0.6f, .pitch = 0.35f, .fovy = 60.0f*DEG2RAD };
    if (path) FrameScene(scene->splats, &scene->initialCam);

    size_t count = scene->splats.size();
    scene->proj.resize(count);
    scene->visible.resize(count);
    scene->sorted.resize(count);
    scene->keys.resize(count);

    // STEP 9: copy the splats to the GPU once
    scene->gpuReady = LoadGpuRenderer(&scene->gpu, scene->splats, scene->sh);
    // STEP 10: the buffers to cull and sort in on the GPU
    scene->gpuSortReady = scene->gpuReady && LoadGpuSort(&scene->gpuSort, (int)count);
    return true;
}

static void UnloadScene(Scene *scene)
{
    if (scene->gpuSortReady) UnloadGpuSort(&scene->gpuSort);
    if (scene->gpuReady) UnloadGpuRenderer(&scene->gpu);
}

// ----------------------------------------------------------------------------
// The three renderers, in the order G cycles through them
// ----------------------------------------------------------------------------
enum Renderer {
    RENDERER_GPU_SORT = 0,  // STEP 10: cull, sort and draw on the GPU
    RENDERER_GPU,           // STEP 9: cull and sort on the CPU, project and draw on the GPU
    RENDERER_CPU,           // STEPs 5-8, all on the CPU: the reference
    RENDERER_COUNT
};
static const char *rendererNames[] = { "GPU, sorted on GPU", "GPU, sorted on CPU", "CPU (reference)" };
static_assert(sizeof(rendererNames)/sizeof(rendererNames[0]) == RENDERER_COUNT, "one name per renderer");

static bool RendererAvailable(const Scene &scene, Renderer r)
{
    if (r == RENDERER_GPU_SORT) return scene.gpuSortReady;
    if (r == RENDERER_GPU) return scene.gpuReady;
    return true;  // RENDERER_CPU
}

// The next renderer after r that this scene can use (G).
static Renderer NextRenderer(const Scene &scene, Renderer r)
{
    do r = (Renderer)((r + 1) % RENDERER_COUNT); while (!RendererAvailable(scene, r));
    return r;
}

// The fastest renderer this scene can use: the first available one.
static Renderer FirstRenderer(const Scene &scene)
{
    return NextRenderer(scene, (Renderer)(RENDERER_COUNT - 1));
}

// ----------------------------------------------------------------------------
// Main: load, then each frame  input -> project -> sort -> draw
// ----------------------------------------------------------------------------
int main(int argc, char **argv)
{
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_VSYNC_HINT);
    InitWindow(1280, 800, "Gaussian splat viewer");

    Scene scene;
    if (!LoadScene((argc > 1) ? argv[1] : NULL, &scene)) { CloseWindow(); return 1; }
    std::string dropFailed;  // name of the last dropped file, if it could not be loaded

    std::vector<unsigned int> buckets(SORT_BUCKETS);  // sort working memory, the same for every scene
    Shader shader = LoadShaderFromMemory(NULL, SPLAT_FS);
    int modeLoc = GetShaderLocation(shader, "mode");

    const float DOT_SIZE = 3.0f;  // dots of the centers view, in pixels
    OrbitCamera cam = scene.initialCam;
    ViewMode mode = scene.pointCloud ? VIEW_CENTERS : VIEW_GAUSSIANS;
    bool sortEnabled = true;
    Renderer renderer = FirstRenderer(scene);
    bool shEnabled = (scene.sh.degree > 0);
    float scaleMod = 1.0f;
    float dotSize = DOT_SIZE;

    while (!WindowShouldClose()) {
        // ---- drag and drop: a .ply or .splat dropped on the window replaces the scene ----
        if (IsFileDropped()) {
            FilePathList dropped = LoadDroppedFiles();
            std::string path = dropped.paths[0];  // of several files, only the first
            UnloadDroppedFiles(dropped);

            // A big file takes seconds to load, so say so on screen first.
            BeginDrawing();
            ClearBackground(Color{ 20, 20, 26, 255 });
            DrawText(TextFormat("Loading %s ...", GetFileName(path.c_str())), 16, 16, 20, RAYWHITE);
            EndDrawing();

            // Load it next to the current scene, so a file that fails to load
            // leaves the current one on screen.
            Scene next;
            if (LoadScene(path.c_str(), &next)) {
                UnloadScene(&scene);
                scene = std::move(next);
                cam = scene.initialCam;  // start fresh, as at launch
                mode = scene.pointCloud ? VIEW_CENTERS : VIEW_GAUSSIANS;
                renderer = FirstRenderer(scene);
                shEnabled = (scene.sh.degree > 0);
                scaleMod = 1.0f;
                dotSize = DOT_SIZE;
                dropFailed.clear();
            } else {
                dropFailed = GetFileName(path.c_str());
            }
        }
        int count = (int)scene.splats.size();

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
        if (IsKeyPressed(KEY_H) && scene.sh.degree > 0) shEnabled = !shEnabled;
        // +/- size what the view shows: the dots in the centers view, else the splats
        float grow = (IsKeyDown(KEY_EQUAL) || IsKeyDown(KEY_KP_ADD)) ? 1.02f : 1.0f;
        if (IsKeyDown(KEY_MINUS) || IsKeyDown(KEY_KP_SUBTRACT)) grow /= 1.02f;
        if (mode == VIEW_CENTERS) dotSize = Clamp(dotSize*grow, 1.0f, 20.0f);
        else scaleMod = Clamp(scaleMod*grow, 0.01f, 4.0f);
        if (IsKeyPressed(KEY_G)) renderer = NextRenderer(scene, renderer);
        if (IsKeyPressed(KEY_F)) {
            // the next up axis: undo the current turn, then make the next one
            UpAxis next = (UpAxis)((scene.upAxis + 1) % UP_AXIS_COUNT);
            Quaternion turn = QuaternionMultiply(UpAxisTurn(next), QuaternionInvert(UpAxisTurn(scene.upAxis)));
            TurnScene(scene.splats, turn);
            scene.upAxis = next;
            if (scene.gpuReady) UpdateGpuGeometry(&scene.gpu, scene.splats);  // keep the GPU copy in sync
            // keep looking at the same part of the scene
            cam.target = Vector3RotateByQuaternion(cam.target, turn);
            scene.initialCam.target = Vector3RotateByQuaternion(scene.initialCam.target, turn);
        }
        if (IsKeyPressed(KEY_R)) { cam = scene.initialCam; scaleMod = 1.0f; dotSize = DOT_SIZE; }
        view = MakeView(&cam, GetScreenWidth(), GetScreenHeight());

        // ---- STEP 5 + 6: project every splat, STEP 5b: its color from this view ----
        // The SH were trained in the file's frame. The scene was turned to put
        // its up axis at +Y, so turn the viewing direction back before evaluating them.
        Quaternion toFileFrame = QuaternionInvert(UpAxisTurn(scene.upAxis));
        double t0 = GetTime();
        int visibleCount = 0;
        if (renderer == RENDERER_GPU_SORT) {
            // STEP 10: cull and sort on the GPU, straight into the draw order
            // (sorting included, so the CPU sort below is skipped).
            visibleCount = CullAndSortGpu(&scene.gpuSort, &scene.gpu, &view, sortEnabled);
        } else if (renderer == RENDERER_GPU) {
            // STEP 9: the CPU only culls and records the depth for the sort.
            // Projection, color and corners happen in the vertex shader (STEP 9).
            for (int i = 0; i < count; i++) {
                float tx, ty, tz;
                if (!ToCameraSpace(scene.splats[i].position, &view, &tx, &ty, &tz)) continue;
                scene.proj[i].depth = tz;
                scene.visible[visibleCount++] = i;
            }
        } else {
            for (int i = 0; i < count; i++) {
                const Splat &sp = scene.splats[i];
                ProjectedSplat &p = scene.proj[i];
                if (!ProjectSplat(&sp, &view, scaleMod, &p)) continue;
                p.color = sp.color;
                if (shEnabled) {
                    const SphericalHarmonics &sh = scene.sh;
                    Vector3 dir = Vector3Normalize(Vector3Subtract(sp.position, view.eye));
                    dir = Vector3RotateByQuaternion(dir, toFileFrame);
                    p.color = EvalSphericalHarmonics(&sh.coeffs[(size_t)i*sh.perSplat], sh.degree, dir, sp.color.a);
                }
                scene.visible[visibleCount++] = i;
            }
        }

        // ---- STEP 7: sort ----
        double t1 = GetTime();
        const int *order = scene.visible.data();  // unsorted = file order
        if (sortEnabled && renderer != RENDERER_GPU_SORT) {
            SortBackToFront(scene.proj.data(), scene.visible.data(), visibleCount, scene.sorted.data(), scene.keys.data(),
                            buckets.data());
            order = scene.sorted.data();
        }

        // ---- STEP 8: draw ----
        double t2 = GetTime();
        BeginDrawing();
        ClearBackground(Color{ 20, 20, 26, 255 });
        if (renderer == RENDERER_CPU) {
            DrawSplats(scene.proj.data(), order, visibleCount, mode, dotSize, shader, modeLoc);
        } else {
            // STEP 10 left the order in the GPU's buffer already: nothing to upload
            Vector2 screen = { (float)GetScreenWidth(), (float)GetScreenHeight() };
            DrawSplatsGpu(&scene.gpu, (renderer == RENDERER_GPU_SORT) ? NULL : order, visibleCount, &view, screen, scaleMod,
                          shEnabled ? scene.sh.degree : 0, toFileFrame, mode, dotSize);
        }
        double t3 = GetTime();

        // ---- HUD ----
        static const char *modeNames[] = { "Gaussians", "Ellipses (2 sigma)", "Centers" };
        DrawRectangle(8, 8, 380, 282, Fade(BLACK, 0.6f));
        DrawText(TextFormat("%d FPS   %d / %d %s visible", GetFPS(), visibleCount, count,
                            scene.pointCloud ? "points" : "splats"), 16, 16, 18, RAYWHITE);
        double cpuMs[3] = { (t1 - t0)*1000.0, (t2 - t1)*1000.0, (t3 - t2)*1000.0 };
        DrawText((renderer == RENDERER_GPU_SORT)
                     ? TextFormat("cull+sort on GPU %.1f ms  draw %.1f ms", scene.gpuSort.lastMs, cpuMs[2])
                 : (renderer == RENDERER_GPU)
                     ? TextFormat("cull %.1f ms  sort %.1f ms  upload+draw %.1f ms", cpuMs[0], cpuMs[1], cpuMs[2])
                     : TextFormat("project %.1f ms  sort %.1f ms  draw %.1f ms", cpuMs[0], cpuMs[1], cpuMs[2]),
                 16, 38, 16, LIGHTGRAY);
        DrawText(dropFailed.empty() ? TextFormat("[drop] file:    %s%s", scene.name.c_str(),
                                                scene.pointCloud ? " (point cloud)" : "")
                                    : TextFormat("[drop] file:    cannot open %s", dropFailed.c_str()), 16, 66, 16,
                 dropFailed.empty() ? RAYWHITE : ORANGE);
        DrawText(TextFormat("[G] renderer:   %s", scene.gpuReady ? rendererNames[renderer] : "CPU (GPU unavailable)"),
                 16, 86, 16, (renderer != FirstRenderer(scene)) ? ORANGE : RAYWHITE);
        DrawText(TextFormat("[1/2/3] view:   %s", modeNames[mode]), 16, 106, 16, RAYWHITE);
        DrawText(TextFormat("[S] sort:       %s", sortEnabled ? "on (back to front)" : "OFF (file order)"), 16, 126, 16,
                 sortEnabled ? RAYWHITE : ORANGE);
        DrawText((scene.sh.degree == 0) ? "[H] SH color:   none in file"
                 : shEnabled            ? TextFormat("[H] SH color:   on (degree %d)", scene.sh.degree)
                                        : "[H] SH color:   OFF (base color)", 16, 146, 16,
                 (scene.sh.degree > 0 && !shEnabled) ? ORANGE : RAYWHITE);
        DrawText((mode == VIEW_CENTERS) ? TextFormat("[+/-] dot size: %.1f px", dotSize)
                                        : TextFormat("[+/-] scale:    %.2fx", scaleMod), 16, 166, 16, RAYWHITE);
        DrawText(TextFormat("[F] up axis:    %s", upAxisNames[scene.upAxis]), 16, 186, 16, RAYWHITE);
        DrawText("[R] reset camera", 16, 206, 16, RAYWHITE);
        DrawText("LMB orbit  RMB/MMB pan  wheel zoom", 16, 234, 16, LIGHTGRAY);
        DrawText("F12 screenshot  Esc quit", 16, 254, 16, LIGHTGRAY);
        EndDrawing();
    }

    UnloadScene(&scene);
    UnloadShader(shader);
    CloseWindow();
    return 0;
}
