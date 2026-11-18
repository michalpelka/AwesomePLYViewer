// ============================================================================
//  STEP 12b: Headless (headless.h)
//
//  raylib starts GLFW and makes its window's OpenGL context with six GLFW calls.
//  The linker is told to send those calls here instead (--wrap, in
//  CMakeLists.txt): a call to glfwX reaches __wrap_glfwX below, which passes it
//  on to GLFW's own, __real_glfwX, unless the viewer is headless. Headless, they
//  do this:
//    _glfwSelectPlatform     GLFW's null platform, which needs no display. GLFW
//                            3.4 builds it in, but glfwInit() cannot pick it (it
//                            can from 3.5 on), so this call of glfwInit()'s, inside
//                            GLFW, is the one to take over
//    glfwCreateWindow        a window without a context (GLFW_NO_API)
//    glfwMakeContextCurrent  makes EGL's context current instead
//    glfwGetProcAddress      looks the OpenGL functions up through EGL
//    glfwSwapInterval        nothing: no screen to wait for
//    glfwSwapBuffers         nothing: no screen to show the frame on
//  The rest of raylib, rlgl included, never knows the difference.
// ============================================================================
#include "headless.h"

#define GLFW_INCLUDE_NONE  // no OpenGL header: glad (in raylib) has the OpenGL functions
#include "external/glfw/include/GLFW/glfw3.h"  // the GLFW built into raylib

#include <cstdio>

bool HaveDisplay()
{
    if (!glfwInit()) return false;
    glfwTerminate();  // InitWindow() starts it again
    return true;
}

#ifdef SPLAT_VIEWER_HEADLESS

#include "rlgl.h"  // rlGetVersion(): the OpenGL version raylib was built for

#include <EGL/egl.h>
#include <EGL/eglext.h>

static bool headless = false;
static EGLDisplay display = EGL_NO_DISPLAY;
static EGLSurface surface = EGL_NO_SURFACE;
static EGLContext context = EGL_NO_CONTEXT;

static bool Fail(const char *what)
{
    fprintf(stderr, "Headless: %s (EGL error 0x%x)\n", what, eglGetError());
    return false;
}

bool StartHeadless()
{
    int major = 4, minor = 3;  // STEP 10's compute shaders
    if (rlGetVersion() == RL_OPENGL_33) minor = major = 3;
    else if (rlGetVersion() != RL_OPENGL_43) { fprintf(stderr, "Headless: needs raylib built for OpenGL 3.3 or 4.3\n"); return false; }

    // EGL's surfaceless platform needs no display server. Through libglvnd, the first
    // driver that offers it answers, NVIDIA's before Mesa's.
    display = eglGetPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL);
    if (display == EGL_NO_DISPLAY || !eglInitialize(display, NULL, NULL)) return Fail("no EGL display without a display server");
    if (!eglBindAPI(EGL_OPENGL_API)) return Fail("EGL has no desktop OpenGL");
    const EGLint configAttributes[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8, EGL_DEPTH_SIZE, 24, EGL_NONE
    };
    EGLConfig config;
    EGLint configs = 0;
    if (!eglChooseConfig(display, configAttributes, &config, 1, &configs) || configs < 1) {
        return Fail("no EGL configuration for OpenGL");
    }
    const EGLint contextAttributes[] = {
        EGL_CONTEXT_MAJOR_VERSION, major, EGL_CONTEXT_MINOR_VERSION, minor,
        EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT, EGL_NONE
    };
    context = eglCreateContext(display, config, EGL_NO_CONTEXT, contextAttributes);
    if (context == EGL_NO_CONTEXT) {
        fprintf(stderr, "Headless: no OpenGL %d.%d core context from EGL (error 0x%x)\n", major, minor, eglGetError());
        return false;
    }
    // A small pbuffer stands in for the window's framebuffer. Nothing shows it: the
    // frames are drawn into textures, and read back from them (STEP 12).
    const EGLint pbufferAttributes[] = { EGL_WIDTH, 16, EGL_HEIGHT, 16, EGL_NONE };
    surface = eglCreatePbufferSurface(display, config, pbufferAttributes);
    if (surface == EGL_NO_SURFACE) return Fail("no pbuffer surface");
    if (!eglMakeCurrent(display, surface, surface, context)) return Fail("the context cannot be made current");

    headless = true;
    fprintf(stderr, "Headless: OpenGL %d.%d through EGL, without a display\n", major, minor);
    return true;
}

extern "C" {
// GLFW's internals (src/internal.h, null_platform.h of the GLFW raylib 5.5 builds in): GLFWbool is an
// int, and the platform's table of functions only goes through here by its address.
int __real__glfwSelectPlatform(int desiredID, void *platform);
int _glfwConnectNull(int platformID, void *platform);

int __wrap__glfwSelectPlatform(int desiredID, void *platform)
{
    return headless ? _glfwConnectNull(GLFW_PLATFORM_NULL, platform) : __real__glfwSelectPlatform(desiredID, platform);
}

GLFWwindow *__real_glfwCreateWindow(int width, int height, const char *title, GLFWmonitor *monitor, GLFWwindow *share);
void __real_glfwMakeContextCurrent(GLFWwindow *window);
GLFWglproc __real_glfwGetProcAddress(const char *name);
void __real_glfwSwapInterval(int interval);
void __real_glfwSwapBuffers(GLFWwindow *window);

GLFWwindow *__wrap_glfwCreateWindow(int width, int height, const char *title, GLFWmonitor *monitor, GLFWwindow *share)
{
    if (headless) glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);  // its context comes from EGL
    return __real_glfwCreateWindow(width, height, title, monitor, share);
}

void __wrap_glfwMakeContextCurrent(GLFWwindow *window)
{
    if (!headless) { __real_glfwMakeContextCurrent(window); return; }
    eglMakeCurrent(display, window ? surface : EGL_NO_SURFACE, window ? surface : EGL_NO_SURFACE,
                   window ? context : EGL_NO_CONTEXT);
    glfwGetError(NULL);  // raylib reads any GLFW error pending now as a context that failed
}

GLFWglproc __wrap_glfwGetProcAddress(const char *name)
{
    return headless ? (GLFWglproc)eglGetProcAddress(name) : __real_glfwGetProcAddress(name);
}

void __wrap_glfwSwapInterval(int interval)
{
    if (!headless) __real_glfwSwapInterval(interval);
}

void __wrap_glfwSwapBuffers(GLFWwindow *window)
{
    if (!headless) __real_glfwSwapBuffers(window);
}
}

#else  // built without EGL (macOS, or no libEGL)

bool StartHeadless()
{
    fprintf(stderr, "Headless: this viewer was built without EGL\n");
    return false;
}

#endif
