// ============================================================================
//  STEP 12b: Headless: stream from a computer with no display (Linux).
//
//  raylib opens a window through GLFW, and on Linux GLFW needs an X11 or
//  Wayland display for it, which an SSH session, or a server without a desktop,
//  does not have. OpenGL itself does not: EGL, the GPU driver's own interface,
//  gives a context on the GPU without any display. With --headless, or with
//  --serve and no display, InitWindow() gets a window of GLFW's null platform,
//  which needs no display and shows nothing, and an OpenGL context from EGL. The
//  frames go to textures and on to the browsers (STEP 12) as before.
//  How raylib gets that context without a change to raylib: see headless.cpp.
// ============================================================================
#pragma once

// True if a window can open: GLFW starts on this computer's display.
bool HaveDisplay();

// Before InitWindow(): the next window comes headless, as above. False, after saying
// why, if EGL gives no OpenGL context (or the viewer was built without EGL).
bool StartHeadless();
