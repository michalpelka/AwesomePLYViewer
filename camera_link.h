// ============================================================================
//  Camera link: viewers open on one computer share one camera, and the view
//  picked with keys 1-7, over ZeroMQ.
//
//  Fly in any of them and the others follow. The first viewer to start becomes
//  the server; the ones after it connect to it as clients. When the server
//  quits, a client takes its place. Everything ZeroMQ is in camera_link.cpp,
//  so splat_viewer.cpp only sees the functions below.
// ============================================================================
#pragma once

// The camera that linked viewers share. It is in the coordinates of the scene
// file, before the turn that brings the file's up axis to +Y (Z and F in the
// viewer), so two viewers that turned the same file differently still look at
// the same part of it.
struct LinkedCamera {
    float position[3];
    float forward[3];  // unit vector, where the camera looks
    float fovy;        // vertical field of view in radians
    float speed;       // how fast WASD moves, in world units per second
};

// Everything linked viewers share. Each part changes on its own: moving the
// camera leaves the view alone, and picking a view leaves the camera alone.
struct LinkedState {
    LinkedCamera camera;
    int viewMode;  // the view last picked with keys 1-7 (ViewMode); -1 = none picked yet
};
enum LinkedPart { LINKED_CAMERA = 1, LINKED_VIEW_MODE = 2 };  // bits, one per part of LinkedState

struct CameraLink;  // the ZeroMQ sockets, and the newest of each part this viewer knows

// Joins the other viewers on this computer, as server or client. NULL if the
// viewer was built without ZeroMQ.
CameraLink *OpenCameraLink();
void CloseCameraLink(CameraLink *link);

// Once per frame, before input: sets *state to the newest of each part this
// viewer knows, and returns the parts (LinkedPart bits) another viewer changed
// since the last call. Without a link, returns 0 and leaves *state alone.
int ReceiveLinked(CameraLink *link, LinkedState *state);

// Once per frame, after input: state is this viewer's camera and view, and
// changedHere the parts (LinkedPart bits) its input changed this frame. Only
// those go to the other viewers (and, from a new server, its camera).
void SendLinked(CameraLink *link, const LinkedState &state, int changedHere);

// For the HUD: "server" or "client"
const char *CameraLinkRole(const CameraLink *link);
