// ============================================================================
//  STEP 12: Stream the viewer to web browsers.
//
//  With --serve the viewer renders as usual, on this computer's GPU, but into a
//  texture the size of the browser's window. Each frame goes to the browsers as
//  a JPEG over a WebSocket, and their keys, mouse and wheel come back over the
//  same socket, to be replayed into raylib as if they were this window's. A
//  browser on a laptop can then fly through millions of splats: only pictures
//  travel, never the scene. Everything network is in web_stream.cpp, along with
//  the web page itself, so splat_viewer.cpp only sees the functions below.
// ============================================================================
#pragma once

#include <vector>

#define WEB_DEFAULT_PORT 8080

struct WebStream;  // the server's sockets and thread

// One input event from a browser. Key and button codes are raylib's (the page translates).
struct WebInput {
    enum Type { KEY, BUTTON, MOVE, WHEEL } type;
    int a, b;  // KEY, BUTTON: the code, and 1 pressed / 0 released;  MOVE: dx, dy in frame pixels;
               // WHEEL: notches, + away from the user, and 0
};

// Serves the page and the stream on every network interface, on its own thread, and
// prints the addresses to open. NULL if the port cannot be had (in use, or not POSIX).
WebStream *OpenWebStream(int port);
void CloseWebStream(WebStream *web);

// Once per frame, before input: the events the browsers sent since the last call, in
// the order they came. A press and release of one key in the same batch would cancel
// out within a frame, so the release waits for the next call. A browser that leaves
// releases what it still held. *width x *height: the frame size the browsers want,
// the window of the one that joined or resized last; 0 x 0 while none is connected.
void ReceiveWebInput(WebStream *web, std::vector<WebInput> *events, int *width, int *height);

// True when a browser can take a frame and the last one is on its way: only then is
// reading a frame back from the GPU worth it.
bool WebWantsFrame(const WebStream *web);

// One frame, RGBA, bottom row first, as glReadPixels gives it. Copied: it is compared
// with the last frame sent (a camera that stands still sends nothing new), encoded and
// sent on the stream's thread.
void SendWebFrame(WebStream *web, const unsigned char *rgba, int width, int height);

int WebViewerCount(const WebStream *web);       // browsers connected now
const char *WebStreamUrl(const WebStream *web); // http://<this computer>:<port>/, for the window
