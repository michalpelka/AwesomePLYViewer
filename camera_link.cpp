// ============================================================================
//  Camera link over ZeroMQ (see camera_link.h)
//
//  Two TCP ports on 127.0.0.1, one for each direction:
//
//      server  PUB  :25150  ->  SUB   clients   the shared camera, every frame
//      server  PULL :25151  <-  PUSH  clients   a camera a client's input moved
//
//  Whoever binds the ports first is the server. The server sends every frame,
//  not only when the camera moves: a viewer that starts later gets the camera
//  at once, and a client that stops hearing it knows the server has gone and
//  binds the ports itself.
//
//  Which value wins: each part of LinkedState (the camera, the view) carries a
//  stamp of its own, a (version, origin) pair, origin being the viewer whose
//  input made it. Input sets the version one past the newest the viewer has
//  seen (a Lamport clock), and a viewer takes a part only if its stamp is
//  greater than the one it has, the origin breaking ties. So a client ignores
//  its own moves coming back from the server, and the server's older camera
//  crossing them on the way; every viewer ends up with the same camera, the
//  one last moved. With a stamp per part, a viewer that moves the camera while
//  another picks a view keeps both changes.
// ============================================================================
#include "camera_link.h"

#ifndef CAMERA_LINK_ZMQ  // built without ZeroMQ: there is no link

CameraLink *OpenCameraLink() { return nullptr; }
void CloseCameraLink(CameraLink *) {}
int ReceiveLinked(CameraLink *, LinkedState *) { return 0; }
void SendLinked(CameraLink *, const LinkedState &, int) {}
const char *CameraLinkRole(const CameraLink *) { return "off"; }

#else

#include <zmq.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <random>

static const char *STATE_ENDPOINT = "tcp://127.0.0.1:25150";  // server PUB -> client SUB
static const char *MOVE_ENDPOINT = "tcp://127.0.0.1:25151";   // client PUSH -> server PULL
static const double TAKEOVER_SECONDS = 2.0;  // a client that has not heard the server this long tries to replace it

// Which change a part's value came from. (0, 0) = none yet.
struct Stamp {
    uint64_t version, origin;
};

// What goes over the wire, as raw bytes: the viewers share one computer, so
// they agree on byte order and layout. MAGIC changes whenever Message does, so
// messages from other versions of the viewer, or from other programs, are dropped.
static const uint32_t MAGIC = 0x324b4c53;  // "SLK2"
struct Message {
    uint32_t magic;
    uint32_t unused;
    Stamp camera, viewMode;  // one stamp per part of state
    LinkedState state;
};

using Clock = std::chrono::steady_clock;

struct CameraLink {
    void *context;
    void *in, *out;               // server: PULL and PUB, bound; client: SUB and PUSH, connected
    bool server;
    uint64_t id;                  // this viewer, random
    Stamp camera, viewMode;       // of the newest of each part this viewer knows
    LinkedState state;            // and their values
    bool announce;                // a new server sends its camera as a move, so that the clients take it
    Clock::time_point lastHeard;  // client: when the server last sent
};

static bool Newer(const Stamp &a, const Stamp &b)
{
    return a.version > b.version || (a.version == b.version && a.origin > b.origin);
}

// Takes one part of a received message, if it is newer than the one the link has.
template <typename T>
static bool Take(Stamp *stamp, T *value, const Stamp &newStamp, const T &newValue)
{
    if (!Newer(newStamp, *stamp)) return false;
    *stamp = newStamp;
    *value = newValue;
    return true;
}

// A change this viewer's input made to one part: one version past the newest seen.
template <typename T>
static void Change(const CameraLink *link, Stamp *stamp, T *value, const T &newValue)
{
    *stamp = Stamp{ stamp->version + 1, link->id };
    *value = newValue;
}

static bool Finite(const LinkedCamera &camera)
{
    const float *f = reinterpret_cast<const float *>(&camera);  // LinkedCamera holds only floats
    for (size_t i = 0; i < sizeof camera/sizeof(float); i++)
        if (!std::isfinite(f[i])) return false;
    return true;
}

static void *Socket(CameraLink *link, int type)
{
    void *socket = zmq_socket(link->context, type);
    int linger = 0;  // closing drops unsent cameras instead of waiting to deliver them
    zmq_setsockopt(socket, ZMQ_LINGER, &linger, sizeof linger);
    return socket;
}

static void CloseSockets(CameraLink *link)
{
    if (link->in) zmq_close(link->in);
    if (link->out) zmq_close(link->out);
    link->in = link->out = nullptr;
}

// Binds both ports. Fails, changing nothing, when another viewer has them: it is the server.
static bool BecomeServer(CameraLink *link)
{
    void *pub = Socket(link, ZMQ_PUB), *pull = Socket(link, ZMQ_PULL);
    if (zmq_bind(pub, STATE_ENDPOINT) != 0 || zmq_bind(pull, MOVE_ENDPOINT) != 0) {
        zmq_close(pub);
        zmq_close(pull);
        return false;
    }
    CloseSockets(link);
    link->in = pull;
    link->out = pub;
    link->server = true;
    link->announce = true;
    return true;
}

// Connecting never fails: ZeroMQ keeps trying in the background until a server is there.
static void BecomeClient(CameraLink *link)
{
    int on = 1;
    void *sub = Socket(link, ZMQ_SUB);
    zmq_setsockopt(sub, ZMQ_CONFLATE, &on, sizeof on);  // keep only the newest camera, a slow viewer skips the rest
    zmq_setsockopt(sub, ZMQ_SUBSCRIBE, "", 0);          // every message
    zmq_connect(sub, STATE_ENDPOINT);
    void *push = Socket(link, ZMQ_PUSH);
    zmq_setsockopt(push, ZMQ_IMMEDIATE, &on, sizeof on);  // with no server, drop moves instead of queueing them
    zmq_connect(push, MOVE_ENDPOINT);
    link->in = sub;
    link->out = push;
    link->server = false;
    link->lastHeard = Clock::now();
}

CameraLink *OpenCameraLink()
{
    CameraLink *link = new CameraLink{};
    link->state.viewMode = -1;  // none picked yet
    link->context = zmq_ctx_new();
    std::random_device random;
    link->id = ((uint64_t)random() << 32 | random()) | 1;  // never 0, which stands for "none"
    if (!BecomeServer(link)) BecomeClient(link);
    return link;
}

void CloseCameraLink(CameraLink *link)
{
    if (!link) return;
    CloseSockets(link);
    zmq_ctx_term(link->context);
    delete link;
}

int ReceiveLinked(CameraLink *link, LinkedState *state)
{
    if (!link) return 0;
    int changed = 0;
    Message m;
    int size;
    // Everything that arrived since the last frame; the newest of each part wins.
    while ((size = zmq_recv(link->in, &m, sizeof m, ZMQ_DONTWAIT)) >= 0) {
        // zmq_recv returns the full size of a longer message it truncated, so this drops those too
        if (size != (int)sizeof m || m.magic != MAGIC || !Finite(m.state.camera)) continue;
        link->lastHeard = Clock::now();
        if (Take(&link->camera, &link->state.camera, m.camera, m.state.camera)) changed |= LINKED_CAMERA;
        if (Take(&link->viewMode, &link->state.viewMode, m.viewMode, m.state.viewMode)) changed |= LINKED_VIEW_MODE;
    }
    // A client that has not heard the server for a while: the server may have quit, so try
    // to take its place. If it is still there (busy loading a big file, say), wait again.
    if (!link->server && Clock::now() - link->lastHeard > std::chrono::duration<double>(TAKEOVER_SECONDS)) {
        if (!BecomeServer(link)) link->lastHeard = Clock::now();
    }
    *state = link->state;
    return changed;
}

void SendLinked(CameraLink *link, const LinkedState &state, int changedHere)
{
    if (!link) return;
    if (link->announce) {
        changedHere |= LINKED_CAMERA;
        link->announce = false;
    }
    if (changedHere & LINKED_CAMERA) Change(link, &link->camera, &link->state.camera, state.camera);
    if (changedHere & LINKED_VIEW_MODE) Change(link, &link->viewMode, &link->state.viewMode, state.viewMode);
    if (!changedHere && !link->server) return;  // a client sends only its own changes, the server all every frame

    Message m{};  // zeroed, padding too
    m.magic = MAGIC;
    m.camera = link->camera;
    m.viewMode = link->viewMode;
    m.state = link->state;
    zmq_send(link->out, &m, sizeof m, ZMQ_DONTWAIT);  // never waits: PUB drops when full, PUSH with no server
}

const char *CameraLinkRole(const CameraLink *link)
{
    return !link ? "off" : link->server ? "server" : "client";
}

#endif
