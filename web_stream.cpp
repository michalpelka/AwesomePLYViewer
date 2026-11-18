// ============================================================================
//  STEP 12: The web stream (web_stream.h), a small HTTP and WebSocket server.
//
//  One thread runs one poll() loop over the listening socket, the browsers'
//  sockets, and a pipe the main thread writes to when it has a frame. It serves
//  two addresses:
//    /     the page (Page(), at the end of this file): it shows the frames and
//          sends the input
//    /ws   the WebSocket (RFC 6455). Frames go out as binary messages, one JPEG
//          each; input comes in as short text messages:
//            k <key> <1|0>        key pressed / released (raylib's key code)
//            b <button> <1|0>     mouse button (raylib's: 0 left, 1 right, 2 middle)
//            m <dx> <dy>          mouse motion while a button is held, in frame pixels
//            w <notches>          wheel, + away from the user
//            s <width> <height>   the browser's window: the frame size it wants
//            a                    a frame is drawn: send the next
//  Flow control: a browser has at most MAX_IN_FLIGHT frames sent but not yet
//  drawn (a). A slow connection then gets fewer frames, each the newest one,
//  instead of a queue of old ones that adds to the delay.
// ============================================================================
#include "web_stream.h"

#include <cstdio>

#ifndef _WIN32

#define STB_IMAGE_WRITE_STATIC  // all of it private to this file: raylib links its own copy
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "external/stb_image_write.h"  // raylib's copy of stb_image_write, for its JPEG encoder

#include <arpa/inet.h>
#include <fcntl.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <utility>

#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0  // macOS: SO_NOSIGPIPE on each socket instead
#endif

static const char *Page();  // the web page, at the end of this file

#define MAX_IN_FLIGHT 2      // frames sent to a browser and not yet drawn by it
#define MAX_CONNECTIONS 16   // browsers, and requests for the page
#define MAX_MESSAGE 4096     // longest message a browser may send; input messages are a few bytes
#define MAX_REQUEST 16384    // longest HTTP request head
#define MAX_QUEUED 4096      // input events waiting for the main thread (it may be loading a file)
#define JPEG_QUALITY 80      // below 90, stb also halves the color resolution (4:2:0), as video does
#define MIN_SIZE 16          // frame size limits, in pixels
#define MAX_WIDTH 3840
#define MAX_HEIGHT 2160

// One connection: a browser's WebSocket, or a request for the page.
struct Browser {
    int fd = -1;
    std::string address;          // where it connects from, for the log
    bool webSocket = false;       // false: an HTTP request, not yet answered
    bool closing = false;         // close once `out` is sent
    bool dead = false;            // closed: remove
    std::string in, out;          // received and not yet parsed; to send and not yet sent
    int inFlight = 0;             // frames sent, not yet drawn
    bool hasNewest = false;       // has been sent the newest frame
    std::set<int> keys, buttons;  // held now: released for it when it leaves
    int width = 0, height = 0;    // the frame size it wants
    long sizeOrder = 0;           // when it asked for that size: the latest ask wins
};

struct WebStream {
    int listener = -1;
    int wake[2] = { -1, -1 };  // the main thread writes a byte here: a frame is waiting, or quit
    std::string url;
    std::thread thread;
    std::atomic<bool> quit{ false };

    // Shared with the main thread, under `mutex`
    std::mutex mutex;
    std::vector<WebInput> input;       // events not yet taken by ReceiveWebInput()
    int width = 0, height = 0;         // the frame size browsers want
    std::vector<unsigned char> frame;  // the newest frame from SendWebFrame()
    int frameWidth = 0, frameHeight = 0;
    std::atomic<bool> framePending{ false };  // `frame` holds one the thread has not taken yet
    std::atomic<bool> ready{ false };         // a browser can take a frame
    std::atomic<int> viewers{ 0 };

    // The thread's own
    std::vector<Browser> browsers;
    std::vector<unsigned char> working, lastSent;  // the frame taken from the main thread, and the last one sent
    int lastWidth = 0, lastHeight = 0;
    std::string newest;      // the newest frame as a WebSocket message, kept for browsers that have not had it
    long sizeCounter = 0;
};

// ----------------------------------------------------------------------------
// The WebSocket handshake: the server proves it speaks WebSocket by answering
// the browser's random key with base64(SHA-1(key + a fixed GUID)).
// ----------------------------------------------------------------------------
static uint32_t RotateLeft(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }

static void Sha1(const std::string &message, unsigned char digest[20])
{
    uint32_t h[5] = { 0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0 };
    std::string m = message + '\x80';
    while (m.size() % 64 != 56) m += '\0';
    uint64_t bits = (uint64_t)message.size()*8;
    for (int i = 7; i >= 0; i--) m += (char)((bits >> (8*i)) & 0xff);
    for (size_t chunk = 0; chunk < m.size(); chunk += 64) {
        uint32_t w[80];
        for (int i = 0; i < 16; i++) {
            const unsigned char *p = (const unsigned char *)&m[chunk + 4*i];
            w[i] = ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3];
        }
        for (int i = 16; i < 80; i++) w[i] = RotateLeft(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
        for (int i = 0; i < 80; i++) {
            uint32_t f, k;
            if (i < 20) { f = (b & c) | (~b & d); k = 0x5A827999; }
            else if (i < 40) { f = b ^ c ^ d; k = 0x6ED9EBA1; }
            else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDC; }
            else { f = b ^ c ^ d; k = 0xCA62C1D6; }
            uint32_t t = RotateLeft(a, 5) + f + e + k + w[i];
            e = d; d = c; c = RotateLeft(b, 30); b = a; a = t;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e;
    }
    for (int i = 0; i < 20; i++) digest[i] = (unsigned char)(h[i/4] >> (24 - 8*(i % 4)));
}

static std::string Base64(const unsigned char *data, size_t size)
{
    static const char digits[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    for (size_t i = 0; i < size; i += 3) {
        uint32_t n = (uint32_t)data[i] << 16;
        if (i + 1 < size) n |= (uint32_t)data[i + 1] << 8;
        if (i + 2 < size) n |= data[i + 2];
        out += digits[(n >> 18) & 63];
        out += digits[(n >> 12) & 63];
        out += (i + 1 < size) ? digits[(n >> 6) & 63] : '=';
        out += (i + 2 < size) ? digits[n & 63] : '=';
    }
    return out;
}

static std::string WebSocketAccept(const std::string &key)
{
    unsigned char digest[20];
    Sha1(key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11", digest);  // the GUID of RFC 6455
    return Base64(digest, sizeof(digest));
}

// The head of a message from the server: final, the opcode, and the length in 1, 3 or 9 bytes.
// Unlike the browser's, the server's messages are not masked.
static std::string MessageHead(int opcode, size_t length)
{
    std::string head(1, (char)(0x80 | opcode));
    if (length < 126) {
        head += (char)length;
    } else if (length < 65536) {
        head += (char)126;
        head += (char)(length >> 8);
        head += (char)(length & 0xff);
    } else {
        head += (char)127;
        for (int i = 7; i >= 0; i--) head += (char)((length >> (8*i)) & 0xff);
    }
    return head;
}

// ----------------------------------------------------------------------------
// Input from a browser
// ----------------------------------------------------------------------------
// The frame size: the browser that asked last. Called under the mutex.
static void UpdateFrameSize(WebStream *web)
{
    const Browser *latest = NULL;
    for (const Browser &b : web->browsers) {
        if (b.webSocket && !b.dead && b.width > 0 && (!latest || b.sizeOrder > latest->sizeOrder)) latest = &b;
    }
    web->width = latest ? latest->width : 0;
    web->height = latest ? latest->height : 0;
}

static void HandleMessage(WebStream *web, Browser *b, const std::string &text)
{
    int x = 0, y = 0;
    const char *args = text.c_str() + 1;
    std::lock_guard<std::mutex> lock(web->mutex);
    switch (text.empty() ? 0 : text[0]) {
    case 'a':
        if (b->inFlight > 0) b->inFlight--;
        break;
    case 'k':  // only real changes go on: a press of a held key, or a release of one not held, is dropped
        if (sscanf(args, "%d %d", &x, &y) == 2 && x > 0 && x < 512) {  // raylib's MAX_KEYBOARD_KEYS
            if (y ? b->keys.insert(x).second : b->keys.erase(x) > 0) web->input.push_back({ WebInput::KEY, x, y ? 1 : 0 });
        }
        break;
    case 'b':
        if (sscanf(args, "%d %d", &x, &y) == 2 && x >= 0 && x <= 2) {  // left, right, middle
            if (y ? b->buttons.insert(x).second : b->buttons.erase(x) > 0) {
                web->input.push_back({ WebInput::BUTTON, x, y ? 1 : 0 });
            }
        }
        break;
    case 'm':
        if (sscanf(args, "%d %d", &x, &y) == 2 && web->input.size() < MAX_QUEUED) {
            web->input.push_back({ WebInput::MOVE, std::clamp(x, -4096, 4096), std::clamp(y, -4096, 4096) });
        }
        break;
    case 'w':
        if (sscanf(args, "%d", &x) == 1 && web->input.size() < MAX_QUEUED) {
            web->input.push_back({ WebInput::WHEEL, std::clamp(x, -20, 20), 0 });
        }
        break;
    case 's':
        if (sscanf(args, "%d %d", &x, &y) == 2) {
            b->width = std::clamp(x, MIN_SIZE, MAX_WIDTH);
            b->height = std::clamp(y, MIN_SIZE, MAX_HEIGHT);
            b->sizeOrder = ++web->sizeCounter;
            UpdateFrameSize(web);
        }
        break;
    }
}

// Parses the WebSocket messages in b->in.
static void ReceiveMessages(WebStream *web, Browser *b)
{
    for (;;) {
        const unsigned char *p = (const unsigned char *)b->in.data();
        size_t have = b->in.size();
        if (have < 2) return;
        bool whole = p[0] & 0x80, masked = p[1] & 0x80;
        int opcode = p[0] & 0x0f;
        uint64_t length = p[1] & 0x7f;
        size_t head = 2;
        if (length == 126) {
            if (have < 4) return;
            length = ((uint64_t)p[2] << 8) | p[3];
            head = 4;
        } else if (length == 127) {
            if (have < 10) return;
            length = 0;
            for (int i = 0; i < 8; i++) length = (length << 8) | p[2 + i];
            head = 10;
        }
        // Browsers always mask, and never split a message this short into pieces.
        if (!masked || !whole || opcode == 0 || length > MAX_MESSAGE) { b->dead = true; return; }
        if (have < head + 4 + length) return;  // the rest has not arrived yet
        const unsigned char *mask = p + head;
        std::string payload(length, '\0');
        for (size_t i = 0; i < length; i++) payload[i] = (char)(p[head + 4 + i] ^ mask[i % 4]);
        b->in.erase(0, head + 4 + length);
        if (opcode == 0x1) {  // text: input
            HandleMessage(web, b, payload);
        } else if (opcode == 0x8) {  // close: answer, then close
            b->out += MessageHead(0x8, 0);
            b->closing = true;
            return;
        } else if (opcode == 0x9) {  // ping: pong
            b->out += MessageHead(0xA, payload.size()) + payload;
        }
    }
}

// A header of an HTTP request head, by name, ignoring case. "" if absent.
static std::string Header(const std::string &head, const char *name)
{
    size_t length = strlen(name), start = 0;
    while ((start = head.find("\r\n", start)) != std::string::npos) {
        start += 2;
        if (head.size() > start + length && head[start + length] == ':' &&
            strncasecmp(head.c_str() + start, name, length) == 0) {
            size_t from = head.find_first_not_of(" \t", start + length + 1);
            size_t to = head.find("\r\n", start);
            if (from == std::string::npos || from >= to) return "";
            return head.substr(from, to - from);
        }
    }
    return "";
}

// Answers the HTTP request in b->in: the page, the WebSocket, or 404.
static void ReceiveRequest(WebStream *web, Browser *b)
{
    size_t end = b->in.find("\r\n\r\n");
    if (end == std::string::npos) {
        if (b->in.size() > MAX_REQUEST) b->dead = true;
        return;
    }
    std::string head = b->in.substr(0, end);
    b->in.erase(0, end + 4);
    char method[16] = "", target[1024] = "";
    sscanf(head.c_str(), "%15s %1023s", method, target);
    std::string path(target);
    path = path.substr(0, path.find('?'));
    std::string key = Header(head, "Sec-WebSocket-Key");
    if (strcmp(method, "GET") == 0 && path == "/ws" && !key.empty()) {
        b->out += "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                  "Sec-WebSocket-Accept: " + WebSocketAccept(key) + "\r\n\r\n";
        b->webSocket = true;
        fprintf(stderr, "Web stream: browser joined from %s\n", b->address.c_str());
        ReceiveMessages(web, b);  // anything sent right behind the request
        return;
    }
    std::string body = (strcmp(method, "GET") == 0 && (path == "/" || path == "/index.html")) ? Page() : "";
    if (body.empty()) {
        b->out += "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
    } else {
        b->out += "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nCache-Control: no-store\r\n"
                  "Content-Length: " + std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n" + body;
    }
    b->closing = true;
}

static void Receive(WebStream *web, Browser *b)
{
    char buffer[65536];
    for (;;) {
        ssize_t n = recv(b->fd, buffer, sizeof(buffer), 0);
        if (n > 0) { b->in.append(buffer, (size_t)n); continue; }
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) break;
        b->dead = true;  // closed by the browser, or broken
        return;
    }
    if (b->closing) b->in.clear();  // answered already: the rest does not matter
    else if (b->webSocket) ReceiveMessages(web, b);
    else ReceiveRequest(web, b);
}

// Sends what the socket takes now; the rest waits for POLLOUT.
static void Flush(Browser *b)
{
    while (!b->out.empty() && !b->dead) {
        ssize_t n = send(b->fd, b->out.data(), b->out.size(), MSG_NOSIGNAL);
        if (n > 0) { b->out.erase(0, (size_t)n); continue; }
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) return;
        b->dead = true;
    }
    if (b->out.empty() && b->closing) b->dead = true;
}

static void Accept(WebStream *web)
{
    for (;;) {
        sockaddr_in from = {};
        socklen_t length = sizeof(from);
        int fd = accept(web->listener, (sockaddr *)&from, &length);
        if (fd < 0) return;
        if (web->browsers.size() >= MAX_CONNECTIONS) { close(fd); continue; }
        fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK);
        int one = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));  // input goes at once, not gathered up
#ifdef SO_NOSIGPIPE
        setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#endif
        char address[INET_ADDRSTRLEN] = "?";
        inet_ntop(AF_INET, &from.sin_addr, address, sizeof(address));
        Browser b;
        b.fd = fd;
        b.address = address;
        web->browsers.push_back(std::move(b));
    }
}

// A browser that left releases what it held, or the camera would fly on.
static void RemoveDead(WebStream *web)
{
    std::lock_guard<std::mutex> lock(web->mutex);
    for (Browser &b : web->browsers) {
        if (!b.dead) continue;
        for (int key : b.keys) web->input.push_back({ WebInput::KEY, key, 0 });
        for (int button : b.buttons) web->input.push_back({ WebInput::BUTTON, button, 0 });
        if (b.webSocket) fprintf(stderr, "Web stream: browser left, %s\n", b.address.c_str());
        close(b.fd);
    }
    web->browsers.erase(std::remove_if(web->browsers.begin(), web->browsers.end(),
                                       [](const Browser &b) { return b.dead; }),
                        web->browsers.end());
    UpdateFrameSize(web);
}

// ----------------------------------------------------------------------------
// Frames to the browsers
// ----------------------------------------------------------------------------
static void AppendBytes(void *context, void *data, int size)
{
    ((std::string *)context)->append((const char *)data, (size_t)size);
}

// Takes the main thread's newest frame, and encodes it if it differs from the last.
static void EncodeNewestFrame(WebStream *web)
{
    if (!web->framePending) return;
    int width, height;
    {
        std::lock_guard<std::mutex> lock(web->mutex);
        web->working.swap(web->frame);
        width = web->frameWidth;
        height = web->frameHeight;
        web->framePending = false;
    }
    // A camera that stands still renders the same picture again: nothing new to send.
    if (width == web->lastWidth && height == web->lastHeight && web->working == web->lastSent) return;
    std::string jpeg;
    stbi_write_jpg_to_func(AppendBytes, &jpeg, width, height, 4, web->working.data(), JPEG_QUALITY);
    web->newest = MessageHead(0x2, jpeg.size()) + jpeg;
    web->working.swap(web->lastSent);
    web->lastWidth = width;
    web->lastHeight = height;
    for (Browser &b : web->browsers) b.hasNewest = false;
}

static void SendFrames(WebStream *web)
{
    bool ready = false;
    int viewers = 0;
    for (Browser &b : web->browsers) {
        if (!b.webSocket || b.closing || b.dead) continue;
        viewers++;
        if (!b.hasNewest && b.inFlight < MAX_IN_FLIGHT && !web->newest.empty()) {
            b.out += web->newest;
            b.inFlight++;
            b.hasNewest = true;
        }
        if (b.inFlight < MAX_IN_FLIGHT) ready = true;
    }
    web->ready = ready;
    web->viewers = viewers;
}

static void Serve(WebStream *web)
{
    std::vector<pollfd> fds;
    while (!web->quit) {
        fds.clear();
        fds.push_back({ web->listener, POLLIN, 0 });
        fds.push_back({ web->wake[0], POLLIN, 0 });
        for (const Browser &b : web->browsers) fds.push_back({ b.fd, (short)(POLLIN | (b.out.empty() ? 0 : POLLOUT)), 0 });
        if (poll(fds.data(), fds.size(), 200) < 0 && errno != EINTR) break;
        if (fds[1].revents & POLLIN) {
            char drain[256];
            while (read(web->wake[0], drain, sizeof(drain)) > 0) {}
        }
        for (size_t i = 0; i + 2 < fds.size(); i++) {  // the browsers polled; ones accepted below come after them
            if (fds[i + 2].revents & (POLLIN | POLLHUP | POLLERR)) Receive(web, &web->browsers[i]);
        }
        if (fds[0].revents & POLLIN) Accept(web);
        EncodeNewestFrame(web);
        SendFrames(web);
        for (Browser &b : web->browsers) Flush(&b);
        RemoveDead(web);
        SendFrames(web);  // again: the counts without the browsers that left
    }
}

// ----------------------------------------------------------------------------
// web_stream.h
// ----------------------------------------------------------------------------
WebStream *OpenWebStream(int port)
{
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return NULL;
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));  // a restarted viewer gets its port back at once
    sockaddr_in address = {};
    address.sin_family = AF_INET;
    address.sin_port = htons((uint16_t)port);
    address.sin_addr.s_addr = htonl(INADDR_ANY);  // every interface: the point is a browser on another computer
    if (bind(fd, (sockaddr *)&address, sizeof(address)) < 0 || listen(fd, 16) < 0) {
        fprintf(stderr, "Web stream: cannot listen on port %d: %s\n", port, strerror(errno));
        close(fd);
        return NULL;
    }
    fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK);

    WebStream *web = new WebStream;
    web->listener = fd;
    if (pipe(web->wake) < 0) {
        close(fd);
        delete web;
        return NULL;
    }
    fcntl(web->wake[0], F_SETFL, fcntl(web->wake[0], F_GETFL) | O_NONBLOCK);
    fcntl(web->wake[1], F_SETFL, fcntl(web->wake[1], F_GETFL) | O_NONBLOCK);
    stbi_flip_vertically_on_write(1);  // the frames come bottom row first

    // The addresses to open: every IPv4 interface but loopback. The window shows the first.
    fprintf(stderr, "Web stream: open one of these in a browser\n");
    ifaddrs *interfaces = NULL;
    if (getifaddrs(&interfaces) == 0) {
        for (ifaddrs *i = interfaces; i; i = i->ifa_next) {
            if (!i->ifa_addr || i->ifa_addr->sa_family != AF_INET) continue;
            char ip[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, &((sockaddr_in *)i->ifa_addr)->sin_addr, ip, sizeof(ip));
            std::string url = "http://" + std::string(ip) + ":" + std::to_string(port) + "/";
            fprintf(stderr, "    %-28s (%s)\n", url.c_str(), i->ifa_name);
            if (web->url.empty() && strcmp(ip, "127.0.0.1") != 0) web->url = url;
        }
        freeifaddrs(interfaces);
    }
    if (web->url.empty()) web->url = "http://localhost:" + std::to_string(port) + "/";
    web->thread = std::thread(Serve, web);
    return web;
}

void CloseWebStream(WebStream *web)
{
    if (!web) return;
    web->quit = true;
    char byte = 0;
    (void)!write(web->wake[1], &byte, 1);
    web->thread.join();
    for (Browser &b : web->browsers) close(b.fd);
    close(web->listener);
    close(web->wake[0]);
    close(web->wake[1]);
    delete web;
}

void ReceiveWebInput(WebStream *web, std::vector<WebInput> *events, int *width, int *height)
{
    events->clear();
    std::lock_guard<std::mutex> lock(web->mutex);
    std::set<std::pair<int, int>> pressed;  // (type, code) pressed in this batch
    size_t n = 0;
    for (; n < web->input.size(); n++) {
        const WebInput &e = web->input[n];
        if (e.type == WebInput::KEY || e.type == WebInput::BUTTON) {
            std::pair<int, int> id(e.type, e.a);
            if (e.b) pressed.insert(id);
            else if (pressed.count(id)) break;  // a tap within one frame: the release goes next frame
        }
        events->push_back(e);
    }
    web->input.erase(web->input.begin(), web->input.begin() + (long)n);
    *width = web->width;
    *height = web->height;
}

bool WebWantsFrame(const WebStream *web)
{
    return web->ready && !web->framePending;
}

void SendWebFrame(WebStream *web, const unsigned char *rgba, int width, int height)
{
    {
        std::lock_guard<std::mutex> lock(web->mutex);
        web->frame.assign(rgba, rgba + (size_t)width*height*4);
        web->frameWidth = width;
        web->frameHeight = height;
        web->framePending = true;
    }
    char byte = 1;
    (void)!write(web->wake[1], &byte, 1);
}

int WebViewerCount(const WebStream *web) { return web->viewers; }

const char *WebStreamUrl(const WebStream *web) { return web->url.c_str(); }

// ----------------------------------------------------------------------------
// The page. It draws each JPEG on a canvas fitted to the window, and sends the
// keys, buttons, mouse motion and wheel over the WebSocket, in raylib's codes.
// ----------------------------------------------------------------------------
static const char *Page()
{
    return R"PAGE(<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gaussian splat viewer</title>
<style>
  html, body { margin: 0; height: 100%; overflow: hidden; background: #14141a; }
  canvas { display: block; width: 100vw; height: 100vh; object-fit: contain; outline: none; }
  .box { position: fixed; padding: 4px 8px; border-radius: 4px; background: rgba(0, 0, 0, 0.6); color: #ddd;
         font: 13px/1.4 system-ui, sans-serif; pointer-events: none; }
  #status { top: 8px; right: 8px; }
  #help { bottom: 8px; right: 8px; text-align: right; }
  #wait { top: 50%; left: 50%; transform: translate(-50%, -50%); font-size: 18px; padding: 12px 20px; }
</style>
</head>
<body>
<canvas id="view"></canvas>
<div id="wait" class="box">Connecting to the viewer&hellip;</div>
<div id="status" class="box"></div>
<div id="help" class="box">Drag to look, middle-drag to pan, WASD to fly, wheel for speed, 1&ndash;7 for views.<br>
Hold Space to fly slowly (Ctrl+W would close this tab). Add ?res=0.5 to the address on a slow link.</div>
<script>
'use strict';
const canvas = document.getElementById('view'), ctx = canvas.getContext('2d');
const waitBox = document.getElementById('wait'), statusBox = document.getElementById('status');
const RES = Number(new URLSearchParams(location.search).get('res')) || 1;  // frame pixels per CSS pixel
const BUTTONS = [0, 2, 1];  // the browser's left, middle, right -> raylib's left 0, right 1, middle 2
// KeyboardEvent.code -> raylib's key code. Ctrl is not sent, as Ctrl+W closes the tab: Space stands in for it.
const KEYS = { BracketLeft: 91, BracketRight: 93, Equal: 61, Minus: 45, NumpadAdd: 334, NumpadSubtract: 333,
               ShiftLeft: 340, ShiftRight: 344, Space: 341 };
function rayKey(code) {
  if (/^Key[A-Z]$/.test(code)) return code.charCodeAt(3);    // KEY_A = 65 ... KEY_Z = 90
  if (/^Digit[0-9]$/.test(code)) return code.charCodeAt(5);  // KEY_ZERO = 48 ... KEY_NINE = 57
  return KEYS[code] || 0;
}

let ws = null, drawing = Promise.resolve();
const keys = new Set(), buttons = new Set();
let moveX = 0, moveY = 0, wheel = 0, frames = 0, bytes = 0;

function send(message) { if (ws && ws.readyState === WebSocket.OPEN) ws.send(message); }
function sendSize() { send(`s ${Math.round(innerWidth*RES)} ${Math.round(innerHeight*RES)}`); }

function connect() {
  ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  ws.binaryType = 'blob';
  ws.onopen = () => { waitBox.textContent = 'Waiting for the first picture…'; sendSize(); };
  ws.onmessage = event => {
    if (typeof event.data === 'string') return;
    const jpeg = event.data;
    drawing = drawing.then(async () => {  // one at a time, in order
      const picture = await createImageBitmap(jpeg);
      if (canvas.width !== picture.width || canvas.height !== picture.height) {
        canvas.width = picture.width;
        canvas.height = picture.height;
      }
      ctx.drawImage(picture, 0, 0);
      picture.close();
      frames++;
      bytes += jpeg.size;
      waitBox.style.display = 'none';
    }).catch(() => {}).then(() => send('a'));  // drawn (or not decodable): the viewer may send the next
  };
  ws.onclose = () => {
    keys.clear();  // the viewer released them when the socket closed
    buttons.clear();
    waitBox.style.display = '';
    waitBox.textContent = 'The viewer is not reachable, retrying…';
    setTimeout(connect, 2000);
  };
}
connect();

addEventListener('keydown', event => {
  const key = rayKey(event.code);
  if (!key || event.ctrlKey || event.metaKey || event.altKey) return;  // the browser's shortcuts stay its own
  event.preventDefault();
  if (!keys.has(key)) { keys.add(key); send(`k ${key} 1`); }
});
addEventListener('keyup', event => {
  const key = rayKey(event.code);
  if (key && keys.delete(key)) { event.preventDefault(); send(`k ${key} 0`); }
});
canvas.addEventListener('contextmenu', event => event.preventDefault());
canvas.addEventListener('mousedown', event => {
  const button = BUTTONS[event.button];
  if (button === undefined) return;
  event.preventDefault();  // no middle-click scrolling, no selecting
  if (!buttons.has(button)) { buttons.add(button); send(`b ${button} 1`); }
});
addEventListener('mouseup', event => {
  const button = BUTTONS[event.button];
  if (button !== undefined && buttons.delete(button)) send(`b ${button} 0`);
});
// The picture is fitted into the window (object-fit: contain): frame pixels per CSS pixel
function frameScale() {
  if (!canvas.width) return RES;
  return 1/Math.min(canvas.clientWidth/canvas.width, canvas.clientHeight/canvas.height);
}
addEventListener('mousemove', event => {
  if (!buttons.size) return;  // the viewer uses motion only while a button is held
  const scale = frameScale();
  moveX += event.movementX*scale;
  moveY += event.movementY*scale;
});
canvas.addEventListener('wheel', event => {
  event.preventDefault();
  wheel -= event.deltaY/[100, 3, 1][event.deltaMode];  // a notch: ~100 pixels, 3 lines or a page
}, { passive: false });
function releaseAll() {
  for (const key of keys) send(`k ${key} 0`);
  for (const button of buttons) send(`b ${button} 0`);
  keys.clear();
  buttons.clear();
}
addEventListener('blur', releaseAll);  // keys released in another window never reach this page
document.addEventListener('visibilitychange', () => { if (document.hidden) releaseAll(); });

// Motion and wheel go once per animation frame, summed, in whole units
function flush() {
  const dx = Math.trunc(moveX), dy = Math.trunc(moveY), notches = Math.trunc(wheel);
  if (dx || dy) { send(`m ${dx} ${dy}`); moveX -= dx; moveY -= dy; }
  if (notches) { send(`w ${notches}`); wheel -= notches; }
  requestAnimationFrame(flush);
}
requestAnimationFrame(flush);

let resizeTimer = 0;
addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(sendSize, 150); });
setInterval(() => {
  statusBox.textContent = (!ws || ws.readyState !== WebSocket.OPEN) ? 'disconnected'
                        : frames ? `${frames} fps, ${Math.round(bytes/frames/1024)} KB a frame` : 'still picture';
  frames = 0;
  bytes = 0;
}, 1000);
</script>
</body>
</html>
)PAGE";
}

#else  // _WIN32: not ported (POSIX sockets)

WebStream *OpenWebStream(int)
{
    fprintf(stderr, "Web stream: --serve is not available on Windows\n");
    return NULL;
}
void CloseWebStream(WebStream *) {}
void ReceiveWebInput(WebStream *, std::vector<WebInput> *events, int *width, int *height)
{
    events->clear();
    *width = *height = 0;
}
bool WebWantsFrame(const WebStream *) { return false; }
void SendWebFrame(WebStream *, const unsigned char *, int, int) {}
int WebViewerCount(const WebStream *) { return 0; }
const char *WebStreamUrl(const WebStream *) { return ""; }

#endif
