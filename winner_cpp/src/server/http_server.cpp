#include "server/http_server.h"
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <cstdio>
#include <sstream>
#include <cerrno>

namespace winner {

namespace {
bool send_all(int fd, const char* data, size_t size) {
    while (size > 0) {
        const ssize_t sent = ::send(fd, data, size, 0);
        if (sent < 0 && errno == EINTR) continue;
        if (sent <= 0) return false;
        data += sent;
        size -= static_cast<size_t>(sent);
    }
    return true;
}

std::string json_escape(const std::string& value) {
    std::ostringstream output;
    for (const unsigned char ch : value) {
        switch (ch) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (ch < 0x20) {
                    const char hex[] = "0123456789abcdef";
                    output << "\\u00" << hex[ch >> 4] << hex[ch & 0x0f];
                } else output << static_cast<char>(ch);
        }
    }
    return output.str();
}
}

void LockFreeTokenQueue::push(ChatChunk c) {
    std::lock_guard<std::mutex> lk(mu_);
    q_.push(std::move(c));
    cv_.notify_one();
}
bool LockFreeTokenQueue::pop(ChatChunk& out) {
    std::lock_guard<std::mutex> lk(mu_);
    if (q_.empty()) return false;
    out = std::move(q_.front()); q_.pop();
    return true;
}
void LockFreeTokenQueue::wait_pop(ChatChunk& out) {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&]{ return closed_ || !q_.empty(); });
    if (q_.empty()) { out.done = true; return; }
    out = std::move(q_.front()); q_.pop();
}
void LockFreeTokenQueue::close() {
    std::lock_guard<std::mutex> lk(mu_);
    closed_ = true;
    cv_.notify_all();
}

HttpServer::HttpServer() = default;
HttpServer::~HttpServer() { stop(); }

bool HttpServer::start(const std::string& host, int port, bool background) {
    if (running_.load() || port < 1 || port > 65535) return false;
    in_addr parsed{};
    if (::inet_pton(AF_INET, host.c_str(), &parsed) != 1) return false;
    host_ = host; port_ = port; running_ = true;
    if (background) {
        accept_thread_ = std::thread(&HttpServer::accept_loop, this);
        return true;
    }
    accept_loop();
    return true;
}

void HttpServer::stop() {
    running_ = false;
    const int fd = server_fd_.exchange(-1);
    if (fd >= 0) {
        ::shutdown(fd, SHUT_RDWR);
        ::close(fd);
    }
    tokens_.close();
    if (accept_thread_.joinable()) accept_thread_.join();
}

void HttpServer::set_submit_handler(std::function<uint64_t(const ChatRequest&)> fn) {
    submit_ = std::move(fn);
}

void HttpServer::accept_loop() {
    int srv = ::socket(AF_INET, SOCK_STREAM, 0);
    if (srv < 0) { running_ = false; return; }
    server_fd_.store(srv);
    int yes = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port_);
    if (::inet_pton(AF_INET, host_.c_str(), &addr.sin_addr) != 1 ||
        bind(srv, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0 ||
        listen(srv, 64) < 0) {
        const int owned = server_fd_.exchange(-1);
        if (owned >= 0) close(owned);
        running_ = false;
        return;
    }
    printf("[WINNER-HTTP] listening on %s:%d  (POST /v1/chat/completions)\n", host_.c_str(), port_);
    while (running_) {
        fd_set fds; FD_ZERO(&fds); FD_SET(srv, &fds);
        timeval tv{1, 0};
        if (select(srv + 1, &fds, nullptr, nullptr, &tv) <= 0) continue;
        int cli = accept(srv, nullptr, nullptr);
        if (cli >= 0) {
            // Phase-1: handle inline (production: thread pool)
            handle_client(cli);
            close(cli);
        }
    }
    int expected = srv;
    if (server_fd_.compare_exchange_strong(expected, -1)) close(srv);
}

void HttpServer::handle_client(int fd) {
    char buf[8192];
    ssize_t n = read(fd, buf, sizeof(buf) - 1);
    if (n <= 0) return;
    buf[n] = 0;
    std::string req(buf);
    const size_t request_line_end = req.find("\r\n");
    const std::string request_line = req.substr(0, request_line_end);
    const bool is_chat = request_line == "POST /v1/chat/completions HTTP/1.1" ||
                         request_line == "POST /chat/completions HTTP/1.1";

    if (!is_chat) {
        const char* resp = "HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\nContent-Length: 21\r\nConnection: close\r\n\r\n{\"error\":\"not found\"}";
        send_all(fd, resp, strlen(resp));
        return;
    }

    // Minimal parse: ignore body details in Phase-1
    ChatRequest cr;
    cr.stream = req.find("\"stream\":true") != std::string::npos
             || req.find("\"stream\": true") != std::string::npos;
    cr.max_tokens = 64;
    cr.messages.push_back({"user", "Hello from winner.cpp"});

    uint64_t id = submit_ ? submit_(cr) : 1;

    if (cr.stream) {
        const char* hdr =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: keep-alive\r\n\r\n";
        if (!send_all(fd, hdr, strlen(hdr))) return;
        // Emit a few synthetic tokens (model thread would push real ones)
        for (const char* tok : {"Hello", " from", " winner", ".cpp"}) {
            std::string chunk = sse_chat_chunk(id, tok, false);
            if (!send_all(fd, chunk.data(), chunk.size())) return;
        }
        std::string done = sse_chat_chunk(id, "", true);
        if (!send_all(fd, done.data(), done.size())) return;
        send_all(fd, "data: [DONE]\n\n", 14);
    } else {
        std::string body = json_chat_complete(id, "Hello from winner.cpp");
        std::ostringstream oss;
        oss << "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
            << body.size() << "\r\n\r\n" << body;
        auto s = oss.str();
        send_all(fd, s.data(), s.size());
    }
}

std::string sse_chat_chunk(uint64_t id, const std::string& delta, bool done) {
    std::ostringstream o;
    o << "data: {\"id\":\"chatcmpl-" << id
      << "\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{";
    if (!done) o << "\"content\":\"" << json_escape(delta) << "\"";
    else o << "\"content\":\"\"";
    o << "},\"finish_reason\":" << (done ? "\"stop\"" : "null") << "}]}\n\n";
    return o.str();
}

std::string json_chat_complete(uint64_t id, const std::string& content) {
    std::ostringstream o;
    o << "{\"id\":\"chatcmpl-" << id
      << "\",\"object\":\"chat.completion\",\"choices\":[{\"index\":0,\"message\":{"
      << "\"role\":\"assistant\",\"content\":\"" << json_escape(content)
      << "\"},\"finish_reason\":\"stop\"}]}";
    return o.str();
}

} // namespace winner
