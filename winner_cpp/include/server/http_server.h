#pragma once
/**
 * Low-latency HTTP + SSE streaming
 * OpenAI-compatible: POST /v1/chat/completions
 * Lock-free-ish queue between HTTP threads and model thread.
 */
#include "winner.h"
#include "sched/batching.h"
#include <string>
#include <functional>
#include <atomic>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>

namespace winner {

struct ChatMessage {
    std::string role;
    std::string content;
};

struct ChatRequest {
    uint64_t id = 0;
    std::vector<ChatMessage> messages;
    int max_tokens = 128;
    float temperature = 0.7f;
    bool stream = true;
};

struct ChatChunk {
    uint64_t req_id = 0;
    std::string delta;     // token text
    bool done = false;
};

class LockFreeTokenQueue {
public:
    void push(ChatChunk c);
    bool pop(ChatChunk& out);          // non-blocking
    void wait_pop(ChatChunk& out);     // blocking
    void close();
private:
    std::mutex mu_;
    std::condition_variable cv_;
    std::queue<ChatChunk> q_;
    bool closed_ = false;
};

class HttpServer {
public:
    HttpServer();
    ~HttpServer();

    // Bind and start (blocking or background)
    bool start(const std::string& host, int port, bool background = true);
    void stop();

    // Wire to continuous batcher / model thread
    void set_submit_handler(std::function<uint64_t(const ChatRequest&)> fn);
    LockFreeTokenQueue& token_queue() { return tokens_; }

private:
    std::string host_ = "127.0.0.1";
    int port_ = 8080;
    std::atomic<bool> running_{false};
    std::atomic<int> server_fd_{-1};
    std::thread accept_thread_;
    std::function<uint64_t(const ChatRequest&)> submit_;
    LockFreeTokenQueue tokens_;
    // Phase-1: minimal TCP accept loop; production would use uWebSockets
    void accept_loop();
    void handle_client(int fd);
};

// Helpers to build OpenAI SSE payloads
std::string sse_chat_chunk(uint64_t id, const std::string& delta, bool done);
std::string json_chat_complete(uint64_t id, const std::string& content);

} // namespace winner
