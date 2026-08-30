#include "vlb_kernel.h"

#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
using socket_t = SOCKET;
static constexpr socket_t invalid_socket_v = INVALID_SOCKET;
static void close_socket(socket_t s) { closesocket(s); }
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
using socket_t = int;
static constexpr socket_t invalid_socket_v = -1;
static void close_socket(socket_t s) { close(s); }
#endif

namespace {

std::atomic<bool> running{true};

void on_signal(int) {
    running.store(false);
}

std::string response(int code, const char* status, const std::string& body) {
    std::ostringstream out;
    out << "HTTP/1.1 " << code << ' ' << status << "\r\n"
        << "Content-Type: application/json; charset=utf-8\r\n"
        << "Content-Length: " << body.size() << "\r\n"
        << "Connection: close\r\n"
        << "X-VLB-Server: native-v1\r\n\r\n"
        << body;
    return out.str();
}

std::string route(const std::string& request) {
    const auto line_end = request.find("\r\n");
    const std::string first = request.substr(0, line_end);

    if (first.rfind("GET /healthz ", 0) == 0) {
        return response(200, "OK",
            std::string("{\"ok\":true,\"server\":\"VLB_NATIVE_SERVER_V1\",\"kernel\":\"")
            + vlb::kernel_backend()
            + "\",\"external_llm_runtime\":false}");
    }

    if (first.rfind("GET /v1/runtime ", 0) == 0) {
        return response(200, "OK",
            std::string("{\"runtime\":\"VLB_NATIVE_RUNTIME_V1\",\"kernel\":\"")
            + vlb::kernel_backend()
            + "\",\"model_forward_ready\":false,\"kr100_certified\":false,"
              "\"note\":\"Native server/kernel milestone only; full Gemma forward is not yet certified\"}");
    }

    if (first.rfind("POST /v1/completions ", 0) == 0 ||
        first.rfind("POST /v1/chat/completions ", 0) == 0) {
        // Deliberately reject until the complete VLB-owned model executor exists.
        // Falling back to another runtime would invalidate the proof.
        return response(503, "Service Unavailable",
            "{\"error\":\"VLB_NATIVE_MODEL_RUNTIME_NOT_YET_CERTIFIED\","
            "\"fallback_used\":false}");
    }

    return response(404, "Not Found", "{\"error\":\"not_found\"}");
}

int parse_port(int argc, char** argv) {
    int port = 8787;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--port" && i + 1 < argc) {
            port = std::stoi(argv[++i]);
        }
    }
    if (port <= 0 || port > 65535) throw std::runtime_error("invalid port");
    return port;
}

} // namespace

int main(int argc, char** argv) {
    try {
#ifdef _WIN32
        WSADATA data{};
        if (WSAStartup(MAKEWORD(2, 2), &data) != 0) {
            throw std::runtime_error("WSAStartup failed");
        }
#endif
        std::signal(SIGINT, on_signal);
        std::signal(SIGTERM, on_signal);
        const int port = parse_port(argc, argv);

        socket_t server = ::socket(AF_INET, SOCK_STREAM, 0);
        if (server == invalid_socket_v) throw std::runtime_error("socket() failed");

#ifndef _WIN32
        int yes = 1;
        setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
#endif

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
        addr.sin_port = htons(static_cast<std::uint16_t>(port));

        if (::bind(server, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
            close_socket(server);
            throw std::runtime_error(std::string("bind() failed: ") + std::strerror(errno));
        }
        if (::listen(server, 16) != 0) {
            close_socket(server);
            throw std::runtime_error("listen() failed");
        }

        std::cout << "VLB native server listening on 0.0.0.0:" << port
                  << " kernel=" << vlb::kernel_backend() << "\n";

        while (running.load()) {
            sockaddr_in client_addr{};
#ifdef _WIN32
            int len = sizeof(client_addr);
#else
            socklen_t len = sizeof(client_addr);
#endif
            socket_t client = ::accept(server, reinterpret_cast<sockaddr*>(&client_addr), &len);
            if (client == invalid_socket_v) {
                if (!running.load()) break;
                continue;
            }

            std::string request;
            char buffer[8192];
#ifdef _WIN32
            const int got = recv(client, buffer, static_cast<int>(sizeof(buffer)), 0);
#else
            const ssize_t got = recv(client, buffer, sizeof(buffer), 0);
#endif
            if (got > 0) request.assign(buffer, buffer + got);
            const std::string payload = route(request);
#ifdef _WIN32
            send(client, payload.data(), static_cast<int>(payload.size()), 0);
#else
            send(client, payload.data(), payload.size(), 0);
#endif
            close_socket(client);
        }

        close_socket(server);
#ifdef _WIN32
        WSACleanup();
#endif
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "vlb-server error: " << exc.what() << "\n";
        return 1;
    }
}
