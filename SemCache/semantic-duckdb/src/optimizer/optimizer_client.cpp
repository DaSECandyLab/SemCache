#include "optimizer/optimizer_client.hpp"
#include <chrono>
#include <cstring>
#include <cerrno>
#include <algorithm>
#include <sys/socket.h>
#include <sys/un.h>
#include <poll.h>
#include <unistd.h>

namespace semantic {
namespace {
struct Socket {
	int fd;
	Socket() : fd(socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0)) {
		if (fd < 0)
			throw ValidationError("optimizer socket creation failed");
	}
	~Socket() {
		close(fd);
	}
};
using Clock = std::chrono::steady_clock;
void Wait(int fd, short events, Clock::time_point deadline) {
	for (;;) {
		auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
		if (remaining <= 0)
			throw ValidationError("optimizer RPC deadline exceeded");
		pollfd item {fd, events, 0};
		auto ready = poll(&item, 1, int(remaining));
		if (ready > 0)
			return;
		if (ready < 0 && errno != EINTR)
			throw ValidationError("optimizer socket poll failed");
	}
}
void Transfer(int fd, char *buffer, size_t length, bool writing, Clock::time_point deadline) {
	// Partial reads/writes are normal on stream sockets; continue until the exact
	// frame length is transferred or the single RPC deadline expires.
	while (length) {
		Wait(fd, writing ? POLLOUT : POLLIN, deadline);
		auto count = writing ? send(fd, buffer, length, MSG_NOSIGNAL) : recv(fd, buffer, length, 0);
		if (count < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK))
			continue;
		if (count <= 0)
			throw ValidationError("optimizer RPC truncated frame or socket failure");
		buffer += count;
		length -= size_t(count);
	}
}
}
UnixSocketOptimizerClient::UnixSocketOptimizerClient(std::string path_p, int timeout_ms_p)
    : path(std::move(path_p)), timeout_ms(timeout_ms_p) {
	if (path.empty() || path.find('\0') != std::string::npos || path.size() >= sizeof(sockaddr_un::sun_path))
		throw ValidationError("optimizer socket path is empty, invalid or too long");
	if (timeout_ms <= 0 || timeout_ms > 600000)
		throw ValidationError("optimizer timeout must be in 1..600000 ms");
}
llm_detail::Json UnixSocketOptimizerClient::Optimize(const llm_detail::Json &query) {
	// One connection carries one length-prefixed request and one response. A hard
	// size cap prevents a local peer from exhausting planner memory.
	auto request = query.dump();
	constexpr size_t limit = 16 * 1024 * 1024;
	if (request.size() > limit)
		throw ValidationError("optimizer QueryIR exceeds 16 MiB");
	Socket connection;
	sockaddr_un address {};
	address.sun_family = AF_UNIX;
	std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
	auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
	if (connect(connection.fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) < 0) {
		// Linux may return EAGAIN for a full Unix socket listen queue. Fail without retry.
		if (errno != EINPROGRESS)
			throw ValidationError("optimizer socket connect failed: " + std::string(std::strerror(errno)));
		Wait(connection.fd, POLLOUT, deadline);
		int error = 0;
		socklen_t size = sizeof(error);
		if (getsockopt(connection.fd, SOL_SOCKET, SO_ERROR, &error, &size) || error)
			throw ValidationError("optimizer socket connect failed");
	}
	char header[4];
	for (size_t i = 0; i < 4; ++i)
		header[i] = char((request.size() >> (24 - 8 * i)) & 255);
	Transfer(connection.fd, header, 4, true, deadline);
	Transfer(connection.fd, &request[0], request.size(), true, deadline);
	Transfer(connection.fd, header, 4, false, deadline);
	size_t length = 0;
	for (auto byte : header)
		length = (length << 8) | static_cast<unsigned char>(byte);
	if (!length || length > limit)
		throw ValidationError("optimizer PlanSpec frame must be 1..16 MiB");
	std::string response(length, '\0');
	Transfer(connection.fd, &response[0], length, false, deadline);
	return llm_detail::ParseJSON(response);
}
}
