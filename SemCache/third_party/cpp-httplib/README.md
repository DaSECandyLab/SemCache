# cpp-httplib 0.18.7

MIT-licensed single header used privately by the semantic HTTP/HTTPS adapter.
HTTPS uses OpenSSL with certificate and hostname verification. Builds do not
download dependencies.

Upstream pinned sources:

- https://raw.githubusercontent.com/yhirose/cpp-httplib/v0.18.7/httplib.h
- https://raw.githubusercontent.com/yhirose/cpp-httplib/v0.18.7/LICENSE

Original upstream SHA-256 (before the local guard below):

```text
4770f8ea3d3fcd27b0b713c783d9b53bce3baddbd5adad13714011df96142526  httplib.h
4b45cbe16d7b71b89ae6127e26e0d90a029198ca5e958ad8e3d0b8bbed364d8b  LICENSE
```

Local patch: the `SSLPeerCouldBeClosed_` resend is guarded by
`CPPHTTPLIB_NO_SSL_PEER_RETRY`. The semantic adapter defines this macro because
an inference POST must never be silently repeated after it may have reached the
server. Dead sockets detected before a send may be replaced, but a failed send
returns an error. Model lanes reuse keep-alive connections; neither redirects
nor application retries are enabled. The upstream license is unchanged.
