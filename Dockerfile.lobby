FROM rust:slim-trixie AS builder

WORKDIR /app
COPY Cargo.toml Cargo.lock* ./

# Real transitive dependencies of powergrid-lobby
COPY crates/powergrid-core crates/powergrid-core
COPY crates/powergrid-session crates/powergrid-session
COPY crates/powergrid-bot-strategy crates/powergrid-bot-strategy
COPY crates/powergrid-lobby crates/powergrid-lobby

COPY assets assets

# Stub out workspace members that are not dependencies of powergrid-lobby
# so Cargo can load the workspace manifest without their full source.
COPY crates/powergrid-bot/Cargo.toml crates/powergrid-bot/Cargo.toml
RUN mkdir -p crates/powergrid-bot/src && echo 'fn main(){}' > crates/powergrid-bot/src/main.rs
COPY crates/powergrid-server/Cargo.toml crates/powergrid-server/Cargo.toml
RUN mkdir -p crates/powergrid-server/src && echo '' > crates/powergrid-server/src/lib.rs && echo 'fn main(){}' > crates/powergrid-server/src/main.rs
COPY crates/powergrid-client/Cargo.toml crates/powergrid-client/Cargo.toml
RUN mkdir -p crates/powergrid-client/src && echo 'fn main(){}' > crates/powergrid-client/src/main.rs

RUN cargo build --release -p powergrid-lobby

FROM debian:trixie-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/powergrid-lobby ./

ENV PORT=3000

EXPOSE 3000

CMD ["./powergrid-lobby"]
