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
COPY crates/powergrid-client/Cargo.toml crates/powergrid-client/Cargo.toml
RUN mkdir -p crates/powergrid-client/src && echo 'fn main(){}' > crates/powergrid-client/src/main.rs
COPY crates/powergrid-py/Cargo.toml crates/powergrid-py/Cargo.toml
RUN mkdir -p crates/powergrid-py/src && echo '' > crates/powergrid-py/src/lib.rs
COPY crates/powergrid-maptool/Cargo.toml crates/powergrid-maptool/Cargo.toml
RUN mkdir -p crates/powergrid-maptool/src && echo 'fn main(){}' > crates/powergrid-maptool/src/main.rs
COPY crates/powergrid-netviz/Cargo.toml crates/powergrid-netviz/Cargo.toml
RUN mkdir -p crates/powergrid-netviz/src && echo 'fn main(){}' > crates/powergrid-netviz/src/main.rs

RUN cargo build --release -p powergrid-lobby

FROM debian:trixie-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/powergrid-lobby ./

ENV PORT=3000

EXPOSE 3000

CMD ["./powergrid-lobby"]
