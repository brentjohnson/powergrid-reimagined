FROM rust:slim-trixie AS builder

WORKDIR /app
COPY Cargo.toml Cargo.lock* ./
COPY crates/powergrid-core crates/powergrid-core
COPY crates/powergrid-server crates/powergrid-server
RUN mkdir -p crates/powergrid-client/src && echo 'fn main(){}' > crates/powergrid-client/src/main.rs
COPY crates/powergrid-client/Cargo.toml crates/powergrid-client/Cargo.toml
RUN mkdir -p crates/powergrid-bot/src && echo 'fn main(){}' > crates/powergrid-bot/src/main.rs
COPY crates/powergrid-bot/Cargo.toml crates/powergrid-bot/Cargo.toml

RUN cargo build --release -p powergrid-server

FROM debian:trixie-slim

WORKDIR /app
COPY --from=builder /app/target/release/powergrid-server ./

ENV PORT=3000

EXPOSE 3000

CMD ["./powergrid-server"]
