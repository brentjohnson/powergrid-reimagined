cargo build --bin powergrid-client
cargo build --bin powergrid-server

trap 'kill $(jobs -p) 2>/dev/null' EXIT

cargo run --bin powergrid-client &
sleep 1
cargo run --bin powergrid-client &
sleep 1
cargo run --bin powergrid-client &
sleep 1
RUST_LOG=info cargo run --bin powergrid-server
