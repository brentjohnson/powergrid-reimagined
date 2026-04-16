cargo build --bin powergrid-client
cargo build --bin powergrid-server

trap 'kill $(jobs -p) 2>/dev/null' EXIT

cargo run --bin powergrid-client -- --name brent --color red --url ws://localhost:3000/ws &
sleep 1
cargo run --bin powergrid-client -- --name brad --color blue --url ws://localhost:3000/ws &
sleep 1
cargo run --bin powergrid-client -- --name nick --color green --url ws://localhost:3000/ws &
sleep 1
RUST_LOG=info cargo run --bin powergrid-server
