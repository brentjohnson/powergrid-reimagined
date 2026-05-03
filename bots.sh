cargo build

trap 'kill $(jobs -p) 2>/dev/null' EXIT

RUST_LOG=info cargo run --bin powergrid-server &
sleep 1

cargo run --bin powergrid-client -- --windowed --name "brent" --color "red" --server localhost &
client_pid=$!
sleep 3

cargo run --bin powergrid-bot -- --name "brad-bot" --color "blue" --server localhost &
cargo run --bin powergrid-bot -- --name "nick-bot" --color "green" --server localhost &
cargo run --bin powergrid-bot -- --name "jamie-bot" --color "yellow" --server localhost &

wait $client_pid
