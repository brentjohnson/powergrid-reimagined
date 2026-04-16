use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Raw TOML-deserializable map format.
#[derive(Debug, Deserialize)]
pub struct MapData {
    pub name: String,
    pub regions: Vec<String>,
    /// Relative path to the board image (e.g. "germany.png"), resolved from the TOML file's directory.
    #[serde(default)]
    pub image: Option<String>,
    pub cities: Vec<CityData>,
    pub connections: Vec<ConnectionData>,
    #[serde(default)]
    pub resource_slots: Vec<ResourceSlotData>,
    #[serde(default)]
    pub turn_order_slots: Vec<TurnOrderSlotData>,
}

/// Raw TOML entry for a single resource market slot position.
#[derive(Debug, Deserialize)]
pub struct ResourceSlotData {
    pub resource: String,
    pub index: usize,
    /// x-position as a fraction of the map image width (0.0–1.0).
    pub x: f32,
    /// y-position as a fraction of the map image height (0.0–1.0).
    pub y: f32,
}

/// Raw TOML entry for a turn order position space on the board.
#[derive(Debug, Deserialize)]
pub struct TurnOrderSlotData {
    /// 0-based position index (0 = first place, 5 = last place).
    pub index: usize,
    /// x-position as a fraction of the map image width (0.0–1.0).
    pub x: f32,
    /// y-position as a fraction of the map image height (0.0–1.0).
    pub y: f32,
}

#[derive(Debug, Deserialize)]
pub struct CityData {
    pub id: String,
    pub name: String,
    pub region: String,
    #[serde(default)]
    pub x: Option<f32>,
    #[serde(default)]
    pub y: Option<f32>,
}

#[derive(Debug, Deserialize)]
pub struct ConnectionData {
    pub from: String,
    pub to: String,
    pub cost: u32,
}

/// Runtime map representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Map {
    pub name: String,
    pub regions: Vec<String>,
    pub cities: HashMap<String, City>,
    /// Adjacency: city_id → list of (neighbor_id, edge_cost).
    pub edges: HashMap<String, Vec<(String, u32)>>,
    /// Positions of resource market slots, ordered by resource and index.
    pub resource_slots: Vec<ResourceSlot>,
    /// Positions of the turn order spaces on the board (up to 6).
    pub turn_order_slots: Vec<TurnOrderSlot>,
}

/// A single resource market slot with its fractional position on the map image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceSlot {
    pub resource: String,
    pub index: usize,
    pub x: f32,
    pub y: f32,
}

/// A single turn order space on the board with its fractional position on the map image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnOrderSlot {
    /// 0-based position index (0 = first place, 5 = last place).
    pub index: usize,
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct City {
    pub id: String,
    pub name: String,
    pub region: String,
    /// Players who have built here (max 3 in base game).
    pub owners: Vec<crate::types::PlayerId>,
    /// Fractional x position on the map image (0.0–1.0). None if not yet placed.
    pub x: Option<f32>,
    /// Fractional y position on the map image (0.0–1.0). None if not yet placed.
    pub y: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConnectionEdge {
    pub from: String,
    pub to: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConnectionNetwork {
    pub route_cost: u32,
    pub edges: Vec<ConnectionEdge>,
}

impl Map {
    pub fn from_data(data: MapData) -> Self {
        let mut cities = HashMap::new();
        for c in data.cities {
            cities.insert(
                c.id.clone(),
                City {
                    id: c.id,
                    name: c.name,
                    region: c.region,
                    owners: Vec::new(),
                    x: c.x,
                    y: c.y,
                },
            );
        }

        let mut edges: HashMap<String, Vec<(String, u32)>> = HashMap::new();
        for conn in data.connections {
            edges
                .entry(conn.from.clone())
                .or_default()
                .push((conn.to.clone(), conn.cost));
            edges
                .entry(conn.to.clone())
                .or_default()
                .push((conn.from.clone(), conn.cost));
        }

        let resource_slots = data
            .resource_slots
            .into_iter()
            .map(|s| ResourceSlot {
                resource: s.resource,
                index: s.index,
                x: s.x,
                y: s.y,
            })
            .collect();

        let turn_order_slots = data
            .turn_order_slots
            .into_iter()
            .map(|s| TurnOrderSlot {
                index: s.index,
                x: s.x,
                y: s.y,
            })
            .collect();

        Self {
            name: data.name,
            regions: data.regions,
            cities,
            edges,
            resource_slots,
            turn_order_slots,
        }
    }

    pub fn load(toml_str: &str) -> Result<Self, toml::de::Error> {
        let data: MapData = toml::from_str(toml_str)?;
        Ok(Self::from_data(data))
    }

    /// Cheapest network connection cost from any city a player owns to `target`.
    /// Uses Dijkstra's algorithm.
    pub fn connection_cost_to(&self, owned_cities: &[String], target: &str) -> Option<u32> {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        if owned_cities.is_empty() {
            // First city: no routing cost, just the city connection fee.
            return Some(0);
        }

        let mut dist: HashMap<&str, u32> = HashMap::new();
        let mut heap = BinaryHeap::new();

        for start in owned_cities {
            dist.insert(start.as_str(), 0);
            heap.push(Reverse((0u32, start.as_str())));
        }

        while let Some(Reverse((cost, node))) = heap.pop() {
            if node == target {
                return Some(cost);
            }
            if dist.get(node).copied().unwrap_or(u32::MAX) < cost {
                continue;
            }
            if let Some(neighbors) = self.edges.get(node) {
                for (neighbor, edge_cost) in neighbors {
                    let next_cost = cost + edge_cost;
                    let entry = dist.entry(neighbor.as_str()).or_insert(u32::MAX);
                    if next_cost < *entry {
                        *entry = next_cost;
                        heap.push(Reverse((next_cost, neighbor.as_str())));
                    }
                }
            }
        }
        None
    }

    /// Exact minimum route network needed to connect `target_cities` to the player's
    /// existing network (`owned_cities`). If `owned_cities` is empty, this returns
    /// the minimum network that interconnects all targets.
    pub fn connection_network_for(
        &self,
        owned_cities: &[String],
        target_cities: &[String],
    ) -> Option<ConnectionNetwork> {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        if target_cities.is_empty() {
            return Some(ConnectionNetwork {
                route_cost: 0,
                edges: Vec::new(),
            });
        }

        let mut unique_targets = Vec::new();
        let mut seen_targets = HashSet::new();
        for target in target_cities {
            if !self.cities.contains_key(target) {
                return None;
            }
            if seen_targets.insert(target.clone()) {
                unique_targets.push(target.clone());
            }
        }
        let terminals = unique_targets;
        if terminals.len() >= usize::BITS as usize {
            return None;
        }

        let mut node_ids: Vec<String> = self.cities.keys().cloned().collect();
        node_ids.sort();
        let n = node_ids.len();
        let mut idx_by_id: HashMap<String, usize> = HashMap::with_capacity(n);
        for (idx, city_id) in node_ids.iter().enumerate() {
            idx_by_id.insert(city_id.clone(), idx);
        }

        let mut graph: Vec<Vec<(usize, u32)>> = vec![Vec::new(); n];
        for (from, neighbors) in &self.edges {
            let Some(&from_idx) = idx_by_id.get(from) else {
                continue;
            };
            for (to, cost) in neighbors {
                if let Some(&to_idx) = idx_by_id.get(to) {
                    graph[from_idx].push((to_idx, *cost));
                }
            }
        }

        let k = terminals.len();
        let full_mask = (1usize << k) - 1;
        let inf = u32::MAX / 4;
        let mut dp: Vec<Vec<u32>> = vec![vec![inf; n]; full_mask + 1];
        let mut merge_from: Vec<Vec<Option<(usize, usize)>>> = vec![vec![None; n]; full_mask + 1];
        let mut path_parent: Vec<Vec<Option<usize>>> = vec![vec![None; n]; full_mask + 1];

        for (i, terminal) in terminals.iter().enumerate() {
            let &t_idx = idx_by_id.get(terminal)?;
            dp[1 << i][t_idx] = 0;
        }

        for mask in 1usize..=full_mask {
            let mut sub = (mask - 1) & mask;
            while sub > 0 {
                let other = mask ^ sub;
                if other == 0 {
                    sub = (sub - 1) & mask;
                    continue;
                }
                if sub > other {
                    sub = (sub - 1) & mask;
                    continue;
                }
                for v in 0..n {
                    let a = dp[sub][v];
                    let b = dp[other][v];
                    if a == inf || b == inf {
                        continue;
                    }
                    let cand = a.saturating_add(b);
                    if cand < dp[mask][v] {
                        dp[mask][v] = cand;
                        merge_from[mask][v] = Some((sub, other));
                        path_parent[mask][v] = None;
                    }
                }
                sub = (sub - 1) & mask;
            }

            let mut heap = BinaryHeap::new();
            for (v, &cost) in dp[mask].iter().enumerate() {
                if cost < inf {
                    heap.push(Reverse((cost, v)));
                }
            }
            while let Some(Reverse((cost, v))) = heap.pop() {
                if cost != dp[mask][v] {
                    continue;
                }
                for &(next, edge_cost) in &graph[v] {
                    let next_cost = cost.saturating_add(edge_cost);
                    if next_cost < dp[mask][next] {
                        dp[mask][next] = next_cost;
                        merge_from[mask][next] = None;
                        path_parent[mask][next] = Some(v);
                        heap.push(Reverse((next_cost, next)));
                    }
                }
            }
        }

        let mut root_dist = vec![inf; n];
        let mut root_parent: Vec<Option<usize>> = vec![None; n];
        if owned_cities.is_empty() {
            for dist in root_dist.iter_mut().take(n) {
                *dist = 0;
            }
        } else {
            let mut heap = BinaryHeap::new();
            for city in owned_cities {
                if let Some(&idx) = idx_by_id.get(city) {
                    if root_dist[idx] > 0 {
                        root_dist[idx] = 0;
                        heap.push(Reverse((0u32, idx)));
                    }
                }
            }
            if heap.is_empty() {
                return None;
            }
            while let Some(Reverse((cost, v))) = heap.pop() {
                if cost != root_dist[v] {
                    continue;
                }
                for &(next, edge_cost) in &graph[v] {
                    let next_cost = cost.saturating_add(edge_cost);
                    if next_cost < root_dist[next] {
                        root_dist[next] = next_cost;
                        root_parent[next] = Some(v);
                        heap.push(Reverse((next_cost, next)));
                    }
                }
            }
        }

        let mut best_end = None;
        let mut best_cost = inf;
        for v in 0..n {
            if dp[full_mask][v] == inf || root_dist[v] == inf {
                continue;
            }
            let total = dp[full_mask][v].saturating_add(root_dist[v]);
            if total < best_cost {
                best_cost = total;
                best_end = Some(v);
            }
        }
        let end = best_end?;

        let mut used_edges: HashSet<(usize, usize)> = HashSet::new();
        fn add_undirected_edge(used_edges: &mut HashSet<(usize, usize)>, a: usize, b: usize) {
            let edge = if a <= b { (a, b) } else { (b, a) };
            used_edges.insert(edge);
        }

        fn collect_edges(
            mask: usize,
            v: usize,
            merge_from: &[Vec<Option<(usize, usize)>>],
            path_parent: &[Vec<Option<usize>>],
            used_edges: &mut HashSet<(usize, usize)>,
        ) {
            if let Some((left, right)) = merge_from[mask][v] {
                collect_edges(left, v, merge_from, path_parent, used_edges);
                collect_edges(right, v, merge_from, path_parent, used_edges);
                return;
            }
            if let Some(parent) = path_parent[mask][v] {
                add_undirected_edge(used_edges, parent, v);
                collect_edges(mask, parent, merge_from, path_parent, used_edges);
            }
        }

        collect_edges(full_mask, end, &merge_from, &path_parent, &mut used_edges);
        if !owned_cities.is_empty() {
            let mut cur = end;
            while let Some(parent) = root_parent[cur] {
                add_undirected_edge(&mut used_edges, parent, cur);
                cur = parent;
            }
        }

        let mut edges: Vec<ConnectionEdge> = used_edges
            .into_iter()
            .map(|(a, b)| ConnectionEdge {
                from: node_ids[a].clone(),
                to: node_ids[b].clone(),
            })
            .collect();
        edges.sort_by(|lhs, rhs| lhs.from.cmp(&rhs.from).then_with(|| lhs.to.cmp(&rhs.to)));

        Some(ConnectionNetwork {
            route_cost: best_cost,
            edges,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_map() -> Map {
        Map::from_data(MapData {
            name: "sample".into(),
            regions: vec!["r".into()],
            image: None,
            cities: vec![
                CityData {
                    id: "a".into(),
                    name: "A".into(),
                    region: "r".into(),
                    x: None,
                    y: None,
                },
                CityData {
                    id: "b".into(),
                    name: "B".into(),
                    region: "r".into(),
                    x: None,
                    y: None,
                },
                CityData {
                    id: "c".into(),
                    name: "C".into(),
                    region: "r".into(),
                    x: None,
                    y: None,
                },
                CityData {
                    id: "d".into(),
                    name: "D".into(),
                    region: "r".into(),
                    x: None,
                    y: None,
                },
            ],
            connections: vec![
                ConnectionData {
                    from: "a".into(),
                    to: "b".into(),
                    cost: 2,
                },
                ConnectionData {
                    from: "b".into(),
                    to: "c".into(),
                    cost: 2,
                },
                ConnectionData {
                    from: "a".into(),
                    to: "c".into(),
                    cost: 10,
                },
                ConnectionData {
                    from: "c".into(),
                    to: "d".into(),
                    cost: 1,
                },
            ],
            resource_slots: vec![],
            turn_order_slots: vec![],
        })
    }

    #[test]
    fn exact_network_connects_to_existing_root() {
        let map = sample_map();
        let network = map
            .connection_network_for(&["a".into()], &["c".into(), "d".into()])
            .expect("network should exist");
        assert_eq!(network.route_cost, 5);
        assert!(network
            .edges
            .iter()
            .any(|e| { (e.from == "a" && e.to == "b") || (e.from == "b" && e.to == "a") }));
        assert!(network
            .edges
            .iter()
            .any(|e| { (e.from == "b" && e.to == "c") || (e.from == "c" && e.to == "b") }));
        assert!(network
            .edges
            .iter()
            .any(|e| { (e.from == "c" && e.to == "d") || (e.from == "d" && e.to == "c") }));
    }

    #[test]
    fn exact_network_without_owned_uses_minimum_tree() {
        let map = sample_map();
        let network = map
            .connection_network_for(&[], &["a".into(), "c".into(), "d".into()])
            .expect("network should exist");
        assert_eq!(network.route_cost, 5);
    }
}
