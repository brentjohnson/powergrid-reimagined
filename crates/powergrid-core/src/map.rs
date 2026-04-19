use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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

/// Result of finding the cheapest connection path from a player's network to a target city.
#[derive(Debug, Clone)]
pub struct ShortestPath {
    /// Total edge cost to reach the target (0 when no routing is needed).
    pub cost: u32,
    /// Edges traversed in the shortest path, each pair stored in lexicographic order
    /// `(smaller_id, larger_id)` to make deduplication easy.
    pub edges: Vec<(String, String)>,
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

        Self {
            name: data.name,
            regions: data.regions,
            cities,
            edges,
        }
    }

    pub fn load(toml_str: &str) -> Result<Self, toml::de::Error> {
        let data: MapData = toml::from_str(toml_str)?;
        Ok(Self::from_data(data))
    }

    /// Cheapest path from any owned city to `target`, including the edges traversed.
    ///
    /// Returns `cost = 0` with empty edges when `owned_cities` is empty (first city) or
    /// when `target` is already in `owned_cities`. Returns `None` when `target` does not
    /// exist in this map or is unreachable.
    pub fn shortest_path_to(&self, owned_cities: &[String], target: &str) -> Option<ShortestPath> {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        if !self.cities.contains_key(target) {
            return None;
        }

        if owned_cities.is_empty() || owned_cities.iter().any(|c| c == target) {
            return Some(ShortestPath {
                cost: 0,
                edges: Vec::new(),
            });
        }

        let mut dist: HashMap<String, u32> = HashMap::new();
        let mut parent: HashMap<String, String> = HashMap::new();
        let mut heap: BinaryHeap<Reverse<(u32, String)>> = BinaryHeap::new();

        for start in owned_cities {
            dist.insert(start.clone(), 0);
            heap.push(Reverse((0u32, start.clone())));
        }

        let mut found_cost: Option<u32> = None;

        while let Some(Reverse((cost, node))) = heap.pop() {
            if node.as_str() == target {
                found_cost = Some(cost);
                break;
            }
            if dist.get(&node).copied().unwrap_or(u32::MAX) < cost {
                continue;
            }
            if let Some(neighbors) = self.edges.get(&node) {
                for (neighbor, edge_cost) in neighbors {
                    let next_cost = cost + edge_cost;
                    let entry = dist.entry(neighbor.clone()).or_insert(u32::MAX);
                    if next_cost < *entry {
                        *entry = next_cost;
                        parent.insert(neighbor.clone(), node.clone());
                        heap.push(Reverse((next_cost, neighbor.clone())));
                    }
                }
            }
        }

        let cost = found_cost?;

        // Reconstruct path from target back to a source node (which has no parent entry).
        let mut edges = Vec::new();
        let mut cur: String = target.to_string();
        while let Some(prev) = parent.get(&cur).cloned() {
            // Store each edge in lexicographic order for easy deduplication.
            let (a, b) = if prev <= cur {
                (prev.clone(), cur.clone())
            } else {
                (cur.clone(), prev.clone())
            };
            edges.push((a, b));
            cur = prev;
        }

        Some(ShortestPath { cost, edges })
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
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a small test map:  a --5-- b --3-- c
    ///                                  \--4-- d
    fn small_map() -> Map {
        Map::from_data(MapData {
            name: "Test".into(),
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
                    cost: 5,
                },
                ConnectionData {
                    from: "b".into(),
                    to: "c".into(),
                    cost: 3,
                },
                ConnectionData {
                    from: "b".into(),
                    to: "d".into(),
                    cost: 4,
                },
            ],
        })
    }

    #[test]
    fn shortest_path_first_city_has_zero_cost_no_edges() {
        let map = small_map();
        let path = map.shortest_path_to(&[], "c").expect("should return Some");
        assert_eq!(path.cost, 0);
        assert!(path.edges.is_empty());
    }

    #[test]
    fn shortest_path_target_already_owned_returns_zero() {
        let map = small_map();
        let owned = vec!["a".to_string(), "c".to_string()];
        let path = map
            .shortest_path_to(&owned, "c")
            .expect("should return Some");
        assert_eq!(path.cost, 0);
        assert!(path.edges.is_empty());
    }

    #[test]
    fn shortest_path_direct_connection() {
        let map = small_map();
        let owned = vec!["a".to_string()];
        let path = map.shortest_path_to(&owned, "b").expect("reachable");
        assert_eq!(path.cost, 5);
        assert_eq!(path.edges.len(), 1);
        // Edge should be in lexicographic order.
        assert!(path.edges.contains(&("a".to_string(), "b".to_string())));
    }

    #[test]
    fn shortest_path_multi_hop() {
        let map = small_map();
        let owned = vec!["a".to_string()];
        let path = map.shortest_path_to(&owned, "c").expect("reachable");
        assert_eq!(path.cost, 8); // a->b (5) + b->c (3)
        assert_eq!(path.edges.len(), 2);
        assert!(path.edges.contains(&("a".to_string(), "b".to_string())));
        assert!(path.edges.contains(&("b".to_string(), "c".to_string())));
    }

    #[test]
    fn shortest_path_multi_source_picks_closer() {
        let map = small_map();
        // Own both a and b; c is adjacent to b at cost 3, not to a directly.
        let owned = vec!["a".to_string(), "b".to_string()];
        let path = map.shortest_path_to(&owned, "c").expect("reachable");
        assert_eq!(path.cost, 3);
        assert_eq!(path.edges.len(), 1);
        assert!(path.edges.contains(&("b".to_string(), "c".to_string())));
    }

    #[test]
    fn shortest_path_nonexistent_target_returns_none() {
        let map = small_map();
        let result = map.shortest_path_to(&["a".to_string()], "z");
        assert!(result.is_none());
    }
}
