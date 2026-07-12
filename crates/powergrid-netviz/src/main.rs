//! Interactive inspector for the RL Expert policy network.
//!
//! Load a `PGRLPOL1` policy file (or use the embedded `expert.bin`), edit the
//! 454-dim observation with labeled sliders, and watch the forward pass —
//! input cells, both 64-wide hidden layers, and the 94 output logits — update
//! live. Click a node to trace its full weighted path through every layer
//! (input → hidden1 → hidden2 → output); edge brightness is normalized
//! per weight matrix so no single connection washes out the rest. In active-
//! game mode, enable "Show Δ from real observation" to color nodes and edges
//! by how much a hand-edited input changes the forward pass relative to the
//! real observation.

mod action_labels;
mod game;
mod obs_layout;

use std::path::PathBuf;
use std::sync::Arc;

use eframe::egui;
use egui::{Color32, FontId, Pos2, Rect, Sense, Stroke, Vec2};
use rand::Rng;

use action_labels::action_label;
use game::{GameConfig, GameDriver};
use powergrid_bot_strategy::encoding::{N_ACTIONS, OBS_SIZE};
use powergrid_bot_strategy::policy::{default_policy, sample_masked, ForwardTrace, MlpPolicy};
use powergrid_core::BotDifficulty;

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug)]
enum Layer {
    Input,
    Hidden1,
    Hidden2,
    Output,
}

/// Which weight matrix an edge belongs to (input→hidden1, hidden1→hidden2,
/// hidden2→output). Edge brightness is normalized independently per
/// transition so one large weight doesn't wash out the others.
#[derive(Clone, Copy)]
enum Transition {
    L1,
    L2,
    Out,
}

impl Transition {
    const COUNT: usize = 3;

    fn idx(self) -> usize {
        match self {
            Transition::L1 => 0,
            Transition::L2 => 1,
            Transition::Out => 2,
        }
    }
}

/// One edge drawn for the selected node: screen-space endpoints, the raw
/// weight, the activation of the edge's source node, and which weight matrix
/// it belongs to.
struct Edge {
    from: Pos2,
    to: Pos2,
    weight: f32,
    src_activation: f32,
    transition: Transition,
}

/// Screen-space positions and node sizes for one frame's network canvas.
struct NetLayout {
    input_pos: Vec<Pos2>,
    input_cell: f32,
    h1_pos: Vec<Pos2>,
    h1_r: f32,
    h2_pos: Vec<Pos2>,
    h2_r: f32,
    out_pos: Vec<Pos2>,
    out_cell: f32,
    headers: [(Pos2, &'static str); 4],
}

/// Place `n` items in a centered grid of `cols` columns within `rect`.
/// Returns the positions and the (square) cell size used.
fn grid_positions(rect: Rect, n: usize, cols: usize) -> (Vec<Pos2>, f32) {
    let cols = cols.max(1);
    let rows = n.div_ceil(cols).max(1);
    let cell = (rect.width() / cols as f32)
        .min(rect.height() / rows as f32)
        .clamp(2.0, 28.0);
    let total_w = cell * cols as f32;
    let origin = Pos2::new(rect.center().x - total_w / 2.0, rect.top());
    let positions = (0..n)
        .map(|i| {
            let c = (i % cols) as f32;
            let r = (i / cols) as f32;
            Pos2::new(origin.x + (c + 0.5) * cell, origin.y + (r + 0.5) * cell)
        })
        .collect();
    (positions, cell)
}

/// Place `n` items in a single vertical column within `rect`.
/// Returns the positions and a suggested node radius.
fn column_positions(rect: Rect, n: usize) -> (Vec<Pos2>, f32) {
    let n = n.max(1);
    let spacing = rect.height() / n as f32;
    let radius = (spacing * 0.5 * 0.8).clamp(1.5, 8.0);
    let x = rect.center().x;
    let positions = (0..n)
        .map(|i| Pos2::new(x, rect.top() + spacing * (i as f32 + 0.5)))
        .collect();
    (positions, radius)
}

impl NetLayout {
    fn new(rect: Rect, obs_size: usize, hidden: usize, n_actions: usize) -> Self {
        let header_h = 22.0;
        let margin = 12.0;
        let col_w = (rect.width() - 5.0 * margin) / 4.0;
        let col_rect = |i: usize| {
            let x = rect.left() + margin + (col_w + margin) * i as f32;
            Rect::from_min_size(
                Pos2::new(x, rect.top() + header_h + margin),
                Vec2::new(col_w, rect.height() - header_h - 2.0 * margin),
            )
        };

        let input_rect = col_rect(0);
        let h1_rect = col_rect(1);
        let h2_rect = col_rect(2);
        let out_rect = col_rect(3);

        let input_cols = (obs_size as f32).sqrt().ceil() as usize;
        let out_cols = (n_actions as f32).sqrt().ceil() as usize;

        let (input_pos, input_cell) = grid_positions(input_rect, obs_size, input_cols);
        let (h1_pos, h1_r) = column_positions(h1_rect, hidden);
        let (h2_pos, h2_r) = column_positions(h2_rect, hidden);
        let (out_pos, out_cell) = grid_positions(out_rect, n_actions, out_cols);

        let header_y = rect.top() + header_h / 2.0;
        let headers = [
            (Pos2::new(input_rect.center().x, header_y), "Input (obs)"),
            (Pos2::new(h1_rect.center().x, header_y), "Hidden 1 (tanh)"),
            (Pos2::new(h2_rect.center().x, header_y), "Hidden 2 (tanh)"),
            (Pos2::new(out_rect.center().x, header_y), "Output (logits)"),
        ];

        Self {
            input_pos,
            input_cell,
            h1_pos,
            h1_r,
            h2_pos,
            h2_r,
            out_pos,
            out_cell,
            headers,
        }
    }

    /// Find the node nearest `pos`, if it's within the node's own size.
    fn hit_test(&self, pos: Pos2) -> Option<(Layer, usize)> {
        let mut best: Option<(Layer, usize, f32)> = None;
        let mut consider = |layer: Layer, positions: &[Pos2], threshold: f32| {
            for (i, &p) in positions.iter().enumerate() {
                let d = (p - pos).length();
                if d <= threshold && best.is_none_or(|(_, _, bd)| d < bd) {
                    best = Some((layer, i, d));
                }
            }
        };
        consider(Layer::Input, &self.input_pos, self.input_cell * 0.7);
        consider(Layer::Hidden1, &self.h1_pos, self.h1_r * 1.5);
        consider(Layer::Hidden2, &self.h2_pos, self.h2_r * 1.5);
        consider(Layer::Output, &self.out_pos, self.out_cell * 0.7);
        best.map(|(l, i, _)| (l, i))
    }
}

// ---------------------------------------------------------------------------
// Colors
// ---------------------------------------------------------------------------

fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let lerp = |a: u8, b: u8| (a as f32 + (b as f32 - a as f32) * t).round() as u8;
    Color32::from_rgb(lerp(a.r(), b.r()), lerp(a.g(), b.g()), lerp(a.b(), b.b()))
}

const COLOR_LOW: Color32 = Color32::from_rgb(50, 90, 200);
const COLOR_MID: Color32 = Color32::from_rgb(225, 225, 230);
const COLOR_HIGH: Color32 = Color32::from_rgb(220, 60, 50);

/// Color a value in `[0, 1]` (e.g. a raw observation entry).
fn heat_color_01(v: f32) -> Color32 {
    let v = v.clamp(0.0, 1.0);
    if v < 0.5 {
        lerp_color(COLOR_LOW, COLOR_MID, v * 2.0)
    } else {
        lerp_color(COLOR_MID, COLOR_HIGH, (v - 0.5) * 2.0)
    }
}

/// Color a value on `[-scale, scale]` (e.g. a tanh activation or a logit).
fn diverging_color(v: f32, scale: f32) -> Color32 {
    let scale = scale.max(1e-6);
    let t = (v / scale).clamp(-1.0, 1.0);
    if t < 0.0 {
        lerp_color(COLOR_MID, COLOR_LOW, -t)
    } else {
        lerp_color(COLOR_MID, COLOR_HIGH, t)
    }
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

struct NetViz {
    policy: Arc<MlpPolicy>,
    policy_source: String,
    obs: Vec<f32>,
    /// The real observation as of the last sync from `self.game`, captured
    /// so hand-edits to `obs` can be diffed against it in "Show Δ from real
    /// observation" mode. `None` until a game has reached an inspected turn.
    baseline_obs: Option<Vec<f32>>,
    diff_mode: bool,
    selected: Option<(Layer, usize)>,
    weight_by_activation: bool,
    game: Option<GameDriver>,
    mask: Option<Vec<u8>>,
    game_cfg: GameConfig,
    status: String,
}

impl NetViz {
    fn new(policy: Arc<MlpPolicy>, policy_source: String) -> Self {
        Self {
            policy,
            policy_source,
            obs: vec![0.0; OBS_SIZE],
            baseline_obs: None,
            diff_mode: false,
            selected: None,
            weight_by_activation: false,
            game: None,
            mask: None,
            game_cfg: GameConfig::default(),
            status: "No game started — observation is hand-edited.".to_string(),
        }
    }

    /// Refreshes `obs`/`mask`/`status` from `self.game` after starting or
    /// stepping it. Leaves `obs` untouched (for hand-tweaking) once it's not
    /// the inspected seat's turn. Also captures the fresh observation as
    /// `baseline_obs` for diff mode.
    fn sync_from_game(&mut self) {
        let Some(game) = &self.game else { return };
        self.status = game.status();
        if game.is_inspected_turn() {
            self.obs = game.observation();
            self.baseline_obs = Some(self.obs.clone());
            self.mask = Some(game.action_mask());
        } else {
            self.mask = None;
        }
    }

    fn show_game_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Game");
        ui.label(egui::RichText::new(&self.status).italics());

        ui.add(egui::Slider::new(&mut self.game_cfg.players, 2..=6).text("players"));

        egui::ComboBox::from_label("bot difficulty")
            .selected_text(difficulty_label(self.game_cfg.difficulty))
            .show_ui(ui, |ui| {
                for d in [
                    BotDifficulty::Easy,
                    BotDifficulty::Normal,
                    BotDifficulty::Hard,
                    BotDifficulty::Expert,
                ] {
                    ui.selectable_value(&mut self.game_cfg.difficulty, d, difficulty_label(d));
                }
            });

        ui.add(egui::DragValue::new(&mut self.game_cfg.seed).prefix("seed: "));

        let mut override_end_game = self.game_cfg.end_game_cities.is_some();
        ui.horizontal(|ui| {
            if ui
                .checkbox(&mut override_end_game, "override end-game cities")
                .changed()
            {
                self.game_cfg.end_game_cities = if override_end_game { Some(17) } else { None };
            }
            if let Some(n) = &mut self.game_cfg.end_game_cities {
                ui.add(egui::DragValue::new(n).range(7..=21));
            }
        });

        if ui.button("New game").clicked() {
            self.baseline_obs = None;
            match GameDriver::new(&self.game_cfg) {
                Ok(driver) => {
                    self.game = Some(driver);
                    self.sync_from_game();
                }
                Err(e) => {
                    self.status = format!("failed to start game: {e}");
                    self.game = None;
                    self.mask = None;
                }
            }
        }

        ui.separator();
    }

    /// Samples a legal action from the policy's logits over the *current*
    /// (possibly hand-tweaked) `obs`, restricted to the real game's mask.
    fn apply_policy_move(&mut self, trace: &ForwardTrace) {
        let Some(mask) = self.mask.clone() else {
            return;
        };
        let Some(game) = &mut self.game else { return };
        let mut rng = rand::thread_rng();
        if let Some(action_id) = sample_masked(&trace.logits, &mask, &mut rng) {
            if let Err(e) = game.step_inspected_macro(action_id as u16) {
                self.status = format!("action error: {e}");
                return;
            }
        }
        self.sync_from_game();
    }

    /// Applies the action currently selected in the output list, if it's
    /// legal per the real game's mask.
    fn apply_selected_action(&mut self) {
        let Some((Layer::Output, action_id)) = self.selected else {
            self.status = "select an output action first".to_string();
            return;
        };
        let Some(mask) = &self.mask else { return };
        if mask[action_id] == 0 {
            self.status = "selected action is not legal in the current position".to_string();
            return;
        }
        let Some(game) = &mut self.game else { return };
        if let Err(e) = game.step_inspected_macro(action_id as u16) {
            self.status = format!("action error: {e}");
            return;
        }
        self.sync_from_game();
    }

    /// All edges along the full path through the network from the selected
    /// node: its own fan-in/fan-out, plus every weight in the layers beyond
    /// it (the next hop's full layer fans out to the rest), so the entire
    /// input→...→output path touching the selection is returned.
    ///
    /// `obs`/`trace` supply source activations and may be the real
    /// observation/trace or a hand-edited one — callers pass both to build
    /// a diff.
    fn edge_list(
        &self,
        layer: Layer,
        idx: usize,
        obs: &[f32],
        trace: &ForwardTrace,
        lay: &NetLayout,
    ) -> Vec<Edge> {
        let (obs_size, hidden, n_actions) = self.policy.dims();
        let mut edges = Vec::new();
        match layer {
            Layer::Input => {
                // input[idx] -> all of hidden1, then the full network beyond.
                let (w, _) = self.policy.l1();
                for o in 0..hidden {
                    edges.push(Edge {
                        from: lay.input_pos[idx],
                        to: lay.h1_pos[o],
                        weight: w[o * obs_size + idx],
                        src_activation: obs[idx],
                        transition: Transition::L1,
                    });
                }
                self.push_full_l2(&mut edges, trace, lay);
                self.push_full_out(&mut edges, trace, lay);
            }
            Layer::Hidden1 => {
                // all of input -> hidden1[idx], and hidden1[idx] -> all of
                // hidden2, then the full network beyond.
                self.push_full_l1(&mut edges, obs, lay);
                let (w, _) = self.policy.l2();
                for o in 0..hidden {
                    edges.push(Edge {
                        from: lay.h1_pos[idx],
                        to: lay.h2_pos[o],
                        weight: w[o * hidden + idx],
                        src_activation: trace.h1_post[idx],
                        transition: Transition::L2,
                    });
                }
                self.push_full_out(&mut edges, trace, lay);
            }
            Layer::Hidden2 => {
                // all of input -> all of hidden1, all of hidden1 ->
                // hidden2[idx], and hidden2[idx] -> all of output.
                self.push_full_l1(&mut edges, obs, lay);
                let (w, _) = self.policy.l2();
                for i in 0..hidden {
                    edges.push(Edge {
                        from: lay.h1_pos[i],
                        to: lay.h2_pos[idx],
                        weight: w[idx * hidden + i],
                        src_activation: trace.h1_post[i],
                        transition: Transition::L2,
                    });
                }
                let (w, _) = self.policy.out();
                for o in 0..n_actions {
                    edges.push(Edge {
                        from: lay.h2_pos[idx],
                        to: lay.out_pos[o],
                        weight: w[o * hidden + idx],
                        src_activation: trace.h2_post[idx],
                        transition: Transition::Out,
                    });
                }
            }
            Layer::Output => {
                // the full network feeding output[idx]: all of input -> all
                // of hidden1, all of hidden1 -> all of hidden2, and all of
                // hidden2 -> output[idx].
                self.push_full_l1(&mut edges, obs, lay);
                self.push_full_l2(&mut edges, trace, lay);
                let (w, _) = self.policy.out();
                for i in 0..hidden {
                    edges.push(Edge {
                        from: lay.h2_pos[i],
                        to: lay.out_pos[idx],
                        weight: w[idx * hidden + i],
                        src_activation: trace.h2_post[i],
                        transition: Transition::Out,
                    });
                }
            }
        }
        edges
    }

    /// Pushes every input -> hidden1 edge.
    fn push_full_l1(&self, edges: &mut Vec<Edge>, obs: &[f32], lay: &NetLayout) {
        let (obs_size, hidden, _) = self.policy.dims();
        let (w, _) = self.policy.l1();
        for o in 0..hidden {
            for i in 0..obs_size {
                edges.push(Edge {
                    from: lay.input_pos[i],
                    to: lay.h1_pos[o],
                    weight: w[o * obs_size + i],
                    src_activation: obs[i],
                    transition: Transition::L1,
                });
            }
        }
    }

    /// Pushes every hidden1 -> hidden2 edge.
    fn push_full_l2(&self, edges: &mut Vec<Edge>, trace: &ForwardTrace, lay: &NetLayout) {
        let (_, hidden, _) = self.policy.dims();
        let (w, _) = self.policy.l2();
        for o in 0..hidden {
            for i in 0..hidden {
                edges.push(Edge {
                    from: lay.h1_pos[i],
                    to: lay.h2_pos[o],
                    weight: w[o * hidden + i],
                    src_activation: trace.h1_post[i],
                    transition: Transition::L2,
                });
            }
        }
    }

    /// Pushes every hidden2 -> output edge.
    fn push_full_out(&self, edges: &mut Vec<Edge>, trace: &ForwardTrace, lay: &NetLayout) {
        let (_, hidden, n_actions) = self.policy.dims();
        let (w, _) = self.policy.out();
        for o in 0..n_actions {
            for i in 0..hidden {
                edges.push(Edge {
                    from: lay.h2_pos[i],
                    to: lay.out_pos[o],
                    weight: w[o * hidden + i],
                    src_activation: trace.h2_post[i],
                    transition: Transition::Out,
                });
            }
        }
    }

    fn show_inputs(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui.button("Zero all").clicked() {
                self.obs.iter_mut().for_each(|v| *v = 0.0);
            }
            if ui.button("Randomize").clicked() {
                let mut rng = rand::thread_rng();
                for v in &mut self.obs {
                    *v = rng.gen::<f32>();
                }
            }
        });
        ui.separator();
        for sec in obs_layout::sections() {
            let default_open = sec.len <= 8;
            egui::CollapsingHeader::new(format!("{} ({})", sec.name, sec.len))
                .default_open(default_open)
                .show(ui, |ui| {
                    for i in 0..sec.len {
                        let idx = sec.start + i;
                        let label = (sec.label)(i);
                        ui.add(egui::Slider::new(&mut self.obs[idx], 0.0..=1.0).text(label));
                    }
                });
        }
    }

    /// `baseline`, when present, is `(real_observation, real_forward_trace)`
    /// — the position before any hand-edits — used by "Show Δ from real
    /// observation" mode.
    fn show_network(
        &mut self,
        ui: &mut egui::Ui,
        trace: &ForwardTrace,
        baseline: Option<(&[f32], &ForwardTrace)>,
    ) {
        ui.horizontal(|ui| {
            ui.checkbox(
                &mut self.weight_by_activation,
                "Scale edges by source activation",
            );
            if ui.button("Clear selection").clicked() {
                self.selected = None;
            }
            if baseline.is_some() {
                ui.checkbox(&mut self.diff_mode, "Show Δ from real observation");
            }
            ui.label(
                egui::RichText::new(
                    "edges: red = positive, blue = negative; width/opacity ∝ |value|, \
                     normalized per transition (input→h1 / h1→h2 / h2→out)",
                )
                .small()
                .weak(),
            );
        });

        let available = ui.available_size();
        let (response, painter) = ui.allocate_painter(available, Sense::click());
        let rect = response.rect;
        painter.rect_filled(rect, 0.0, Color32::from_rgb(18, 22, 30));

        let (obs_size, hidden, n_actions) = self.policy.dims();
        let lay = NetLayout::new(rect, obs_size, hidden, n_actions);

        // Active only when the user has both opted in and a real observation
        // exists to diff against.
        let (b_obs, b_trace) = match baseline {
            Some((o, t)) if self.diff_mode => (Some(o), Some(t)),
            _ => (None, None),
        };

        // Edges along the full path through the network from the selected
        // node, drawn beneath the nodes.
        if let Some((layer, idx)) = self.selected {
            let edges = self.edge_list(layer, idx, &self.obs, trace, &lay);
            let values: Vec<f32> = match (b_obs, b_trace) {
                (Some(bo), Some(bt)) => {
                    // Δcontribution: how much this edge's contribution to its
                    // target changes due to the hand-edited observation.
                    let b_edges = self.edge_list(layer, idx, bo, bt, &lay);
                    edges
                        .iter()
                        .zip(&b_edges)
                        .map(|(e, b)| e.weight * (e.src_activation - b.src_activation))
                        .collect()
                }
                _ => edges
                    .iter()
                    .map(|e| {
                        if self.weight_by_activation {
                            e.weight * e.src_activation
                        } else {
                            e.weight
                        }
                    })
                    .collect(),
            };

            // Normalize brightness independently per weight matrix so the
            // (much larger) input→h1 fan-out doesn't drown out h1→h2/h2→out.
            let mut max_abs = [1e-6f32; Transition::COUNT];
            for (e, &v) in edges.iter().zip(&values) {
                let i = e.transition.idx();
                max_abs[i] = max_abs[i].max(v.abs());
            }
            for (e, &v) in edges.iter().zip(&values) {
                let m = max_abs[e.transition.idx()];
                let t = (v.abs() / m).clamp(0.0, 1.0).powf(0.6);
                let alpha = (40.0 + 180.0 * t) as u8;
                let color = if v >= 0.0 {
                    Color32::from_rgba_unmultiplied(
                        COLOR_HIGH.r(),
                        COLOR_HIGH.g(),
                        COLOR_HIGH.b(),
                        alpha,
                    )
                } else {
                    Color32::from_rgba_unmultiplied(
                        COLOR_LOW.r(),
                        COLOR_LOW.g(),
                        COLOR_LOW.b(),
                        alpha,
                    )
                };
                painter.line_segment([e.from, e.to], Stroke::new(0.4 + 2.6 * t, color));
            }
        }

        // Nodes.
        for (i, &p) in lay.input_pos.iter().enumerate() {
            let cell = Rect::from_center_size(p, Vec2::splat(lay.input_cell - 1.0));
            let color = match b_obs {
                Some(bo) => diverging_color(self.obs[i] - bo[i], 1.0),
                None => heat_color_01(self.obs[i]),
            };
            painter.rect_filled(cell, 1.0, color);
        }
        let h1_delta_scale = b_trace.map_or(1.0, |bt| {
            trace
                .h1_post
                .iter()
                .zip(&bt.h1_post)
                .fold(1e-6f32, |m, (a, b)| m.max((a - b).abs()))
        });
        for (i, &p) in lay.h1_pos.iter().enumerate() {
            let color = match b_trace {
                Some(bt) => diverging_color(trace.h1_post[i] - bt.h1_post[i], h1_delta_scale),
                None => diverging_color(trace.h1_post[i], 1.0),
            };
            painter.circle(p, lay.h1_r, color, Stroke::NONE);
        }
        let h2_delta_scale = b_trace.map_or(1.0, |bt| {
            trace
                .h2_post
                .iter()
                .zip(&bt.h2_post)
                .fold(1e-6f32, |m, (a, b)| m.max((a - b).abs()))
        });
        for (i, &p) in lay.h2_pos.iter().enumerate() {
            let color = match b_trace {
                Some(bt) => diverging_color(trace.h2_post[i] - bt.h2_post[i], h2_delta_scale),
                None => diverging_color(trace.h2_post[i], 1.0),
            };
            painter.circle(p, lay.h2_r, color, Stroke::NONE);
        }
        let logit_scale = trace.logits.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
        let logit_delta_scale = b_trace.map_or(logit_scale, |bt| {
            trace
                .logits
                .iter()
                .zip(&bt.logits)
                .fold(1e-6f32, |m, (a, b)| m.max((a - b).abs()))
        });
        for (i, &p) in lay.out_pos.iter().enumerate() {
            let cell = Rect::from_center_size(p, Vec2::splat(lay.out_cell - 1.0));
            let color = match b_trace {
                Some(bt) => diverging_color(trace.logits[i] - bt.logits[i], logit_delta_scale),
                None => diverging_color(trace.logits[i], logit_scale),
            };
            painter.rect_filled(cell, 1.0, color);
        }

        // Selection ring.
        if let Some((layer, idx)) = self.selected {
            let (pos, r) = match layer {
                Layer::Input => (lay.input_pos[idx], lay.input_cell * 0.7),
                Layer::Hidden1 => (lay.h1_pos[idx], lay.h1_r + 2.0),
                Layer::Hidden2 => (lay.h2_pos[idx], lay.h2_r + 2.0),
                Layer::Output => (lay.out_pos[idx], lay.out_cell * 0.7),
            };
            painter.circle(
                pos,
                r,
                Color32::TRANSPARENT,
                Stroke::new(2.0, Color32::YELLOW),
            );
        }

        // Column headers.
        for &(pos, text) in &lay.headers {
            painter.text(
                pos,
                egui::Align2::CENTER_CENTER,
                text,
                FontId::proportional(13.0),
                Color32::from_gray(200),
            );
        }

        // Click to select.
        if response.clicked() {
            if let Some(pos) = response.interact_pointer_pos() {
                if let Some(hit) = lay.hit_test(pos) {
                    self.selected = Some(hit);
                }
            }
        }

        // Hover tooltip.
        if let Some(pos) = response.hover_pos() {
            if let Some((layer, idx)) = lay.hit_test(pos) {
                let mut text = match layer {
                    Layer::Input => format!("{}\nvalue = {:.4}", obs_label(idx), self.obs[idx]),
                    Layer::Hidden1 => format!(
                        "Hidden1[{idx}]\npre  = {:.4}\ntanh = {:.4}",
                        trace.h1_pre[idx], trace.h1_post[idx]
                    ),
                    Layer::Hidden2 => format!(
                        "Hidden2[{idx}]\npre  = {:.4}\ntanh = {:.4}",
                        trace.h2_pre[idx], trace.h2_post[idx]
                    ),
                    Layer::Output => {
                        let (prob, _) = softmax_at(&trace.logits, idx);
                        format!(
                            "{}\nlogit = {:.4}\nprob  = {:.2}%",
                            action_label(idx),
                            trace.logits[idx],
                            prob * 100.0
                        )
                    }
                };
                match (layer, b_obs, b_trace) {
                    (Layer::Input, Some(bo), _) => {
                        text.push_str(&format!(
                            "\nreal  = {:.4}\nΔ     = {:+.4}",
                            bo[idx],
                            self.obs[idx] - bo[idx]
                        ));
                    }
                    (Layer::Hidden1, _, Some(bt)) => {
                        text.push_str(&format!(
                            "\nΔtanh = {:+.4}",
                            trace.h1_post[idx] - bt.h1_post[idx]
                        ));
                    }
                    (Layer::Hidden2, _, Some(bt)) => {
                        text.push_str(&format!(
                            "\nΔtanh = {:+.4}",
                            trace.h2_post[idx] - bt.h2_post[idx]
                        ));
                    }
                    (Layer::Output, _, Some(bt)) => {
                        text.push_str(&format!(
                            "\nΔlogit = {:+.4}",
                            trace.logits[idx] - bt.logits[idx]
                        ));
                    }
                    _ => {}
                }
                draw_tooltip(&painter, pos, &text);
            }
        }
    }

    fn show_outputs(
        &mut self,
        ui: &mut egui::Ui,
        trace: &ForwardTrace,
        baseline: Option<(&[f32], &ForwardTrace)>,
    ) {
        let baseline = baseline.filter(|_| self.diff_mode);
        ui.label(egui::RichText::new(self.selected_info(trace, baseline)).monospace());
        ui.separator();

        if self
            .game
            .as_ref()
            .is_some_and(GameDriver::is_inspected_turn)
        {
            ui.horizontal(|ui| {
                if ui.button("Apply policy move").clicked() {
                    self.apply_policy_move(trace);
                }
                if ui.button("Apply selected action").clicked() {
                    self.apply_selected_action();
                }
            });
            ui.separator();
        }

        let header = if self.mask.is_some() {
            format!(
                "Actions ({N_ACTIONS}), sorted by logit — legal moves bold; prob. columns are all-actions / legal-only"
            )
        } else {
            format!("Actions ({N_ACTIONS}), sorted by logit")
        };
        ui.label(egui::RichText::new(header).strong());

        let max_logit = trace
            .logits
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = trace
            .logits
            .iter()
            .map(|&l| (l - max_logit).exp())
            .collect();
        let sum: f32 = exps.iter().sum();

        let legal_exps: Option<(Vec<f32>, f32)> = self.mask.as_ref().map(|mask| {
            let legal_max = trace
                .logits
                .iter()
                .zip(mask)
                .filter(|(_, &m)| m != 0)
                .map(|(&l, _)| l)
                .fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = trace
                .logits
                .iter()
                .zip(mask)
                .map(|(&l, &m)| {
                    if m != 0 && legal_max.is_finite() {
                        (l - legal_max).exp()
                    } else {
                        0.0
                    }
                })
                .collect();
            let sum = exps.iter().sum();
            (exps, sum)
        });

        let mut order: Vec<usize> = (0..trace.logits.len()).collect();
        order.sort_by(|&a, &b| trace.logits[b].total_cmp(&trace.logits[a]));

        egui::ScrollArea::vertical().show(ui, |ui| {
            for i in order {
                let prob = exps[i] / sum;
                let legal = self.mask.as_ref().map(|m| m[i] != 0);

                let legal_prob = match (&legal_exps, legal) {
                    (Some((legal_exps, legal_sum)), Some(true)) if *legal_sum > 0.0 => {
                        format!("{:>5.1}%", legal_exps[i] / legal_sum * 100.0)
                    }
                    (Some(_), _) => "   -  ".to_string(),
                    (None, _) => String::new(),
                };

                let selected = self.selected == Some((Layer::Output, i));
                let text = if self.mask.is_some() {
                    format!(
                        "{:>5.1}% {}  {:+7.3}  {}",
                        prob * 100.0,
                        legal_prob,
                        trace.logits[i],
                        action_label(i)
                    )
                } else {
                    format!(
                        "{:>5.1}%  {:+7.3}  {}",
                        prob * 100.0,
                        trace.logits[i],
                        action_label(i)
                    )
                };
                let text = match legal {
                    Some(true) => egui::RichText::new(text).strong(),
                    Some(false) => egui::RichText::new(text).weak(),
                    None => egui::RichText::new(text),
                };
                if ui.selectable_label(selected, text).clicked() {
                    self.selected = Some((Layer::Output, i));
                }
            }
        });
    }

    /// `baseline`, when present, is `(real_observation, real_forward_trace)`
    /// — used to append a `Δ` line for the selected node in diff mode.
    fn selected_info(
        &self,
        trace: &ForwardTrace,
        baseline: Option<(&[f32], &ForwardTrace)>,
    ) -> String {
        match self.selected {
            None => {
                "Click a node in the network view to inspect\nits value and weighted connections."
                    .to_string()
            }
            Some((Layer::Input, i)) => {
                let mut s = format!(
                    "Selected: Input[{i}]\n{}\nvalue = {:.4}",
                    obs_label(i),
                    self.obs[i]
                );
                if let Some((b_obs, _)) = baseline {
                    s.push_str(&format!(
                        "\nreal  = {:.4}\nΔ     = {:+.4}",
                        b_obs[i],
                        self.obs[i] - b_obs[i]
                    ));
                }
                s
            }
            Some((Layer::Hidden1, i)) => {
                let (_, b) = self.policy.l1();
                let mut s = format!(
                    "Selected: Hidden1[{i}]\npre-activation = {:.4}\nbias           = {:.4}\ntanh           = {:.4}",
                    trace.h1_pre[i], b[i], trace.h1_post[i]
                );
                if let Some((_, b_trace)) = baseline {
                    s.push_str(&format!(
                        "\nΔtanh          = {:+.4}",
                        trace.h1_post[i] - b_trace.h1_post[i]
                    ));
                }
                s
            }
            Some((Layer::Hidden2, i)) => {
                let (_, b) = self.policy.l2();
                let mut s = format!(
                    "Selected: Hidden2[{i}]\npre-activation = {:.4}\nbias           = {:.4}\ntanh           = {:.4}",
                    trace.h2_pre[i], b[i], trace.h2_post[i]
                );
                if let Some((_, b_trace)) = baseline {
                    s.push_str(&format!(
                        "\nΔtanh          = {:+.4}",
                        trace.h2_post[i] - b_trace.h2_post[i]
                    ));
                }
                s
            }
            Some((Layer::Output, i)) => {
                let (_, b) = self.policy.out();
                let (prob, _) = softmax_at(&trace.logits, i);
                let mut s = format!(
                    "Selected: Output[{i}]: {}\nlogit = {:.4}\nbias  = {:.4}\nprob  = {:.2}%",
                    action_label(i),
                    trace.logits[i],
                    b[i],
                    prob * 100.0
                );
                if let Some((_, b_trace)) = baseline {
                    s.push_str(&format!(
                        "\nΔlogit = {:+.4}",
                        trace.logits[i] - b_trace.logits[i]
                    ));
                }
                s
            }
        }
    }
}

/// Softmax probability of logit `i`, plus the normalizing sum.
fn softmax_at(logits: &[f32], i: usize) -> (f32, f32) {
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
    let sum: f32 = exps.iter().sum();
    (exps[i] / sum, sum)
}

/// Display name for a bot difficulty in the "New game" setup panel.
fn difficulty_label(d: BotDifficulty) -> &'static str {
    match d {
        BotDifficulty::Easy => "Easy",
        BotDifficulty::Normal => "Normal",
        BotDifficulty::Hard => "Hard",
        BotDifficulty::Expert => "Expert",
    }
}

/// Human-readable label for observation index `i`, e.g. "Self cities: atlanta".
fn obs_label(i: usize) -> String {
    for s in obs_layout::sections() {
        if i >= s.start && i < s.start + s.len {
            return format!("{}: {}", s.name, (s.label)(i - s.start));
        }
    }
    format!("obs[{i}]")
}

fn draw_tooltip(painter: &egui::Painter, anchor: Pos2, text: &str) {
    let lines = text.lines().count().max(1);
    let size = Vec2::new(240.0, 16.0 * lines as f32 + 8.0);
    let pos = Pos2::new(anchor.x + 10.0, anchor.y - size.y - 10.0);
    let rect = Rect::from_min_size(pos, size);
    painter.rect_filled(rect, 3.0, Color32::from_rgba_unmultiplied(15, 15, 20, 235));
    painter.text(
        rect.min + Vec2::new(6.0, 4.0),
        egui::Align2::LEFT_TOP,
        text,
        FontId::monospace(12.0),
        Color32::from_gray(230),
    );
}

// ---------------------------------------------------------------------------
// eframe::App
// ---------------------------------------------------------------------------

impl eframe::App for NetViz {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let trace = self.policy.forward_trace(&self.obs);
        // Clone (not borrow) so `baseline` doesn't alias `self` while we hold
        // `&mut self` for the panel-rendering calls below.
        let baseline_obs = self.baseline_obs.clone();
        let baseline_trace = baseline_obs.as_ref().map(|b| self.policy.forward_trace(b));
        let baseline: Option<(&[f32], &ForwardTrace)> = match (&baseline_obs, &baseline_trace) {
            (Some(o), Some(t)) => Some((o.as_slice(), t)),
            _ => None,
        };
        let (obs_size, hidden, n_actions) = self.policy.dims();

        egui::TopBottomPanel::top("title").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Power Grid RL Policy Inspector").strong());
                ui.separator();
                ui.label(format!("model: {}", self.policy_source));
                ui.separator();
                ui.label(format!(
                    "obs={obs_size} → {hidden} → {hidden} → logits={n_actions}"
                ));
            });
        });

        egui::SidePanel::left("inputs")
            .min_width(320.0)
            .max_width(420.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.show_game_panel(ui);
                    ui.heading("Observation inputs");
                    self.show_inputs(ui);
                });
            });

        egui::SidePanel::right("outputs")
            .min_width(340.0)
            .max_width(440.0)
            .show(ctx, |ui| {
                ui.heading("Selection & outputs");
                self.show_outputs(ui, &trace, baseline);
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            self.show_network(ui, &trace, baseline);
        });
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() -> eframe::Result {
    let args: Vec<String> = std::env::args().collect();

    let (policy, source) = match args.get(1) {
        Some(path) => {
            let bytes =
                std::fs::read(path).unwrap_or_else(|e| panic!("failed to read {path}: {e}"));
            let policy = MlpPolicy::from_bytes(&bytes)
                .unwrap_or_else(|e| panic!("invalid policy file {path}: {e:?}"));
            (Arc::new(policy), PathBuf::from(path).display().to_string())
        }
        None => {
            let policy = default_policy().expect("embedded policy must load");
            (policy, "embedded expert.bin".to_string())
        }
    };

    eframe::run_native(
        "Power Grid RL Policy Inspector",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([1500.0, 950.0])
                .with_title("Power Grid RL Policy Inspector"),
            ..Default::default()
        },
        Box::new(move |_cc| Ok(Box::new(NetViz::new(policy, source)))),
    )
}
