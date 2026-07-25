//! Integration tests for the Expert (RL policy) bot: encoding smoke tests,
//! non-default-map fallback, and an ignored full-game strength test.

use powergrid_bot_strategy::{default_registry, encoding, policy, Bot};
use powergrid_core::{
    actions::Action,
    map::{default_map, Map},
    rules::apply_action,
    state::GameState,
    types::{BotDifficulty, Phase, PlayerColor, PlayerId},
};

const GERMANY_TOML: &str = include_str!("../../../assets/maps/germany.toml");

const COLORS: [PlayerColor; 4] = [
    PlayerColor::Red,
    PlayerColor::Blue,
    PlayerColor::Green,
    PlayerColor::Yellow,
];

/// Start a 4-player game on `map` and return (state, player ids in join order).
fn start_game(map: Map, seed: u64) -> (GameState, Vec<PlayerId>) {
    let mut state = GameState::new_with_seed(map, 4, seed);
    let ids: Vec<PlayerId> = (0..4)
        .map(|i| PlayerId::from_u128(((seed as u128) << 8) | (i + 1) as u128))
        .collect();
    for (i, id) in ids.iter().enumerate() {
        apply_action(
            &mut state,
            *id,
            Action::JoinGame {
                name: format!("P{i}"),
                color: COLORS[i],
            },
        )
        .expect("join");
    }
    apply_action(&mut state, ids[0], Action::StartGame).expect("start");
    (state, ids)
}

fn make_bot(id: PlayerId, difficulty: BotDifficulty) -> Bot {
    let registry = default_registry();
    let profile = registry.profile_for(difficulty).clone();
    let mut bot = Bot::new(
        id,
        format!("{difficulty:?}"),
        PlayerColor::Red,
        profile,
        id.as_u128() as u64,
    );
    if difficulty == BotDifficulty::Expert {
        bot = bot.with_policy(policy::default_policy().expect("embedded policy must load"));
    }
    bot
}

#[test]
fn observation_and_mask_are_well_formed_on_default_map() {
    let (state, _) = start_game(default_map(), 42);
    let actor = encoding::current_actor_id(&state).expect("game has a current actor");

    let obs = encoding::build_observation(&state, actor);
    assert_eq!(obs.len(), encoding::OBS_SIZE);
    assert!(obs.iter().all(|v| (0.0..=1.0).contains(v)));

    let mask = encoding::build_action_mask(&state, actor);
    assert_eq!(mask.len(), encoding::N_ACTIONS);
    assert!(
        mask.iter().any(|&m| m != 0),
        "first actor must have a legal action"
    );
}

#[test]
fn map_matches_default_distinguishes_maps() {
    assert!(encoding::map_matches_default(&default_map()));
    let germany = Map::load(GERMANY_TOML).expect("parse germany map");
    assert!(!encoding::map_matches_default(&germany));
}

#[test]
fn expert_bot_falls_back_to_heuristic_on_non_default_map() {
    // Doesn't depend on the embedded policy loading at all (non-default map
    // always falls back), but skip explicitly attaching it since the embedded
    // expert.bin is currently a stale 143-action export — see
    // expert_bot_plays_policy_action_on_its_turn.
    let germany = Map::load(GERMANY_TOML).expect("parse germany map");
    let (state, ids) = start_game(germany, 7);
    let actor = match &state.phase {
        Phase::Auction {
            current_bidder_idx, ..
        } => state.player_order[*current_bidder_idx],
        other => panic!("expected auction phase after start, got {other:?}"),
    };
    assert!(ids.contains(&actor));

    let registry = default_registry();
    let profile = registry.profile_for(BotDifficulty::Expert).clone();
    let mut bot = Bot::new(actor, "Expert".to_string(), PlayerColor::Red, profile, 7);
    let action = bot.decide(&state);
    assert!(
        action.is_some(),
        "expert bot must produce a heuristic action on a non-default map"
    );
}

#[test]
#[ignore = "embedded expert.bin is a 26-action export; the buy-quantity ladder moved \
            N_ACTIONS to 29, so it fails MlpPolicy::from_bytes's dim check and the Expert \
            bot falls back to the heuristic. Un-ignore once a 29-macro policy is exported."]
fn expert_bot_plays_policy_action_on_its_turn() {
    let (state, _) = start_game(default_map(), 42);
    let actor = encoding::current_actor_id(&state).unwrap();

    let mut bot = make_bot(actor, BotDifficulty::Expert);
    let action = bot.decide(&state).expect("expert must act on its turn");
    // Round 1 auction: the policy may only select a plant (passing is illegal).
    assert!(
        matches!(action, Action::SelectPlant { .. }),
        "unexpected round-1 action: {action:?}"
    );

    // Not its turn: a different seat must stay silent rather than fall back.
    let other = state
        .players
        .iter()
        .map(|p| p.id)
        .find(|id| *id != actor)
        .unwrap();
    let mut other_bot = make_bot(other, BotDifficulty::Expert);
    assert!(other_bot.decide(&state).is_none());
}

/// Full-game strength measurement: 1 Expert vs 3 Hard bots. Run manually:
/// `cargo test -p powergrid-bot-strategy --release -- --ignored expert_vs_hard --nocapture`
///
/// This measures (not asserts) the win rate: expert strength is a property of
/// the exported checkpoint, not of this code. The torch reference for the same
/// checkpoint is `python scripts/evaluate.py` — the two should roughly agree.
/// Games hitting the step cap count as truncated (the policy can stall games
/// by never pushing the board to end_game_cities; so can its torch original).
#[test]
#[ignore = "slow; run manually to measure expert strength. Also requires a \
            29-macro policy export (see expert_bot_plays_policy_action_on_its_turn)"]
fn expert_vs_hard_win_rate() {
    const GAMES: u64 = 50;
    const STEP_CAP: usize = 5000;

    let mut expert_wins = 0u32;
    let mut finished = 0u32;

    for game in 0..GAMES {
        let (mut state, ids) = start_game(default_map(), 1000 + game);
        // Rotate the expert seat so turn-order advantage averages out.
        let expert_id = ids[(game % 4) as usize];
        let mut bots: Vec<Bot> = ids
            .iter()
            .map(|&id| {
                make_bot(
                    id,
                    if id == expert_id {
                        BotDifficulty::Expert
                    } else {
                        BotDifficulty::Hard
                    },
                )
            })
            .collect();

        for _ in 0..STEP_CAP {
            if matches!(state.phase, Phase::GameOver { .. }) {
                break;
            }
            // Mirror run_bot_pump: first bot with a move acts.
            let mut acted = false;
            for bot in bots.iter_mut() {
                if let Some(action) = bot.decide(&state) {
                    apply_action(&mut state, bot.id, action).expect("bot move must be legal");
                    acted = true;
                    break;
                }
            }
            assert!(acted, "no bot has a move in phase {:?}", state.phase);
        }

        if let Phase::GameOver { winner } = &state.phase {
            finished += 1;
            if *winner == expert_id {
                expert_wins += 1;
            }
        }
    }

    let win_rate = expert_wins as f32 / finished.max(1) as f32;
    println!(
        "expert won {expert_wins}/{finished} finished games (win rate {win_rate:.2}, \
         {} truncated at step cap)",
        GAMES as u32 - finished
    );
    assert!(
        finished > 0,
        "no game finished — harness or policy is broken"
    );
}
