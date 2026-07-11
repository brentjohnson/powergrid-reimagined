//! Evolutionary tuner for the heuristic bot's `BotProfile` weights (CMA-ES).
//!
//! Fitness = paired, deterministic (jitter=0) game outcomes of a candidate seat
//! vs a fixed opponent lineup over a common seed block, seat-rotated to remove
//! position bias. See `crates/powergrid-evolve/README.md` and RL-TRAINING-JOURNAL.md.

mod cmaes;
mod games;
mod genome;

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use powergrid_bot_strategy::{embedded_registry, BotProfile, ProfileRegistry};

use cmaes::CmaEs;
use games::{Fitness, Match};
use genome::N_PARAMS;

struct Config {
    out_dir: PathBuf,
    opponents: String,
    pool_dir: Option<PathBuf>,
    lambda: usize,
    games_per_eval: usize,
    seat_rotations: usize,
    generations: usize,
    seed_block_rotate: usize,
    seed_base: u64,
    sigma0: f64,
    num_players: usize,
    threads: usize,
    resume: Option<PathBuf>,
    cma_seed: u64,
    /// If set, skip training: load this profile's `hard` as the candidate,
    /// evaluate it once on the given seed block (jitter=0), print, and exit.
    eval_toml: Option<PathBuf>,
    /// If set, every non-candidate seat plays this profile's `hard` (overrides
    /// --opponents / --pool-dir). The sharp "can anything beat 3× this?" probe.
    opponent_toml: Option<PathBuf>,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            out_dir: PathBuf::from("runs/evolve1"),
            opponents: "normal".into(),
            pool_dir: None,
            lambda: 0,
            games_per_eval: 600,
            seat_rotations: 4,
            generations: 200,
            seed_block_rotate: 10,
            seed_base: 1,
            sigma0: 0.4,
            num_players: 4,
            threads: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
            resume: None,
            cma_seed: 42,
            eval_toml: None,
            opponent_toml: None,
        }
    }
}

fn parse_args() -> Config {
    let mut c = Config::default();
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut val = || {
            args.next()
                .unwrap_or_else(|| panic!("missing value for {flag}"))
        };
        match flag.as_str() {
            "--out-dir" => c.out_dir = PathBuf::from(val()),
            "--opponents" => c.opponents = val(),
            "--pool-dir" => c.pool_dir = Some(PathBuf::from(val())),
            "--pop" | "--lambda" => c.lambda = val().parse().unwrap(),
            "--games-per-eval" => c.games_per_eval = val().parse().unwrap(),
            "--seat-rotations" => c.seat_rotations = val().parse().unwrap(),
            "--generations" => c.generations = val().parse().unwrap(),
            "--seed-block-rotate" => c.seed_block_rotate = val().parse().unwrap(),
            "--seed-base" => c.seed_base = val().parse().unwrap(),
            "--sigma0" => c.sigma0 = val().parse().unwrap(),
            "--num-players" => c.num_players = val().parse().unwrap(),
            "--threads" => c.threads = val().parse().unwrap(),
            "--resume" => c.resume = Some(PathBuf::from(val())),
            "--cma-seed" => c.cma_seed = val().parse().unwrap(),
            "--eval-toml" => c.eval_toml = Some(PathBuf::from(val())),
            "--opponent-toml" => c.opponent_toml = Some(PathBuf::from(val())),
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            other => panic!("unknown flag: {other}"),
        }
    }
    assert!(
        c.seat_rotations >= 1 && c.seat_rotations <= c.num_players,
        "--seat-rotations must be in 1..=num_players"
    );
    assert!(
        c.games_per_eval % c.seat_rotations == 0,
        "--games-per-eval must be divisible by --seat-rotations"
    );
    c
}

fn print_help() {
    println!(
        "powergrid-evolve — CMA-ES tuner for BotProfile weights\n\n\
         --out-dir DIR            output directory (history.csv, best.toml, checkpoint.json)\n\
         --opponents KIND         normal|easy|hard|pool  (default normal)\n\
         --pool-dir DIR           directory of champion *.toml for --opponents pool\n\
         --pop N / --lambda N     population size (0 = auto)\n\
         --games-per-eval N       games per candidate per generation (default 600)\n\
         --seat-rotations R       seat rotations per base seed (default 4)\n\
         --generations N          number of generations (default 200)\n\
         --seed-block-rotate N    rotate the seed block every N generations (default 10)\n\
         --seed-base S            first base game seed (default 1)\n\
         --sigma0 F               initial step size (default 0.4)\n\
         --num-players N          players per game (default 4)\n\
         --threads N              worker threads (default: all cores)\n\
         --resume FILE            resume from a checkpoint.json\n\
         --cma-seed S             RNG seed for CMA sampling (default 42)\n\
         --eval-toml FILE         score this profile's `hard` on the seed block and exit\n\
                                  (no training; use --seed-base 90000+ for held-out)\n\
         --opponent-toml FILE     every opponent seat plays this profile's `hard`\n\
                                  (overrides --opponents; the beat-3x-champion probe)"
    );
}

/// Opponent profiles (noise silenced) that fill the non-candidate seats.
fn build_opponents(cfg: &Config, reg: &ProfileRegistry) -> Vec<BotProfile> {
    let sil = |mut p: BotProfile| {
        genome::silence_noise(&mut p);
        p
    };
    // A single fixed opponent profile (its `hard`) fills every non-candidate
    // seat — no normal anchor, no mixing. This is the sharp exploitability probe:
    // "can anything beat 3 copies of THIS profile?" (e.g. the current champion).
    if let Some(path) = &cfg.opponent_toml {
        let text = fs::read_to_string(path).expect("read --opponent-toml");
        let reg: ProfileRegistry =
            toml::from_str(&text).expect("--opponent-toml must be a ProfileRegistry TOML");
        println!("opponents: 3× {:?} (hard)", path);
        return vec![sil(reg.hard)];
    }
    match cfg.opponents.as_str() {
        "normal" => vec![sil(reg.normal.clone())],
        "easy" => vec![sil(reg.easy.clone())],
        "hard" => vec![sil(reg.hard.clone())],
        "pool" => {
            let dir = cfg
                .pool_dir
                .as_ref()
                .expect("--opponents pool requires --pool-dir");
            let mut pool = vec![sil(reg.normal.clone())]; // normal anchor
            for entry in fs::read_dir(dir).expect("read pool-dir") {
                let path = entry.unwrap().path();
                if path.extension().and_then(|e| e.to_str()) == Some("toml") {
                    let text = fs::read_to_string(&path).unwrap();
                    let champ: ProfileRegistry =
                        toml::from_str(&text).expect("pool profile must be a ProfileRegistry TOML");
                    pool.push(sil(champ.hard));
                }
            }
            println!(
                "loaded {} pool opponent profiles from {:?}",
                pool.len(),
                dir
            );
            pool
        }
        other => panic!("unknown --opponents {other}"),
    }
}

/// Build the (fixed-for-this-generation) match schedule.
fn build_schedule(cfg: &Config, gen: usize, n_opponents: usize) -> Vec<Match> {
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    let seeds_per_block = cfg.games_per_eval / cfg.seat_rotations;
    let rotation = gen / cfg.seed_block_rotate;
    let offset = cfg.seed_base + (rotation * seeds_per_block) as u64;

    let mut schedule = Vec::with_capacity(cfg.games_per_eval);
    for s in 0..seeds_per_block {
        let seed = offset + s as u64;
        for r in 0..cfg.seat_rotations {
            let candidate_seat = r % cfg.num_players;
            // Deterministic opponent picks per (seed, seat) — identical for every
            // candidate this generation, so comparisons stay paired.
            let mut rng = SmallRng::seed_from_u64(seed ^ ((r as u64) << 32));
            let opponent_pick: Vec<usize> = (0..cfg.num_players - 1)
                .map(|_| rng.gen_range(0..n_opponents))
                .collect();
            schedule.push(Match {
                seed,
                candidate_seat,
                opponent_pick,
            });
        }
    }
    schedule
}

/// Serialize a champion vector as a full ProfileRegistry TOML (easy/normal kept
/// from the embedded defaults so the eval yardstick is untouched; hard = expert
/// = champion, with hard's original noise restored for in-game variety).
fn write_best_toml(path: &Path, base_reg: &ProfileRegistry, init_raw: &[f64; N_PARAMS], x: &[f64]) {
    let raw = genome::x_to_raw(init_raw, x);
    let mut champ = genome::apply_raw(&base_reg.hard, &raw);
    // Restore shipped noise (fitness silenced it; deployed play keeps jitter).
    champ.temperature = base_reg.hard.temperature;
    champ.jitter = base_reg.hard.jitter;
    champ.max_jitter = base_reg.hard.max_jitter;

    let out = ProfileRegistry {
        easy: base_reg.easy.clone(),
        normal: base_reg.normal.clone(),
        hard: champ.clone(),
        expert: champ,
    };
    let text = toml::to_string_pretty(&out).expect("serialize champion registry");
    fs::write(path, text).expect("write best.toml");
}

fn main() {
    let cfg = parse_args();

    let base_reg = embedded_registry();

    // Eval-only gate: score a champion TOML on a (held-out) seed block, no training.
    if let Some(path) = &cfg.eval_toml {
        eval_only(&cfg, &base_reg, path);
        return;
    }

    fs::create_dir_all(&cfg.out_dir).expect("create out-dir");

    let init_raw = genome::profile_to_raw(&base_reg.hard);
    let opponents = build_opponents(&cfg, &base_reg);

    // Optimizer: fresh or resumed.
    let mut es = if let Some(path) = &cfg.resume {
        let text = fs::read_to_string(path).expect("read resume checkpoint");
        let mut es: CmaEs = serde_json::from_str(&text).expect("parse checkpoint");
        es.after_load();
        println!("resumed from {:?} at generation {}", path, es.gen);
        es
    } else {
        CmaEs::new(N_PARAMS, cfg.sigma0, cfg.lambda, cfg.cma_seed)
    };

    let history_path = cfg.out_dir.join("history.csv");
    let mut history = open_history(&history_path, cfg.resume.is_some());

    let mut best_win_rate = f64::NEG_INFINITY;
    let start_gen = es.gen;
    let end_gen = start_gen + cfg.generations;

    for gen in start_gen..end_gen {
        let t0 = std::time::Instant::now();
        let schedule = build_schedule(&cfg, gen, opponents.len());

        // Evaluate the population.
        let batch = es.ask();
        let mut fits = Vec::with_capacity(batch.len());
        let mut pop_best = Fitness::default();
        let mut pop_best_win = f64::NEG_INFINITY;
        for x in &batch {
            let prof = genome::x_to_eval_profile(&base_reg.hard, &init_raw, x);
            let f = games::evaluate(&prof, &opponents, &schedule, cfg.num_players, cfg.threads);
            if f.win_rate > pop_best_win {
                pop_best_win = f.win_rate;
                pop_best = f;
            }
            fits.push(-f.mean_rank_value); // CMA minimizes
        }
        es.tell(&fits);

        // Evaluate the distribution mean — the shippable candidate this gen.
        let mean_prof = genome::x_to_eval_profile(&base_reg.hard, &init_raw, &es.mean);
        let mean_fit = games::evaluate(
            &mean_prof,
            &opponents,
            &schedule,
            cfg.num_players,
            cfg.threads,
        );

        if mean_fit.win_rate > best_win_rate {
            best_win_rate = mean_fit.win_rate;
            write_best_toml(
                &cfg.out_dir.join("best.toml"),
                &base_reg,
                &init_raw,
                &es.mean,
            );
        }

        // Persist resumable state every generation.
        let ckpt = serde_json::to_string(&es).unwrap();
        fs::write(cfg.out_dir.join("checkpoint.json"), ckpt).expect("write checkpoint");

        let secs = t0.elapsed().as_secs_f64();
        write_history_row(
            &mut history,
            gen,
            &es,
            &mean_fit,
            pop_best_win,
            &init_raw,
            secs,
        );
        println!(
            "gen {gen:4}  mean_win {:.3} ({} games)  mean_rankval {:+.3}  pop_best_win {:.3}  sigma {:.3}  aborted {}  {:.1}s",
            mean_fit.win_rate, mean_fit.games, mean_fit.mean_rank_value, pop_best_win, es.sigma,
            mean_fit.aborted + pop_best.aborted, secs
        );
    }

    // Always leave a best.toml, even if the mean never beat gen-0 (writes gen-0).
    if best_win_rate == f64::NEG_INFINITY {
        write_best_toml(
            &cfg.out_dir.join("best.toml"),
            &base_reg,
            &init_raw,
            &es.mean,
        );
    }
    println!(
        "done. best mean win rate {:.3}. outputs in {:?}",
        best_win_rate, cfg.out_dir
    );
}

/// Score one champion profile's `hard` seat vs the configured opponents on the
/// current seed block (jitter=0 — same paired methodology as training). Use
/// `--seed-base 90000+` for an honest held-out gate. Prints win rate / rank value.
fn eval_only(cfg: &Config, base_reg: &ProfileRegistry, path: &Path) {
    let text = fs::read_to_string(path).expect("read --eval-toml");
    let champ: ProfileRegistry =
        toml::from_str(&text).expect("--eval-toml must be a ProfileRegistry TOML");
    let mut candidate = champ.hard;
    genome::silence_noise(&mut candidate);

    let opponents = build_opponents(cfg, base_reg);
    // Reuse the training schedule builder (gen 0 → seed block starts at seed_base).
    let schedule = build_schedule(cfg, 0, opponents.len());
    let fit = games::evaluate(
        &candidate,
        &opponents,
        &schedule,
        cfg.num_players,
        cfg.threads,
    );
    println!(
        "eval {:?} vs {} | seeds {}..{} | win_rate {:.4}  rank_value {:+.4}  games {}  aborted {}",
        path,
        cfg.opponents,
        cfg.seed_base,
        cfg.seed_base + (cfg.games_per_eval / cfg.seat_rotations) as u64,
        fit.win_rate,
        fit.mean_rank_value,
        fit.games,
        fit.aborted,
    );
}

fn open_history(path: &Path, resuming: bool) -> fs::File {
    let exists = path.exists();
    let mut f = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .expect("open history.csv");
    if !exists || !resuming {
        // Fresh file (or overwriting a non-resume run): (re)write the header.
        if !resuming && exists {
            f = fs::File::create(path).expect("truncate history.csv");
        }
        let mut header = String::from(
            "gen,evals,sigma,mean_win_rate,mean_rank_value,pop_best_win_rate,elapsed_s",
        );
        for s in genome::SPECS.iter() {
            header.push(',');
            header.push_str(s.name);
        }
        writeln!(f, "{header}").unwrap();
    }
    f
}

fn write_history_row(
    f: &mut fs::File,
    gen: usize,
    es: &CmaEs,
    mean_fit: &Fitness,
    pop_best_win: f64,
    init_raw: &[f64; N_PARAMS],
    secs: f64,
) {
    let raw = genome::x_to_raw(init_raw, &es.mean);
    let mut row = format!(
        "{gen},{},{:.5},{:.5},{:.5},{:.5},{:.2}",
        es.counteval, es.sigma, mean_fit.win_rate, mean_fit.mean_rank_value, pop_best_win, secs
    );
    for v in raw.iter() {
        row.push_str(&format!(",{:.4}", v));
    }
    writeln!(f, "{row}").unwrap();
    f.flush().unwrap();
}
