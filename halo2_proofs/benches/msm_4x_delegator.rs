use std::sync::{Arc, Barrier};
use std::time::Instant;

use ff::PrimeField;
use halo2_proofs::arithmetic::CurveAffine;
use halo2_proofs::pasta::EpAffine;
use halo2_proofs::poly::commitment::Params;

fn black_box<T>(x: T) -> T {
    unsafe { std::ptr::read_volatile(&x) }
}

/// Pin the calling thread to a specific logical core.
///
/// - Linux/Android: hard pin via `sched_setaffinity`.
/// - macOS: no hard pin exists; set a distinct affinity tag per worker
///   (spreads threads across cores/clusters as a scheduler hint) and lift
///   the QoS class so workers stay on performance cores.
#[cfg(feature = "fuji")]
fn pin_worker(tid: usize, count: usize) {
    #[cfg(target_os = "linux")]
    {
        let mut cpuset: libc::cpu_set_t = unsafe { std::mem::zeroed() };
        unsafe { libc::CPU_SET(tid % count, &mut cpuset) };
        let rc = unsafe {
            libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &cpuset)
        };
        if rc != 0 {
            eprintln!("  [pin] sched_setaffinity({}) failed: {}", tid % count, std::io::Error::last_os_error());
        }
    }
    #[cfg(target_os = "macos")]
    {
        let _ = count;
        unsafe {
            let rc = libc::pthread_set_qos_class_self_np(
                libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE,
                0,
            );
            if rc != 0 {
                eprintln!("  [pin] pthread_set_qos_class_self_np failed: {}", std::io::Error::last_os_error());
            }
            let policy = libc::thread_affinity_policy_data_t {
                affinity_tag: (tid as i32) + 1,
            };
            let rc = libc::thread_policy_set(
                libc::pthread_mach_thread_np(libc::pthread_self()),
                libc::THREAD_AFFINITY_POLICY as libc::thread_policy_flavor_t,
                &policy as *const _ as libc::thread_policy_t,
                libc::THREAD_AFFINITY_POLICY_COUNT,
            );
            if rc != 0 {
                eprintln!("  [pin] thread_policy_set kern_return={}", rc);
            }
        }
    }
}

// ── Benchmark configuration ──────────────────────────────────────
// Default k range 18..21; override with FUJI_K_START / FUJI_K_END.

fn main() {
    #[cfg(feature = "fuji")]
    run();
}

#[cfg(feature = "fuji")]
fn fill_scalars(out: &mut [fuji::FujiField], seed: u64) {
    let mut x = seed.wrapping_add(0x9e3779b97f4a7c15);
    for s in out.iter_mut() {
        x = x.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        let mut b = [0u8; 32];
        b[..8].copy_from_slice(&z.to_le_bytes());
        b[8..16].copy_from_slice(&(z >> 8).to_le_bytes());
        b[16..24].copy_from_slice(&(z >> 16).to_le_bytes());
        b[24..32].copy_from_slice(&(z >> 24).to_le_bytes());
        *s = fuji::FujiField::from_bytes(&b);
    }
}

#[cfg(feature = "fuji")]
fn run() {
    use fuji::{FujiAffine, FujiCurve, FujiField};

    let curve = FujiCurve::Pallas;

    let threads = std::env::var("FUJI_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1))
        .max(1);
    let rounds = std::env::var("FUJI_ROUNDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3usize)
        .max(1);
    let pin = std::env::var("FUJI_PIN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1i32) != 0;

    let k_start = std::env::var("FUJI_K_START").ok().and_then(|s| s.parse().ok()).unwrap_or(18);
    let k_end = std::env::var("FUJI_K_END").ok().and_then(|s| s.parse().ok()).unwrap_or(21);

    println!("\nlogical cores: {}, threads={}, rounds={}, pin={}",
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0), threads, rounds, pin);

    for k in k_start..k_end {
        let pb_ = Instant::now();
        let params = Params::<EpAffine>::load_or_init(k);
        let pb_t = pb_.elapsed().as_secs_f64() * 1000.0;
        println!("\nParams::<EpAffine>::load_or_init({}) took {} millis...", k, pb_t);
        let n = 1usize << k;

        let bases_srs_mont: Vec<FujiAffine> = params.get_g().iter().map(|base| {
            let coords = base.coordinates().unwrap();
            let mut xb = [0u8; 32]; xb.copy_from_slice(coords.x().to_repr().as_ref());
            let mut yb = [0u8; 32]; yb.copy_from_slice(coords.y().to_repr().as_ref());
            FujiAffine::from_coordinates(
                FujiField::from_bytes(&xb).to_mont(curve),
                FujiField::from_bytes(&yb).to_mont(curve),
            )
        }).collect();

        let flat_scalars: Vec<Vec<FujiField>> = (0..threads)
            .map(|t| {
                let mut v = vec![FujiField::from_bytes(&[0u8; 32]); 4 * n];
                fill_scalars(&mut v, t as u64);
                v
            })
            .collect();

        println!("  scalars: {:.1} MiB/thread  bases: {:.1} MiB shared",
            (4 * n * 32) as f64 / (1024.0 * 1024.0),
            (n * 64) as f64 / (1024.0 * 1024.0));

        // Single-thread baseline (also warms up AMX detection + page faults)
        let single_start = Instant::now();
        let r = fuji::msm::prl_pippenger_batch_4(&flat_scalars[0], &bases_srs_mont, curve).unwrap();
        black_box(r);
        let single_ms = single_start.elapsed().as_secs_f64() * 1000.0;
        println!("  single-core batch-4: {:>8.3} ms  ({:.2} MSMs/s)",
            single_ms, 4.0 / (single_ms / 1000.0));

        // Delegated fleet: threads × (one batch-4 per round), barrier-synced
        let barrier = Arc::new(Barrier::new(threads));
        let bases_arc = Arc::new(bases_srs_mont);
        let start = Instant::now();
        let handles: Vec<_> = flat_scalars
            .into_iter()
            .enumerate()
            .map(|(tid, scalars)| {
                let barrier = barrier.clone();
                let bases = bases_arc.clone();
                std::thread::spawn(move || {
                    if pin {
                        pin_worker(tid, threads);
                    }
                    for _ in 0..rounds {
                        barrier.wait();
                        let r = fuji::msm::prl_pippenger_batch_4(&scalars, &bases, curve).unwrap();
                        black_box(r);
                        barrier.wait();
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        let elapsed = start.elapsed();

        let total_msms = threads * 4 * rounds;
        let round_ms = elapsed.as_secs_f64() * 1000.0 / rounds as f64;
        let msms_per_sec = total_msms as f64 / elapsed.as_secs_f64();
        let single_rate = 4.0 / (single_ms / 1000.0);
        let scaling = msms_per_sec / single_rate;

        println!("prl-srs-batch-deleg-4x/k={:<2}: {:>8.3} ms/round  {} thr  {} MSMs in {:.1} ms = {:.2} MSMs/s  ({:.2}x single-core scaling, {:.1}% of linear)",
            k, round_ms, threads, total_msms, elapsed.as_secs_f64() * 1000.0,
            msms_per_sec, scaling, scaling / threads as f64 * 100.0);

        use std::io::Write;
        std::io::stdout().flush().ok();
    }
}
