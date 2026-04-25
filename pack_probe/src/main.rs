// pack_probe — throwaway Rust loader to introspect the runtime pack.
//
// Phase A: untyped serde_json::Value walk. Print top-level shape, object_type
//          histogram, and one full sample object per object_type.
// Phase B: minimal typed deserialization. Catalog every serde failure as a
//          probe target back on the parser side.
//
// NOT a real runtime. No game logic. Read-and-introspect only.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::Value;

const PACK_DIR: &str = "../parser/out/runtime_pack";

fn main() -> Result<()> {
    let pack_dir = Path::new(PACK_DIR);
    let manifest_path = pack_dir.join("manifest.json");
    let objects_path = pack_dir.join("object_bank/objects.json");

    println!("=== pack_probe — runtime pack introspection ===");
    println!("pack dir : {}", pack_dir.display());
    println!();

    phase_a_untyped(&manifest_path, &objects_path)?;
    println!();
    phase_b_typed(&objects_path)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Phase A — Untyped Walk
// ---------------------------------------------------------------------------

fn phase_a_untyped(manifest_path: &Path, objects_path: &Path) -> Result<()> {
    println!("--- Phase A: untyped Value walk ---");

    let manifest: Value = load_json(manifest_path)?;
    println!();
    println!("[manifest.json] top-level keys:");
    print_top_level(&manifest);

    let objects_doc: Value = load_json(objects_path)?;
    println!();
    println!("[objects.json] top-level keys:");
    print_top_level(&objects_doc);

    // Aggregate header fields we expect.
    println!();
    println!("[objects.json] header values:");
    for key in [
        "active_count",
        "active_decoded_count",
        "active_animation_frames",
        "active_animation_directions",
        "count",
    ] {
        if let Some(v) = objects_doc.get(key) {
            println!("  {key} = {v}");
        }
    }

    // Object_type histogram + one sample per type.
    let objects = objects_doc
        .get("objects")
        .and_then(Value::as_array)
        .context("objects.json missing 'objects' array")?;

    let mut histogram: BTreeMap<String, usize> = BTreeMap::new();
    let mut sample_per_type: BTreeMap<String, &Value> = BTreeMap::new();

    for obj in objects {
        let ty = obj
            .get("object_type")
            .map(|v| match v {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            })
            .unwrap_or_else(|| "<missing object_type>".to_string());
        *histogram.entry(ty.clone()).or_insert(0) += 1;
        sample_per_type.entry(ty).or_insert(obj);
    }

    println!();
    println!("[objects.json] object_type histogram (total {}):", objects.len());
    for (ty, n) in &histogram {
        println!("  {ty:>30}  {n:>4}");
    }

    println!();
    println!("[objects.json] one sample object per type (keys only):");
    for (ty, sample) in &sample_per_type {
        let keys = sample
            .as_object()
            .map(|m| m.keys().cloned().collect::<Vec<_>>())
            .unwrap_or_default();
        println!("  {ty}: {keys:?}");
    }

    println!();
    println!("[objects.json] FULL DUMP — one sample object per type (first 60 lines):");
    for (ty, sample) in &sample_per_type {
        println!("  --- {ty} ---");
        let pretty = serde_json::to_string_pretty(sample)?;
        for line in pretty.lines().take(60) {
            println!("    {line}");
        }
        let nlines = pretty.lines().count();
        if nlines > 60 {
            println!("    ... ({} more lines)", nlines - 60);
        }
    }

    Ok(())
}

fn print_top_level(v: &Value) {
    if let Some(map) = v.as_object() {
        for (k, val) in map {
            let summary = match val {
                Value::Array(a) => format!("<array len={}>", a.len()),
                Value::Object(m) => format!("<object keys={}>", m.len()),
                Value::String(s) if s.len() > 60 => format!("\"{}…\"", &s[..60]),
                other => other.to_string(),
            };
            println!("  {k}: {summary}");
        }
    } else {
        println!("  <top-level is not an object: {}>", v);
    }
}

fn load_json(path: &Path) -> Result<Value> {
    let raw = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let v: Value = serde_json::from_str(&raw)
        .with_context(|| format!("parsing {} as JSON", path.display()))?;
    Ok(v)
}

// ---------------------------------------------------------------------------
// Phase B — Typed Walk (minimal envelope structs)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct ObjectsFile {
    active_count: u32,
    active_decoded_count: u32,
    active_animation_frames: u32,
    active_animation_directions: u32,
    count: u32,
    objects: Vec<Value>, // intentionally untyped — inner pass is future work
}

#[derive(Debug, Deserialize)]
struct Manifest {
    pack_version: String,
    decoder_version: String,
    source_file: String,
    source_sha256: String,
    counts: BTreeMap<String, Value>,
    files: BTreeMap<String, Value>,
}

fn phase_b_typed(objects_path: &Path) -> Result<()> {
    println!("--- Phase B: typed envelope deserialization ---");

    let manifest_raw = fs::read_to_string(Path::new(PACK_DIR).join("manifest.json"))?;
    let objects_raw = fs::read_to_string(objects_path)?;

    let manifest: Manifest = serde_json::from_str(&manifest_raw)
        .context("manifest.json failed typed deserialization")?;
    println!(
        "  manifest OK : pack={} decoder={} source={} files={} counts_keys={}",
        manifest.pack_version,
        manifest.decoder_version,
        manifest.source_file,
        manifest.files.len(),
        manifest.counts.len(),
    );

    let objects: ObjectsFile = serde_json::from_str(&objects_raw)
        .context("objects.json failed typed deserialization")?;
    println!(
        "  objects OK  : count={} active_count={} active_decoded_count={} \
         active_animation_frames={} active_animation_directions={} objects.len()={}",
        objects.count,
        objects.active_count,
        objects.active_decoded_count,
        objects.active_animation_frames,
        objects.active_animation_directions,
        objects.objects.len(),
    );

    // Oracle assertions (Layer 1 — deterministic).
    let mut failures = Vec::new();
    if objects.active_count != 124 {
        failures.push(format!("active_count={} expected 124", objects.active_count));
    }
    if objects.active_decoded_count != objects.active_count {
        failures.push(format!(
            "active_decoded_count={} != active_count={}",
            objects.active_decoded_count, objects.active_count
        ));
    }
    if objects.objects.len() as u32 != objects.count {
        failures.push(format!(
            "objects array len {} != header count {}",
            objects.objects.len(),
            objects.count,
        ));
    }
    if objects.count != 196 {
        failures.push(format!(
            "total object count={} expected 196 (124A+44C+15B+10T+3E)",
            objects.count,
        ));
    }

    println!();
    if failures.is_empty() {
        println!("  ORACLE: all layer-1 assertions PASS");
    } else {
        println!("  ORACLE: {} layer-1 assertion failures:", failures.len());
        for f in &failures {
            println!("    - {f}");
        }
    }

    Ok(())
}
