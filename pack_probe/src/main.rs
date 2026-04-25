// pack_probe — throwaway Rust loader to introspect the runtime pack.
//
// Phase A: untyped serde_json::Value walk. Print top-level shape, object_type
//          histogram, and one full sample object per object_type.
// Phase B: minimal typed deserialization at THREE levels of strictness:
//          (1) envelope on every object — catches header/envelope drift,
//          (2) per-object_type inner deserialize:
//                - Active (124): full ActiveTyped (animations + summary),
//                - Counter (44): CounterPropertiesSummary (display_style + handles),
//                - non-(Active|Counter) (28): assert "no decode yet" null pattern.
//          Catalog every failure as a probe target back on the parser side.
//
// NOT a real runtime. No game logic. Read-and-introspect only.

use std::collections::{BTreeMap, BTreeSet};
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
        "counter_count",
        "counter_decoded_count",
        "counter_total_handle_refs",
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
    active_unique_image_handles: Vec<u32>,
    counter_count: u32,
    counter_decoded_count: u32,
    counter_total_handle_refs: u32,
    counter_unique_image_handles: Vec<u32>,
    count: u32,
    #[serde(default)]
    deferred_sub_chunk_ids_seen: Vec<String>,
    objects: Vec<Value>, // walked again below as typed envelopes / typed inner
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

// ---- Object envelope (shared by all 7 object_types) ----

#[derive(Debug, Deserialize)]
#[allow(dead_code)] // schema-receiver: fields exist to assert shape, not be read
#[serde(deny_unknown_fields)]
struct ObjectEnvelope {
    handle: u32,
    header: ObjectHeader,
    name: String,
    object_type: u32,
    object_type_name: String,
    properties_decoded: bool,
    properties_len: u32,
    properties_summary: Option<Value>, // typed per-object_type below
    animations: Option<Value>,         // typed per-object_type below
    effects_len: u32,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)] // schema-receiver: fields exist to assert shape, not be read
#[serde(deny_unknown_fields)]
struct ObjectHeader {
    antialias: bool,
    flags: u32,
    handle: u32,
    ink_effect: u32,
    ink_effect_id: u32,
    ink_effect_param: u32,
    object_type: u32,
    object_type_name: String,
    reserved: i32, // can be negative, e.g. Text reserved=-16
    transparent: bool,
}

// ---- Active-specific inner shape ----

#[derive(Debug, Deserialize)]
#[allow(dead_code)] // schema-receiver: fields exist to assert shape, not be read
#[serde(deny_unknown_fields)]
struct ActiveAnimations {
    items: Vec<Animation>,
    summary: AnimationsSummary,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)] // schema-receiver: fields exist to assert shape, not be read
#[serde(deny_unknown_fields)]
struct Animation {
    animation_index: u32,
    animation_name: String,
    directions: Vec<Direction>,
    image_handles: Vec<u32>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)] // schema-receiver: fields exist to assert shape, not be read
#[serde(deny_unknown_fields)]
struct Direction {
    back_to: u32,
    direction_index: u32,
    frame_count: u32,
    image_handles: Vec<u32>,
    max_speed: u32,
    min_speed: u32,
    repeat: u32,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)] // schema-receiver: fields exist to assert shape, not be read
#[serde(deny_unknown_fields)]
struct AnimationsSummary {
    non_empty_animations: u32,
    total_animations: u32,
    total_directions: u32,
    total_frames: u32,
    unique_image_handles: Vec<u32>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)] // schema-receiver: fields exist to assert shape, not be read
#[serde(deny_unknown_fields)]
struct ActivePropertiesSummary {
    known_zero_pad_gaps: u32,
    movements_decoded: bool,
    movements_raw_len: u32,
    non_empty_animations: u32,
    opaque_tables: BTreeMap<String, u32>,
    total_animations: u32,
    total_directions: u32,
    total_frames: u32,
    unconsumed_gap_count: u32,
    unique_image_handles: Vec<u32>,
}

// ---- Counter-specific inner shape ----
//
// FNAF 1 Counter property body decodes into display_style (u32 enum) + an
// image-handle list (u16-prefixed, max 14 in baseline). The per-object
// `properties_summary` carries those plus a deduped `unique_image_handles`
// view of the handles list and the on-wire body `size`.
#[derive(Debug, Deserialize)]
#[allow(dead_code)] // schema-receiver: fields exist to assert shape, not be read
#[serde(deny_unknown_fields)]
struct CounterPropertiesSummary {
    size: u32,
    display_style: u32,
    image_handle_count: u32,
    unique_image_handles: Vec<u32>,
}

fn phase_b_typed(objects_path: &Path) -> Result<()> {
    println!("--- Phase B: typed deserialization ---");

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

    // Failure catalog — every entry is a parser-side probe target.
    let mut failures: Vec<String> = Vec::new();

    // ---- Layer-1 envelope counts ----
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

    // ---- Per-object typed walk ----
    //
    // 1. Every object → ObjectEnvelope (deny_unknown_fields). Catches schema drift
    //    on the shared shape (handle/header/name/object_type/...).
    // 2. Active (object_type==2) → ActiveAnimations + ActivePropertiesSummary
    //    (deny_unknown_fields). Catches drift in the rich Active inner shape.
    // 3. Non-Active (other types) → must currently match the "no decode yet"
    //    null pattern: properties_decoded=false, properties_summary=null,
    //    animations=null. When parser ships Counter/Backdrop/Text/Extension
    //    decoders, this assertion will start failing — that's the trigger to
    //    add typed structs for those types.

    let mut envelope_ok = 0_u32;
    let mut active_typed_ok = 0_u32;
    let mut counter_typed_ok = 0_u32;
    let mut nonactive_null_ok = 0_u32;

    // Accumulate Active-side stats once typed deserialization succeeds.
    let mut active_max_anim_items = 0_u32;
    let mut active_total_directions = 0_u32;
    let mut active_total_frames = 0_u32;
    let mut active_total_unique_handles = 0_u32;
    let mut walked_active_unique_handles: BTreeSet<u32> = BTreeSet::new();

    // Accumulate Counter-side stats once typed deserialization succeeds.
    let mut walked_counter_total_handle_refs = 0_u32;
    let mut walked_counter_unique_handles: BTreeSet<u32> = BTreeSet::new();

    for raw in &objects.objects {
        // (1) envelope
        let envelope: ObjectEnvelope = match serde_json::from_value(raw.clone()) {
            Ok(e) => {
                envelope_ok += 1;
                e
            }
            Err(err) => {
                let handle = raw.get("handle").map(|v| v.to_string()).unwrap_or_default();
                let name = raw.get("name").and_then(Value::as_str).unwrap_or("?");
                failures.push(format!(
                    "envelope deserialize failed (handle={handle} name={name:?}): {err}"
                ));
                continue;
            }
        };

        // Cross-check: header.object_type / header.handle should match envelope.
        if envelope.header.object_type != envelope.object_type {
            failures.push(format!(
                "header.object_type={} != envelope.object_type={} (handle={} name={:?})",
                envelope.header.object_type,
                envelope.object_type,
                envelope.handle,
                envelope.name,
            ));
        }
        if envelope.header.handle != envelope.handle {
            failures.push(format!(
                "header.handle={} != envelope.handle={} (name={:?})",
                envelope.header.handle, envelope.handle, envelope.name,
            ));
        }

        // (2) per-object_type inner
        match envelope.object_type {
            2 => {
                // Active — full typed inner expected.
                if !envelope.properties_decoded {
                    failures.push(format!(
                        "Active handle={} name={:?} has properties_decoded=false (expected true)",
                        envelope.handle, envelope.name,
                    ));
                    continue;
                }
                let anims_v = match envelope.animations.as_ref() {
                    Some(v) => v,
                    None => {
                        failures.push(format!(
                            "Active handle={} name={:?} has animations=null",
                            envelope.handle, envelope.name,
                        ));
                        continue;
                    }
                };
                let summary_v = match envelope.properties_summary.as_ref() {
                    Some(v) => v,
                    None => {
                        failures.push(format!(
                            "Active handle={} name={:?} has properties_summary=null",
                            envelope.handle, envelope.name,
                        ));
                        continue;
                    }
                };
                let anims: ActiveAnimations = match serde_json::from_value(anims_v.clone()) {
                    Ok(a) => a,
                    Err(e) => {
                        failures.push(format!(
                            "Active handle={} name={:?} animations typed deser failed: {e}",
                            envelope.handle, envelope.name,
                        ));
                        continue;
                    }
                };
                let summary: ActivePropertiesSummary =
                    match serde_json::from_value(summary_v.clone()) {
                        Ok(s) => s,
                        Err(e) => {
                            failures.push(format!(
                                "Active handle={} name={:?} properties_summary typed deser failed: {e}",
                                envelope.handle, envelope.name,
                            ));
                            continue;
                        }
                    };

                // Cross-check: summary stats vs items walk.
                // animations.items contains ONLY non-empty animation slots
                // (vanilla Active has 16 slots, most are empty). So walked items
                // must equal non_empty_animations, not total_animations.
                let walked_items = anims.items.len() as u32;
                let walked_dirs: u32 = anims.items.iter().map(|a| a.directions.len() as u32).sum();
                let walked_frames: u32 = anims
                    .items
                    .iter()
                    .flat_map(|a| a.directions.iter())
                    .map(|d| d.frame_count)
                    .sum();
                if walked_items != summary.non_empty_animations {
                    failures.push(format!(
                        "Active handle={} name={:?} animations.items.len()={} != properties_summary.non_empty_animations={}",
                        envelope.handle, envelope.name, walked_items, summary.non_empty_animations,
                    ));
                }
                if anims.summary.non_empty_animations != summary.non_empty_animations {
                    failures.push(format!(
                        "Active handle={} name={:?} animations.summary.non_empty_animations={} != properties_summary.non_empty_animations={}",
                        envelope.handle,
                        envelope.name,
                        anims.summary.non_empty_animations,
                        summary.non_empty_animations,
                    ));
                }
                if walked_dirs != summary.total_directions {
                    failures.push(format!(
                        "Active handle={} name={:?} walked directions={} != properties_summary.total_directions={}",
                        envelope.handle, envelope.name, walked_dirs, summary.total_directions,
                    ));
                }
                if walked_frames != summary.total_frames {
                    failures.push(format!(
                        "Active handle={} name={:?} walked frames={} != properties_summary.total_frames={}",
                        envelope.handle, envelope.name, walked_frames, summary.total_frames,
                    ));
                }

                active_max_anim_items = active_max_anim_items.max(walked_items);
                active_total_directions += walked_dirs;
                active_total_frames += walked_frames;
                active_total_unique_handles += summary.unique_image_handles.len() as u32;
                for d in anims.items.iter().flat_map(|a| a.directions.iter()) {
                    for h in &d.image_handles {
                        walked_active_unique_handles.insert(*h);
                    }
                }
                active_typed_ok += 1;
            }
            7 => {
                // Counter — typed inner expected (decoder shipped 2026-04-25).
                // animations must be null (Counter has no animation table);
                // properties_summary must typed-deserialize as
                // CounterPropertiesSummary.
                if !envelope.properties_decoded {
                    failures.push(format!(
                        "Counter handle={} name={:?} has properties_decoded=false (expected true)",
                        envelope.handle, envelope.name,
                    ));
                    continue;
                }
                if envelope.animations.is_some() {
                    failures.push(format!(
                        "Counter handle={} name={:?} has animations present (expected null)",
                        envelope.handle, envelope.name,
                    ));
                }
                let summary_v = match envelope.properties_summary.as_ref() {
                    Some(v) => v,
                    None => {
                        failures.push(format!(
                            "Counter handle={} name={:?} has properties_summary=null",
                            envelope.handle, envelope.name,
                        ));
                        continue;
                    }
                };
                let summary: CounterPropertiesSummary =
                    match serde_json::from_value(summary_v.clone()) {
                        Ok(s) => s,
                        Err(e) => {
                            failures.push(format!(
                                "Counter handle={} name={:?} properties_summary typed deser failed: {e}",
                                envelope.handle, envelope.name,
                            ));
                            continue;
                        }
                    };

                // Cross-check: unique_image_handles is the deduped view of the
                // on-wire handle list, so |unique| <= image_handle_count.
                if summary.unique_image_handles.len() as u32 > summary.image_handle_count {
                    failures.push(format!(
                        "Counter handle={} name={:?} unique_image_handles({}) > image_handle_count({})",
                        envelope.handle,
                        envelope.name,
                        summary.unique_image_handles.len(),
                        summary.image_handle_count,
                    ));
                }

                walked_counter_total_handle_refs += summary.image_handle_count;
                for h in &summary.unique_image_handles {
                    walked_counter_unique_handles.insert(*h);
                }
                counter_typed_ok += 1;
            }
            _ => {
                // Non-(Active|Counter) — no decoder shipped yet. Assert the null pattern.
                let mut pattern_failures = Vec::new();
                if envelope.properties_decoded {
                    pattern_failures.push("properties_decoded=true");
                }
                if envelope.properties_summary.is_some() {
                    pattern_failures.push("properties_summary present");
                }
                if envelope.animations.is_some() {
                    pattern_failures.push("animations present");
                }
                if pattern_failures.is_empty() {
                    nonactive_null_ok += 1;
                } else {
                    // This is the GOOD trigger — when parser ships decoders for
                    // Backdrop/Text/Extension, this fires. Re-flag as "type new structs."
                    failures.push(format!(
                        "non-(Active|Counter) type={} ({}) handle={} name={:?} broke null-pattern: {} \
                         — TIME TO ADD TYPED STRUCTS FOR THIS TYPE",
                        envelope.object_type,
                        envelope.object_type_name,
                        envelope.handle,
                        envelope.name,
                        pattern_failures.join(", "),
                    ));
                }
            }
        }
    }

    // ---- Layer-1 typed-pass assertions ----
    if envelope_ok != objects.count {
        failures.push(format!(
            "envelope-typed pass parsed {} of {} objects",
            envelope_ok, objects.count,
        ));
    }
    if active_typed_ok != objects.active_count {
        failures.push(format!(
            "Active typed-pass parsed {} of {} actives",
            active_typed_ok, objects.active_count,
        ));
    }
    if objects.counter_count != 44 {
        failures.push(format!("counter_count={} expected 44", objects.counter_count));
    }
    if objects.counter_decoded_count != objects.counter_count {
        failures.push(format!(
            "counter_decoded_count={} != counter_count={}",
            objects.counter_decoded_count, objects.counter_count,
        ));
    }
    if counter_typed_ok != objects.counter_count {
        failures.push(format!(
            "Counter typed-pass parsed {} of {} counters",
            counter_typed_ok, objects.counter_count,
        ));
    }
    let nonactive_total = objects.count - objects.active_count - objects.counter_count;
    if nonactive_null_ok != nonactive_total {
        failures.push(format!(
            "non-(Active|Counter) null-pattern matched {} of {} remaining",
            nonactive_null_ok, nonactive_total,
        ));
    }

    // Cross-check: header rollup counts == walked rollup counts.
    if active_total_directions != objects.active_animation_directions {
        failures.push(format!(
            "walked Active directions={} != header active_animation_directions={}",
            active_total_directions, objects.active_animation_directions,
        ));
    }
    if active_total_frames != objects.active_animation_frames {
        failures.push(format!(
            "walked Active frames={} != header active_animation_frames={}",
            active_total_frames, objects.active_animation_frames,
        ));
    }
    let header_unique: BTreeSet<u32> =
        objects.active_unique_image_handles.iter().copied().collect();
    if walked_active_unique_handles != header_unique {
        let only_walked: Vec<u32> = walked_active_unique_handles
            .difference(&header_unique)
            .copied()
            .collect();
        let only_header: Vec<u32> = header_unique
            .difference(&walked_active_unique_handles)
            .copied()
            .collect();
        failures.push(format!(
            "walked Active unique image handles ({}) != header active_unique_image_handles ({}). \
             only-walked={:?} only-header={:?}",
            walked_active_unique_handles.len(),
            header_unique.len(),
            only_walked,
            only_header,
        ));
    }

    // Counter rollup cross-checks.
    if walked_counter_total_handle_refs != objects.counter_total_handle_refs {
        failures.push(format!(
            "walked Counter total handle refs={} != header counter_total_handle_refs={}",
            walked_counter_total_handle_refs, objects.counter_total_handle_refs,
        ));
    }
    let header_counter_unique: BTreeSet<u32> =
        objects.counter_unique_image_handles.iter().copied().collect();
    if walked_counter_unique_handles != header_counter_unique {
        let only_walked: Vec<u32> = walked_counter_unique_handles
            .difference(&header_counter_unique)
            .copied()
            .collect();
        let only_header: Vec<u32> = header_counter_unique
            .difference(&walked_counter_unique_handles)
            .copied()
            .collect();
        failures.push(format!(
            "walked Counter unique image handles ({}) != header counter_unique_image_handles ({}). \
             only-walked={:?} only-header={:?}",
            walked_counter_unique_handles.len(),
            header_counter_unique.len(),
            only_walked,
            only_header,
        ));
    }

    // Surface the deferred-chunk antibody signal.
    if !objects.deferred_sub_chunk_ids_seen.is_empty() {
        println!(
            "[deferred] sub_chunk_ids_seen = {:?} (signal: parser left these undecoded — \
             will drop to [] when Backdrop/Text/Extension body decoders ship)",
            objects.deferred_sub_chunk_ids_seen,
        );
    }

    println!();
    println!("[typed pass] envelope OK              : {envelope_ok} / {}", objects.count);
    println!(
        "[typed pass] Active inner OK          : {active_typed_ok} / {}",
        objects.active_count,
    );
    println!(
        "[typed pass] Counter inner OK         : {counter_typed_ok} / {}",
        objects.counter_count,
    );
    println!(
        "[typed pass] non-(Active|Counter) OK  : {nonactive_null_ok} / {nonactive_total}",
    );
    println!(
        "[Active rollup]  max_anim_items={} total_directions={} total_frames={} \
         unique_handles_summed={}",
        active_max_anim_items,
        active_total_directions,
        active_total_frames,
        active_total_unique_handles,
    );
    println!(
        "[Counter rollup] total_handle_refs={} unique_handles={}",
        walked_counter_total_handle_refs,
        walked_counter_unique_handles.len(),
    );

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
