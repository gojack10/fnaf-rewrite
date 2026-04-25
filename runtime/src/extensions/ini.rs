//! Ini extension (object_type=33, FNAF 1 handle 10).
//!
//! ## FNAF 1 usage
//!
//! Per the 2026-04-24 runtime-usage probe and authoritative cross-check
//! against `reference/nuclearrt/exporter/src/Exporters/Extensions/IniExporter.cs`:
//!
//! | FNAF call | nuclearrt name      | Args               | Calls |
//! |-----------|---------------------|--------------------|-------|
//! | ExtAction_86 | `SetFileName(name)` | string           | 6 (always 'freddy') |
//! | ExtAction_87 | `SetValue(item, v)` | string + int     | 22 |
//! | ExtExpression_82 | `GetValue(item)` | string          | 7 |
//!
//! FNAF never calls `SetCurrentGroup` (action 80) — every read/write goes
//! to the default empty section. The resulting `freddy.ini` on disk is a
//! flat key=value file with no `[section]` headers.
//!
//! Five integer keys ever touched: `level`, `beatgame`, `beat6`, `beat7`,
//! `lives`.
//!
//! ## Body is dead weight
//!
//! Per nuclearrt's `IniExporter.ExportExtension`, the 658-byte property
//! body serialises as `i16 flags + wide_string default_name`. FNAF
//! overrides the default name via the runtime SetFileName action so the
//! body's default-name field is dead, and the flags are unused. No body
//! decoder ships parser-side for this type.
//!
//! ## Save directory
//!
//! `dirs::data_dir()` resolves the platform's user data directory:
//! - Linux:   `$XDG_DATA_HOME` or `~/.local/share`
//! - macOS:   `~/Library/Application Support`
//! - Windows: `%APPDATA%`
//!
//! We append `fnaf-rewrite/` so saves don't collide with the original
//! game's `.ini` (which lived in `%APPDATA%/Roaming/MMFApplications/` on
//! Windows). If the data dir cannot be resolved (extremely unusual), we
//! fall back to the current working directory so the runtime stays
//! functional rather than panicking on construction.
//!
//! See [[Ini Extension Runtime|e661dfa9-947f-45d3-a305-7862bd8348ff]] on
//! the FNAF Speedrun ideation tree.

use std::path::{Path, PathBuf};

use ini::Ini;

const APP_DIRECTORY_NAME: &str = "fnaf-rewrite";
const DEFAULT_SECTION: Option<&str> = None;

/// Single Clickteam Ini extension instance.
///
/// One instance per game. FNAF only constructs one (handle 10).
pub struct IniExtension {
    /// Filename last set by `SetFileName`. Empty until `set_file_name` is
    /// called the first time. Resolved against the platform data directory
    /// to produce the on-disk path.
    file_name: String,
    /// Cached on-disk path. Recomputed every `set_file_name` call.
    file_path: Option<PathBuf>,
    /// In-memory mirror of the .ini file. Loaded on first `set_file_name`
    /// or on first read; flushed to disk on every `set_value`.
    ini: Ini,
}

impl IniExtension {
    /// Construct a fresh, file-less Ini instance. FNAF wires the filename
    /// via `set_file_name` immediately after construction at frame load.
    pub fn new() -> Self {
        Self {
            file_name: String::new(),
            file_path: None,
            ini: Ini::new(),
        }
    }

    /// `Ini.SetFileName(name)` — Clickteam action 86.
    ///
    /// Resolves `<data_dir>/fnaf-rewrite/<name>.ini`, creates the
    /// `fnaf-rewrite` subdirectory if missing, and loads any existing
    /// content from disk. Subsequent `get_value` calls see the loaded
    /// state; subsequent `set_value` calls overwrite it on disk.
    ///
    /// Returns the resolved path so callers / tests can inspect it.
    pub fn set_file_name(&mut self, name: &str) -> std::io::Result<&Path> {
        self.set_file_name_in(name, &Self::default_data_dir())
    }

    /// Test-friendly variant that allows pinning the data directory.
    /// Production code calls `set_file_name`; tests use a tempdir to avoid
    /// touching the user's real save directory.
    pub fn set_file_name_in(
        &mut self,
        name: &str,
        data_dir: &Path,
    ) -> std::io::Result<&Path> {
        self.file_name = name.to_string();

        let app_dir = data_dir.join(APP_DIRECTORY_NAME);
        std::fs::create_dir_all(&app_dir)?;

        // Match Clickteam Ini convention: append .ini if not present.
        let mut filename = name.to_string();
        if !Path::new(&filename)
            .extension()
            .map(|e| e.eq_ignore_ascii_case("ini"))
            .unwrap_or(false)
        {
            filename.push_str(".ini");
        }
        let path = app_dir.join(filename);

        // Load any existing state. Missing file is fine — start with empty.
        self.ini = match Ini::load_from_file(&path) {
            Ok(parsed) => parsed,
            Err(ini::Error::Io(ref e)) if e.kind() == std::io::ErrorKind::NotFound => {
                Ini::new()
            }
            Err(e) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("failed to load existing ini at {}: {e}", path.display()),
                ));
            }
        };

        self.file_path = Some(path);
        Ok(self.file_path.as_deref().unwrap())
    }

    /// `Ini.SetValue(item, value)` — Clickteam action 87.
    ///
    /// Writes `item=value` into the default empty section (FNAF never
    /// switches sections) and flushes to disk synchronously so a crash
    /// between writes can't lose progress.
    pub fn set_value(&mut self, item: &str, value: i64) -> std::io::Result<()> {
        self.ini
            .with_section(DEFAULT_SECTION)
            .set(item, value.to_string());
        self.flush()
    }

    /// `Ini.GetValue(item)` — Clickteam expression 82.
    ///
    /// Returns the integer stored at `item` in the default empty section,
    /// or 0 if the key is missing or non-numeric. Mirrors nuclearrt's
    /// `IniExtension::GetValue`: `value.empty() ? 0 : std::stoi(value)`.
    pub fn get_value(&self, item: &str) -> i64 {
        self.ini
            .section(DEFAULT_SECTION)
            .and_then(|s| s.get(item))
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(0)
    }

    /// On-disk path. None until `set_file_name` is called.
    pub fn file_path(&self) -> Option<&Path> {
        self.file_path.as_deref()
    }

    fn flush(&self) -> std::io::Result<()> {
        let path = self.file_path.as_ref().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::Other,
                "set_value called before set_file_name",
            )
        })?;
        self.ini
            .write_to_file(path)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    fn default_data_dir() -> PathBuf {
        dirs::data_dir().unwrap_or_else(|| PathBuf::from("."))
    }
}

impl Default for IniExtension {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn fresh(tmp: &TempDir) -> IniExtension {
        let mut ext = IniExtension::new();
        ext.set_file_name_in("freddy", tmp.path()).unwrap();
        ext
    }

    #[test]
    fn set_file_name_creates_subdir_and_resolves_path() {
        let tmp = TempDir::new().unwrap();
        let mut ext = IniExtension::new();
        let path = ext.set_file_name_in("freddy", tmp.path()).unwrap();
        assert_eq!(path, &tmp.path().join("fnaf-rewrite").join("freddy.ini"));
        assert!(path.parent().unwrap().exists(), "subdir should be created");
    }

    #[test]
    fn set_value_then_get_value_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let mut ext = fresh(&tmp);
        ext.set_value("level", 1).unwrap();
        assert_eq!(ext.get_value("level"), 1);
    }

    #[test]
    fn get_value_missing_key_returns_zero() {
        let tmp = TempDir::new().unwrap();
        let ext = fresh(&tmp);
        assert_eq!(
            ext.get_value("nonexistent"),
            0,
            "missing keys must return 0 per nuclearrt GetValue contract",
        );
    }

    #[test]
    fn missing_file_does_not_panic() {
        // First-run case: no .ini file on disk yet, GetValue must return 0
        // for everything without panicking. This is exactly FNAF's first-
        // launch experience.
        let tmp = TempDir::new().unwrap();
        let ext = fresh(&tmp);
        for key in ["level", "beatgame", "beat6", "beat7", "lives"] {
            assert_eq!(ext.get_value(key), 0, "key {key} should default to 0");
        }
    }

    #[test]
    fn persistence_across_instances() {
        let tmp = TempDir::new().unwrap();
        {
            let mut ext = fresh(&tmp);
            ext.set_value("level", 7).unwrap();
            ext.set_value("beatgame", 1).unwrap();
        }
        // Drop, recreate, verify state persisted.
        let ext = fresh(&tmp);
        assert_eq!(ext.get_value("level"), 7);
        assert_eq!(ext.get_value("beatgame"), 1);
    }

    #[test]
    fn all_fnaf1_keys_roundtrip_independently() {
        // FNAF 1 only ever writes these 5 keys. Each one must be
        // independently addressable in the default empty section.
        let tmp = TempDir::new().unwrap();
        let mut ext = fresh(&tmp);
        let keys_and_values = [
            ("level", 1_i64),
            ("beatgame", 1),
            ("beat6", 0),
            ("beat7", 0),
            ("lives", 5),
        ];
        for (k, v) in keys_and_values {
            ext.set_value(k, v).unwrap();
        }
        for (k, expected) in keys_and_values {
            assert_eq!(ext.get_value(k), expected, "key {k}");
        }
    }

    #[test]
    fn set_value_without_filename_errors() {
        let mut ext = IniExtension::new();
        let err = ext.set_value("level", 1).unwrap_err();
        // Must be a clear error, not a panic. If FNAF ever writes before
        // SetFileName, that's a runtime contract violation worth surfacing.
        assert!(err.to_string().contains("set_file_name"));
    }

    #[test]
    fn explicit_ini_extension_in_filename_is_not_doubled() {
        // If a user passes "freddy.ini" explicitly, we should NOT produce
        // "freddy.ini.ini". (FNAF passes "freddy" so this guards future
        // misuse.)
        let tmp = TempDir::new().unwrap();
        let mut ext = IniExtension::new();
        let path = ext.set_file_name_in("freddy.ini", tmp.path()).unwrap();
        assert_eq!(
            path,
            &tmp.path().join("fnaf-rewrite").join("freddy.ini"),
        );
    }

    #[test]
    fn flat_file_format_no_section_header() {
        // FNAF reads its own ini back, but a human inspecting freddy.ini
        // should see flat key=value with no [section] header. Verify by
        // writing a value and inspecting the on-disk file directly.
        let tmp = TempDir::new().unwrap();
        let mut ext = fresh(&tmp);
        ext.set_value("level", 1).unwrap();
        let on_disk = std::fs::read_to_string(ext.file_path().unwrap()).unwrap();
        assert!(
            !on_disk.contains('['),
            "expected flat file, got section header in:\n{on_disk}",
        );
        assert!(
            on_disk.contains("level=1"),
            "expected level=1 line, got:\n{on_disk}",
        );
    }
}
