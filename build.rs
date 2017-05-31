const REPLACE: &'static str = "HSIZE";

const KEY_SIZES: &'static str = "HASH_SIZES";

const GENERATED_DIR: &'static str = "src/generated/";

const TEMPLATES_DIR: &'static str = "src/templates/";

use std::env;
use std::ffi::OsStr;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

fn main() {
    let sizes = get_precomp_sizes();

    if sizes.is_empty() { return; }

    let _ = fs::create_dir_all(GENERATED_DIR)
            .and_then(|_| get_templates())
            .and_then(|templates| expand_templates(templates, sizes));
}

fn expand_templates(templates: Vec<Template>, sizes: Vec<String>) -> io::Result<()> {
    for template in templates {
        let outfile = File::create(&template.path)?;
        expand(template, &sizes, outfile);
    }

    Ok(())
}

fn expand(template: Template, sizes: &[String], mut outfile: File) -> io::Result<()> {
    if template.is_header {
        for line in template.val.lines() {
            if line.contains(REPLACE) {
                for size in sizes {
                    let sized_template = line.replace(REPLACE, size);
                    write!(outfile, "{}\n", sized_template)?;
                }
            } else {
                write!(outfile, "{}\n", line)?;
            }
        }
    } else {
        for size in sizes {
            let sized_template = template.val.replace(REPLACE, size);
            write!(outfile, "{}\n", sized_template)?;
        }
    }

    Ok(())
}

struct Template {
    is_header: bool,
    path: PathBuf,
    val: String,
}

fn get_templates() -> io::Result<Vec<Template>> {
    let mut templates = Vec::new();

    for template in fs::read_dir(TEMPLATES_DIR)? {
        let template = template?;

        let template_path = template.path();

        if template_path.extension().map_or(false, |ext| ext != ("rs".as_ref() as &OsStr)) { continue; }

        let is_header = template_path.file_stem().expect("File had extension but no name?")
            .to_str().expect("Cannot handle non-Unicode filenames").ends_with("header");

        let value = read_to_string(&template_path)?;

        let file_name = template_path.file_name().expect("unreachable");

        let path = Path::new(GENERATED_DIR).join(file_name);

        templates.push(Template { is_header: is_header, path: path, val: value });
    }

    Ok(templates)
}

fn read_to_string(path: &Path) -> io::Result<String> {
    let mut s = String::new();

    File::open(path)?.read_to_string(&mut s)?;

    Ok(s)
}

fn get_precomp_sizes() -> Vec<String> {
    env::var(KEY_SIZES).unwrap_or(String::new())
        .split(',').map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
        .collect()
}
