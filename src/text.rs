//! Byte-bounded, UTF-8-safe text sampling shared by every module.

/// Keep the head and tail of `text` within `max_bytes`, eliding the middle.
/// Never splits a UTF-8 character.
pub(crate) fn elide_middle(text: &str, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return text.to_string();
    }
    const MARKER: &str = "\n\n… [bytes elided] …\n\n";
    let retained_budget = max_bytes.saturating_sub(MARKER.len());
    if retained_budget == 0 {
        return text[..floor_char_boundary(text, max_bytes)].to_string();
    }
    let head_budget = retained_budget / 2;
    let tail_budget = retained_budget.saturating_sub(head_budget);
    let head_end = floor_char_boundary(text, head_budget);
    let tail_start = ceil_char_boundary(text, text.len().saturating_sub(tail_budget));
    format!("{}{MARKER}{}", &text[..head_end], &text[tail_start..])
}

pub(crate) fn floor_char_boundary(text: &str, mut index: usize) -> usize {
    index = index.min(text.len());
    while index > 0 && !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

pub(crate) fn ceil_char_boundary(text: &str, mut index: usize) -> usize {
    index = index.min(text.len());
    while index < text.len() && !text.is_char_boundary(index) {
        index += 1;
    }
    index
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elide_middle_is_bounded_and_utf8_safe() {
        let text = "编译失败🙂".repeat(2_000);
        let sample = elide_middle(&text, 4 * 1024);
        assert!(sample.contains("bytes elided"));
        assert!(sample.starts_with('编'));
        assert!(sample.ends_with('🙂'));
        assert!(sample.len() <= 4 * 1024);
    }

    #[test]
    fn short_text_is_unchanged() {
        assert_eq!(elide_middle("ok", 64), "ok");
    }
}
