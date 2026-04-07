# Page Conventions

Formatting rules for all wiki pages. Follow these precisely to ensure consistency and git-friendliness.

## General Formatting

- **Line endings:** UNIX (`\n`), no trailing spaces
- **Encoding:** UTF-8
- **Max line length:** Soft limit of 120 characters for prose; no limit for code blocks or tables
- **Blank lines:** One blank line between sections; two blank lines before a `##` heading (except the first)
- **Lists:** Use `-` for bullet lists (not `*` or `+`)
- **Bold:** `**text**` for emphasis
- **Code:** Backtick inline code for file paths, source IDs, commands, and field names

## Frontmatter

Every wiki page (except `index.md`, `log.md`, `overview.md`) MUST start with a YAML frontmatter block:

```
---
field: value
---
```

- No blank line between `---` and fields
- String values with special characters must be quoted
- Lists use YAML flow style: `[item1, item2]`
- Dates use `YYYY-MM-DD` format
- Timestamps use ISO 8601 UTC: `YYYY-MM-DDTHH:MM:SSZ`

## Headings

- The page title (`# H1`) appears **after** the frontmatter and is the first heading
- Use `## H2` for major sections (Summary, Description, Sources, etc.)
- Use `### H3` for subsections within a section
- Do not skip heading levels
- Heading text uses title case for proper nouns, sentence case otherwise

## Links

### Internal Links (relative)
All internal links use relative paths from the current file's location:

```markdown
<!-- From wiki/sources/foo.md -->
[Entity Name](../entities/entity-slug.md)
[Concept Name](../concepts/concept-slug.md)

<!-- From wiki/entities/bar.md -->
[Another Entity](./other-entity.md)
[Source Title](../sources/source-id.md)
```

### External Links
```markdown
[Link text](https://example.com)
```

Never use absolute vault paths in links. Use relative paths only.

## Citations in Sources Sections

The `## Sources` section on entity, concept, and topic pages uses this format:

```markdown
## Sources

- [Source Title](../sources/source-id.md) — Brief note on what this source says about the subject
- [Another Source](../sources/other-id.md) — "Direct quote if relevant"
```

## The Bold Description Paragraph

Every page body (after the `# Title` heading) MUST start with a bold one-sentence description:

```markdown
# Transformer Architecture

**The Transformer is a neural network architecture that relies entirely on self-attention mechanisms, eliminating the need for recurrence or convolution.**
```

This sentence should be self-contained — a reader who sees only this sentence should understand what the page is about.

## Sections That Must Exist

Do not leave required sections empty. If a section truly has no content, omit it entirely rather than leaving it blank or writing "None."

### Source pages
1. `## Summary` — required
2. `## Key Points` — required (at least 2 bullet points)
3. `## Entities` — omit if none identified
4. `## Concepts` — omit if none identified
5. `## Topics` — omit if none identified
6. `## Source` — required (links to raw/normalized files)

### Entity pages
1. `## Description` — required
2. `## Sources` — required (at least one citation)
3. `## Related Entities` — omit if none
4. `## Related Concepts` — omit if none

### Concept pages
1. `## Description` — required
2. `## Sources` — required
3. `## Related Concepts` — omit if none

### Topic pages
1. `## Overview` — required
2. `## Sources` — required
3. `## Related Topics` — omit if none

## Do Not

- Add a section called `## References` — use `## Sources` instead
- Use `[[wiki-link]]` syntax — use standard markdown links
- Use HTML tags (e.g., `<br>`, `<hr>`) — use markdown equivalents
- Hard-code absolute paths to the vault root
- Add sections like "Last Updated" — this is tracked by git
- Create empty pages — a page must have at least a description and one section
