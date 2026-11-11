"""Compact Markdown tables with single-line cells and horizontal overflow."""


def table(lines, headers, rows, *, right=()):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines.extend(
        [
            '<div style="max-width:100%; overflow-x:auto; white-space:nowrap;">',
            "",
            "| " + " | ".join(cell(value) for value in headers) + " |",
            "|"
            + "|".join("---:" if i in right else "---" for i in range(len(headers)))
            + "|",
        ]
    )
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    lines.extend(["", "</div>", ""])
