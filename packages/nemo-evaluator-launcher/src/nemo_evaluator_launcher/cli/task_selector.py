# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Custom task selector with preview panel using prompt_toolkit."""

from typing import Any, Optional

from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.formatted_text import FormattedText, to_formatted_text
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import (
    ConditionalContainer,
    Dimension,
    FormattedTextControl,
    HSplit,
    Layout,
    VSplit,
    Window,
    WindowAlign,
)
from prompt_toolkit.styles import Style as PTStyle
from prompt_toolkit.widgets import Box, Frame, TextArea
from questionary import Style as QStyle


def run_task_selector(
    task_names: list[str],
    task_metadata: dict[str, dict[str, Any]],
    style: Optional[QStyle] = None,
) -> Optional[list[str]]:
    """Run an interactive task selector with preview panel.

    Args:
        task_names: List of task names to select from
        task_metadata: Dict mapping task name to metadata (description, harness, etc.)
        style: Optional questionary style (not fully used yet)

    Returns:
        List of selected task names, or None if cancelled
    """
    # State
    selected: set[str] = set()
    highlighted_index = 0
    filter_text = ""
    cancelled = False
    confirmed = False

    def get_filtered_tasks() -> list[str]:
        """Get tasks matching the current filter."""
        if not filter_text:
            return task_names
        lower_filter = filter_text.lower()
        return [t for t in task_names if lower_filter in t.lower()]

    def get_task_list_text() -> FormattedText:
        """Generate the task list with selection indicators."""
        filtered = get_filtered_tasks()
        result: list[tuple[str, str]] = []

        for i, task in enumerate(filtered):
            # Highlight indicator
            if i == highlighted_index:
                result.append(("class:pointer", " ❯ "))
            else:
                result.append(("", "   "))

            # Selection indicator
            if task in selected:
                result.append(("class:selected", "● "))
            else:
                result.append(("", "○ "))

            # Task name
            if i == highlighted_index:
                result.append(("class:highlighted", task))
            else:
                result.append(("", task))

            result.append(("", "\n"))

        if not filtered:
            result.append(("class:instruction", "   No tasks match filter"))

        return FormattedText(result)

    def get_preview_text() -> FormattedText:
        """Generate the preview panel content for highlighted task."""
        filtered = get_filtered_tasks()
        if not filtered or highlighted_index >= len(filtered):
            return FormattedText([("", "Select a task to see details")])

        task = filtered[highlighted_index]
        meta = task_metadata.get(task, {})

        result: list[tuple[str, str]] = []

        # Task name header
        result.append(("class:bold", f"{task}\n"))
        result.append(("class:instruction", "─" * min(40, len(task) + 5) + "\n\n"))

        # Description (word-wrapped)
        desc = meta.get("description", "No description available")
        # Simple word wrap at ~38 chars
        words = desc.split()
        line = ""
        for word in words:
            if len(line) + len(word) + 1 > 38:
                result.append(("", line + "\n"))
                line = word
            else:
                line = f"{line} {word}".strip()
        if line:
            result.append(("", line + "\n"))

        result.append(("", "\n"))

        # Harness
        harness = meta.get("harness", "unknown")
        result.append(("class:label", "Harness: "))
        result.append(("class:value", f"{harness}\n"))

        # Defaults - show all available params
        params = meta.get("params", {})
        has_defaults = any(v is not None for v in params.values())

        if has_defaults:
            result.append(("class:label", "\nDefaults:\n"))

            # Display in consistent order with nice names
            param_display = [
                ("temperature", "temperature"),
                ("top_p", "top_p"),
                ("max_new_tokens", "max_new_tokens"),
                ("parallelism", "parallelism"),
                ("num_fewshot", "num_fewshot"),
            ]

            for key, display_name in param_display:
                value = params.get(key)
                if value is not None:
                    result.append(("", f"  {display_name}: "))
                    result.append(("class:value", f"{value}\n"))
        else:
            result.append(("class:instruction", "\n(uses harness defaults)\n"))

        # Endpoint types
        endpoint_types = meta.get("endpoint_types", [])
        if endpoint_types:
            result.append(("class:label", "\nEndpoints: "))
            result.append(("class:value", ", ".join(endpoint_types) + "\n"))

        return FormattedText(result)

    def get_status_text() -> FormattedText:
        """Generate status bar text."""
        filtered = get_filtered_tasks()
        count = len(selected)
        total = len(filtered)

        parts: list[tuple[str, str]] = []
        parts.append(("class:instruction", f" {count} selected"))
        if filter_text:
            parts.append(("class:instruction", f" | Filter: '{filter_text}' ({total} matches)"))
        parts.append(("class:instruction", " | ↑↓:navigate  space:select  enter:confirm  esc:cancel"))

        return FormattedText(parts)

    # Key bindings
    kb = KeyBindings()

    @kb.add("up")
    def move_up(event: Any) -> None:
        nonlocal highlighted_index
        filtered = get_filtered_tasks()
        if filtered:
            highlighted_index = (highlighted_index - 1) % len(filtered)

    @kb.add("down")
    def move_down(event: Any) -> None:
        nonlocal highlighted_index
        filtered = get_filtered_tasks()
        if filtered:
            highlighted_index = (highlighted_index + 1) % len(filtered)

    @kb.add("space")
    def toggle_selection(event: Any) -> None:
        filtered = get_filtered_tasks()
        if filtered and highlighted_index < len(filtered):
            task = filtered[highlighted_index]
            if task in selected:
                selected.discard(task)
            else:
                selected.add(task)

    @kb.add("enter")
    def confirm_selection(event: Any) -> None:
        nonlocal confirmed
        if selected:  # Only confirm if at least one task selected
            confirmed = True
            event.app.exit()

    @kb.add("escape")
    @kb.add("c-c")
    def cancel(event: Any) -> None:
        nonlocal cancelled
        cancelled = True
        event.app.exit()

    @kb.add("backspace")
    def delete_char(event: Any) -> None:
        nonlocal filter_text, highlighted_index
        if filter_text:
            filter_text = filter_text[:-1]
            highlighted_index = 0

    @kb.add("<any>")
    def type_char(event: Any) -> None:
        nonlocal filter_text, highlighted_index
        char = event.data
        # Only accept printable characters for filtering
        if char.isprintable() and len(char) == 1:
            filter_text += char
            highlighted_index = 0

    # Styles
    pt_style = PTStyle.from_dict({
        "pointer": "#76b900 bold",  # NVIDIA green
        "selected": "#76b900",
        "highlighted": "bold",
        "instruction": "gray",
        "bold": "bold",
        "label": "bold",
        "value": "#76b900",
        "border": "gray",
    })

    # Layout
    task_list = Window(
        content=FormattedTextControl(get_task_list_text),
        width=Dimension(min=30, preferred=40),
        wrap_lines=False,
    )

    preview_panel = Window(
        content=FormattedTextControl(get_preview_text),
        width=Dimension(min=30, preferred=45),
        wrap_lines=True,
    )

    # Divider between panels
    divider = Window(
        content=FormattedTextControl(lambda: FormattedText([("class:border", "│")])),
        width=1,
    )

    main_content = VSplit([
        task_list,
        divider,
        preview_panel,
    ])

    # Header
    header = Window(
        content=FormattedTextControl(
            lambda: FormattedText([
                ("class:bold", "? "),
                ("bold", "Select tasks to run:"),
            ])
        ),
        height=1,
    )

    # Status bar
    status_bar = Window(
        content=FormattedTextControl(get_status_text),
        height=1,
    )

    layout = Layout(
        HSplit([
            header,
            Window(height=1),  # Spacer
            Frame(main_content, style="class:border"),
            status_bar,
        ])
    )

    # Create and run application
    app: Application[None] = Application(
        layout=layout,
        key_bindings=kb,
        style=pt_style,
        full_screen=False,
        mouse_support=False,
    )

    app.run()

    if cancelled:
        return None

    return list(selected) if selected else None
