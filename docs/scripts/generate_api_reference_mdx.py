#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Standalone Griffe-based generator for curated Core Python API reference data.

"""Optional Griffe helper to draft Core Python API reference MDX.

The published API reference under ``docs/references/api/nemo-evaluator/`` is
hand-maintained static MDX. Run this script with ``--draft-dir`` when you want
to compare Griffe output against those pages after API changes.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
import textwrap
from typing import TYPE_CHECKING

_SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import griffe  # noqa: E402

if TYPE_CHECKING:
    from griffe import Attribute, Class, Function, Module, Object

API_DATACLASSES_GROUPS: dict[str, list[str]] = {
    "Modeling Target": ["ApiEndpoint", "EndpointType", "EvaluationTarget"],
    "Modeling Evaluation": ["EvaluationConfig", "ConfigParams"],
    "Modeling Result": [
        "EvaluationResult",
        "GroupResult",
        "MetricResult",
        "Score",
        "ScoreStats",
        "TaskResult",
    ],
}

ADAPTERS_AUTOSUMMARY: dict[str, list[tuple[str, str, str]]] = {
    "Configuration": [
        ("DiscoveryConfig", "adapter-config", "nemo_evaluator.adapters.adapter_config"),
        (
            "InterceptorConfig",
            "adapter-config",
            "nemo_evaluator.adapters.adapter_config",
        ),
        (
            "PostEvalHookConfig",
            "adapter-config",
            "nemo_evaluator.adapters.adapter_config",
        ),
        ("AdapterConfig", "adapter-config", "nemo_evaluator.adapters.adapter_config"),
    ],
    "Interceptors": [
        ("CachingInterceptor", "interceptors", "nemo_evaluator.adapters.interceptors"),
        ("EndpointInterceptor", "interceptors", "nemo_evaluator.adapters.interceptors"),
        (
            "PayloadParamsModifierInterceptor",
            "interceptors",
            "nemo_evaluator.adapters.interceptors",
        ),
        (
            "ProgressTrackingInterceptor",
            "interceptors",
            "nemo_evaluator.adapters.interceptors",
        ),
        (
            "RaiseClientErrorInterceptor",
            "interceptors",
            "nemo_evaluator.adapters.interceptors",
        ),
        (
            "RequestLoggingInterceptor",
            "interceptors",
            "nemo_evaluator.adapters.interceptors",
        ),
        (
            "ResponseLoggingInterceptor",
            "interceptors",
            "nemo_evaluator.adapters.interceptors",
        ),
        (
            "ResponseReasoningInterceptor",
            "interceptors",
            "nemo_evaluator.adapters.interceptors",
        ),
        (
            "ResponseStatsInterceptor",
            "interceptors",
            "nemo_evaluator.adapters.interceptors",
        ),
        (
            "SystemMessageInterceptor",
            "interceptors",
            "nemo_evaluator.adapters.interceptors",
        ),
    ],
    "PostEvalHooks": [
        ("PostEvalReportHook", "adapter-config", "nemo_evaluator.adapters.reports"),
    ],
    "Interfaces": [
        ("RequestInterceptor", "types", "nemo_evaluator.adapters.types"),
        ("ResponseInterceptor", "types", "nemo_evaluator.adapters.types"),
        ("RequestToResponseInterceptor", "types", "nemo_evaluator.adapters.types"),
        ("PostEvalHook", "types", "nemo_evaluator.adapters.types"),
    ],
}

SKIP_FIELD_NAMES = frozenset({"model_config"})

INTERNAL_METHOD_PREFIXES = ("handle_", "sanitize_")

PUBLIC_METHOD_NAMES = frozenset(
    {
        "start",
        "stop",
        "exec",
        "upload",
        "download",
        "resolve_outside_endpoint",
        "from_dict",
        "from_legacy_config",
        "get_interceptor_configs",
        "ensure_image_built",
        "get_ecr_image_tag",
        "image_exists_in_ecr",
        "intercept_request",
        "intercept_response",
        "health",
        "reconnect_tunnel",
        "describe_task",
    }
)

PROTOCOL_DUNDER_CLASSES = frozenset({"Sandbox", "EcsFargateSandbox"})

ALLOWED_DUNDER_METHODS = frozenset({"__enter__", "__exit__"})

FIELDS_ONLY_CLASSES = frozenset(
    {
        "ExecResult",
        "OutsideEndpoint",
        "EcsFargateConfig",
        "SshSidecarConfig",
    }
)

METHODS_ONLY_CLASSES = frozenset({"SshTunnel", "ExecClient"})

_DOCSECTION_HEADERS = {
    "Args:": "Args",
    "Arguments:": "Arguments",
    "Parameters:": "Parameters",
    "Returns:": "Returns",
    "Return:": "Returns",
    "Raises:": "Raises",
    "Yields:": "Yields",
    "Note:": "Note",
    "Notes:": "Notes",
}

_ARG_ITEM = re.compile(r"^[A-Za-z_][\w]*(\s*\(.+\))?\s*:")

_MODULE_CACHE: dict[str, Module] = {}
_DOC_CACHE: dict[str, str] = {}


def _clear_caches() -> None:
    _MODULE_CACHE.clear()
    _DOC_CACHE.clear()


def _load_module(module_name: str) -> Module:
    if module_name not in _MODULE_CACHE:
        _MODULE_CACHE[module_name] = griffe.load(module_name)
    return _MODULE_CACHE[module_name]


def _object_cache_key(obj: Object) -> str:
    obj = _resolve(obj)
    path = getattr(obj, "path", None)
    if path is not None:
        return str(path)
    return str(id(obj))


def _resolve(obj: Object) -> Object:
    if getattr(obj, "is_alias", False):
        return obj.final_target
    return obj


def _clean_docstring(text: str) -> str:
    if not text:
        return ""

    def _inline_admonition(match: re.Match[str]) -> str:
        kind = match.group(1).strip().capitalize()
        body = match.group(2).strip()
        return f"> **{kind}:** {body}"

    def _block_admonition(match: re.Match[str]) -> str:
        kind = match.group(1).strip().capitalize()
        body = textwrap.dedent(match.group(2)).strip()
        return f"> **{kind}:** {body}"

    text = re.sub(r"\.\.\s+(\w+)::\s+([^\n]+)", _inline_admonition, text)
    text = re.sub(
        r"^\.\.\s+(\w+)::\s*\n((?:[ \t].*\n?)*)",
        _block_admonition,
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(r":(?:class|func|meth|attr|mod|data|obj):`([^`]+)`", r"`\1`", text)
    text = re.sub(
        r":(?:class|func|meth|attr|mod|data|obj):`?([^`\s]+)`?", r"`\1`", text
    )
    text = _convert_rst_literal_blocks(text)
    text = _normalize_rst_inline(text)
    text = _format_docstring_sections(text)
    return text.strip()


def _normalize_rst_inline(text: str) -> str:
    fences: list[str] = []

    def _stash_fence(match: re.Match[str]) -> str:
        fences.append(match.group(0))
        return f"\0FENCE{len(fences) - 1}\0"

    text = re.sub(r"```[\s\S]*?```", _stash_fence, text)
    text = re.sub(r"``([^`]+)``", r"`\1`", text)
    text = re.sub(
        r"(?<!\*\*)\bImportant:\s+",
        "> **Important:** ",
        text,
    )
    for header in _DOCSECTION_HEADERS:
        text = re.sub(rf"([^\n])\n{re.escape(header)}", rf"\1\n\n{header}", text)
        text = re.sub(rf"([^\n]){re.escape(header)}", rf"\1\n\n{header}", text)
    for index, fence in enumerate(fences):
        text = text.replace(f"\0FENCE{index}\0", fence)
    return text


def _convert_rst_literal_blocks(text: str) -> str:
    lines = text.split("\n")
    result: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.rstrip().endswith("::"):
            result.append(line.rstrip().removesuffix("::") + ":")
            i += 1
            while i < len(lines) and lines[i].strip() == "":
                i += 1
            block: list[str] = []
            while i < len(lines) and (
                lines[i].startswith(" ") or lines[i].startswith("\t")
            ):
                block.append(lines[i].strip())
                i += 1
            if block:
                result.append("")
                result.append("```")
                result.extend(block)
                result.append("```")
                continue
        result.append(line)
        i += 1
    return "\n".join(result)


def _format_docstring_sections(text: str) -> str:
    lines = text.split("\n")
    result: list[str] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped in _DOCSECTION_HEADERS:
            title = _DOCSECTION_HEADERS[stripped]
            result.extend([f"**{title}**", ""])
            i += 1
            items: list[str] = []
            while i < len(lines):
                inner = lines[i]
                inner_stripped = inner.strip()
                if inner_stripped in _DOCSECTION_HEADERS:
                    break
                if inner_stripped == "":
                    j = i + 1
                    while j < len(lines) and lines[j].strip() == "":
                        j += 1
                    if j < len(lines) and lines[j].strip() in _DOCSECTION_HEADERS:
                        break
                    i += 1
                    continue
                if inner.startswith((" ", "\t")):
                    content = inner_stripped
                    if (
                        items
                        and not _ARG_ITEM.match(content)
                        and not re.match(r"^\d+\.", content)
                    ):
                        items[-1] = f"{items[-1]} {content}"
                    else:
                        items.append(content)
                elif re.match(r"^\d+\.", inner_stripped):
                    items.append(inner_stripped)
                elif inner_stripped in _DOCSECTION_HEADERS:
                    break
                else:
                    items.append(inner_stripped)
                i += 1
            for item in items:
                if re.match(r"^\d+\.", item):
                    result.append(item)
                elif re.search(r"\s\d+\.\s", item):
                    parts = re.split(r"\s+(?=\d+\.\s)", item.strip())
                    for part in parts:
                        if re.match(r"^\d+\.", part):
                            result.append(part)
                        else:
                            result.append(f"- {part}")
                else:
                    result.append(f"- {item}")
            result.append("")
            continue
        result.append(lines[i])
        i += 1
    return "\n".join(result)


def _doc(obj: Object) -> str:
    obj = _resolve(obj)
    cache_key = _object_cache_key(obj)
    if cache_key in _DOC_CACHE:
        return _DOC_CACHE[cache_key]
    if obj.docstring is None:
        cleaned = ""
    else:
        cleaned = _clean_docstring(obj.docstring.value.strip())
    _DOC_CACHE[cache_key] = cleaned
    return cleaned


def _escape_table_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _is_defined_in_module(obj: Object, module_name: str) -> bool:
    obj = _resolve(obj)
    if getattr(obj, "is_alias", False):
        return False
    path = getattr(obj, "path", None)
    return bool(path and str(path).startswith(f"{module_name}."))


def _is_enum_class(cls: Class) -> bool:
    cls = _resolve(cls)
    return any("Enum" in str(base) for base in cls.bases)


def _should_include_field(
    name: str, cls: Class, *, typ: str = "", desc: str = ""
) -> bool:
    if name in SKIP_FIELD_NAMES:
        return False
    if name.startswith("_"):
        return False
    if _is_enum_class(cls):
        return False
    if not typ.strip() and not desc.strip():
        return False
    return True


def _should_include_method(fn: Function, class_name: str) -> bool:
    fn = _resolve(fn)
    name = fn.name

    if class_name in FIELDS_ONLY_CLASSES or class_name in METHODS_ONLY_CLASSES:
        return False

    if any(name.startswith(prefix) for prefix in INTERNAL_METHOD_PREFIXES):
        return False

    if name.startswith("__") and name.endswith("__"):
        if name not in ALLOWED_DUNDER_METHODS:
            return False
        return class_name in PROTOCOL_DUNDER_CLASSES

    if name.startswith("_"):
        return False

    if _doc(fn):
        return True

    return name in PUBLIC_METHOD_NAMES


def _render_enum_values(cls: Class) -> str:
    values: list[str] = []
    for name, member in sorted(cls.all_members.items()):
        if name.startswith("_") or not member.is_attribute:
            continue
        line = f"- `{name}`"
        member_doc = _doc(member)
        if member_doc:
            line += f" — {member_doc}"
        values.append(line)
    if not values:
        return ""
    return "**Values**\n\n" + "\n".join(values) + "\n"


def _annotation_text(annotation: object | None) -> str:
    if annotation is None:
        return ""
    return (
        str(annotation).replace("ExprName(name='", "").split("'")[0]
        if "ExprName" in str(annotation)
        else str(annotation)
    )


def _annotation(attr: Attribute) -> str:
    return _annotation_text(getattr(attr, "annotation", None))


def _function_signature(fn: Function) -> str:
    fn = _resolve(fn)
    if not hasattr(fn, "parameters"):
        return f"{fn.name}(...)"
    parts: list[str] = []
    for param in fn.parameters:
        ann = _annotation_text(getattr(param, "annotation", None))
        if ann:
            parts.append(f"{param.name}: {ann}")
        else:
            parts.append(param.name)
    return f"{fn.name}({', '.join(parts)})"


def _method_heading(method_name: str) -> str:
    if method_name == "__init__":
        return "Constructor"
    return method_name


def _member_anchor(parent: str, member_name: str) -> str:
    return _slugify(f"{parent}-{member_name}")


def _should_skip_init(fn: Function, fields: list[tuple[str, str, str]]) -> bool:
    fn = _resolve(fn)
    if not fields:
        return False
    if len(_function_signature(fn)) > 120:
        return True
    param_names = {p.name for p in fn.parameters if p.name not in {"self", "cls"}}
    field_names = {name for name, _, _ in fields}
    return bool(param_names and param_names <= field_names)


def _render_method(fn: Function, class_name: str) -> str:
    fn = _resolve(fn)
    anchor = _member_anchor(class_name, fn.name)
    title = _method_heading(fn.name)
    parts = [
        f'<span id="{anchor}"></span>',
        "",
        f"#### {title}",
        "",
        _md_code(_function_signature(fn)),
        "",
    ]
    fn_doc = _doc(fn)
    if fn_doc:
        parts.extend([fn_doc, ""])
    return "\n".join(parts)


def _slugify(name: str) -> str:
    return re.sub(r"[^\w\-]", "-", name).lower()


def _md_code(text: str, lang: str = "python") -> str:
    return f"```{lang}\n{text.rstrip()}\n```"


def _render_summary_table(
    rows: list[tuple[str, str]], *, link_names: bool = False
) -> str:
    if not rows:
        return ""
    out = ["| Name | Description |", "| --- | --- |"]
    for name, desc in rows:
        desc = _escape_table_cell(desc)
        name_cell = name if link_names else f"`{name}`"
        out.append(f"| {name_cell} | {desc} |")
    return "\n".join(out)


def _render_class(obj: Class) -> str:
    obj = _resolve(obj)
    anchor = _slugify(obj.name)
    parts = [f'<span id="{anchor}"></span>', "", f"### `{obj.name}`", ""]
    doc = _doc(obj)
    if doc:
        parts.extend([doc, ""])

    if _is_enum_class(obj):
        enum_values = _render_enum_values(obj)
        if enum_values:
            parts.extend([enum_values, ""])
        return "\n".join(parts)

    fields: list[tuple[str, str, str]] = []
    methods: list[Function] = []
    for name, member in sorted(obj.all_members.items()):
        if member.is_class:
            continue
        if member.is_function:
            if _should_include_method(member, obj.name):
                methods.append(member)
        elif member.is_attribute:
            typ = _annotation(member)
            desc = _doc(member)
            if _should_include_field(name, obj, typ=typ, desc=desc):
                fields.append((name, typ, desc))

    if fields:
        parts.extend(["| Field | Type | Description |", "| --- | --- | --- |"])
        for name, typ, desc in fields:
            parts.append(
                f"| `{name}` | `{_escape_table_cell(typ)}` | {_escape_table_cell(desc)} |"
            )
        parts.append("")

    for fn in methods:
        fn = _resolve(fn)
        if fn.name == "__init__" and _should_skip_init(fn, fields):
            continue
        parts.extend([_render_method(fn, obj.name), ""])

    return "\n".join(parts)


def _render_function(obj: Function) -> str:
    obj = _resolve(obj)
    anchor = _slugify(obj.name)
    parts = [
        f'<span id="{anchor}"></span>',
        "",
        f"### `{obj.name}`",
        "",
        _md_code(_function_signature(obj)),
        "",
    ]
    doc = _doc(obj)
    if doc:
        parts.append(doc)
    return "\n".join(parts)


def _render_module(
    module_name: str, *, extra_class_paths: list[str] | None = None
) -> str:
    module = _load_module(module_name)
    parts: list[str] = []

    for name in sorted(module.classes.keys()):
        cls = module.classes[name]
        if name.startswith("_") or not _is_defined_in_module(cls, module_name):
            continue
        parts.extend([_render_class(cls), ""])

    for name in sorted(module.functions.keys()):
        fn = module.functions[name]
        if name.startswith("_"):
            continue
        parts.extend([_render_function(fn), ""])

    for name in sorted(module.attributes.keys()):
        attr = module.attributes[name]
        if name.startswith("_"):
            continue
        parts.extend([f"### `{name}`", ""])
        doc = _doc(attr)
        if doc:
            parts.append(doc)
        if attr.annotation is not None:
            parts.append(f"\nType: `{_annotation(attr)}`")
        parts.append("")

    if extra_class_paths:
        for path in extra_class_paths:
            extra_mod = _load_module(".".join(path.split(".")[:-1]))
            class_name = path.split(".")[-1]
            if class_name in extra_mod.classes:
                parts.extend([_render_class(extra_mod.classes[class_name]), ""])

    return "\n".join(parts).strip()


def _render_grouped_module(module_name: str, groups: dict[str, list[str]]) -> str:
    module = _load_module(module_name)
    parts: list[str] = []
    documented: set[str] = set()

    for heading, class_names in groups.items():
        rows: list[tuple[str, str]] = []
        for class_name in class_names:
            if class_name not in module.classes:
                continue
            cls = module.classes[class_name]
            if not _is_defined_in_module(cls, module_name):
                continue
            rows.append((class_name, _doc(cls)))
            documented.add(class_name)
        if not rows:
            continue
        parts.extend([f"## {heading}", "", _render_summary_table(rows), ""])
        for class_name in class_names:
            if class_name not in documented:
                continue
            parts.extend([_render_class(module.classes[class_name]), ""])

    remaining = [
        n
        for n in sorted(module.classes.keys())
        if not n.startswith("_")
        and n not in documented
        and _is_defined_in_module(module.classes[n], module_name)
    ]
    if remaining:
        rows = [
            (class_name, _doc(module.classes[class_name])) for class_name in remaining
        ]
        parts.extend(["## Other", "", _render_summary_table(rows), ""])
        for class_name in remaining:
            parts.extend([_render_class(module.classes[class_name]), ""])

    for name in sorted(module.functions.keys()):
        if name.startswith("_"):
            continue
        parts.extend([_render_function(module.functions[name]), ""])

    return "\n".join(parts).strip()


def _render_adapters_summary() -> str:
    parts: list[str] = []
    for heading, entries in ADAPTERS_AUTOSUMMARY.items():
        rows: list[tuple[str, str]] = []
        for class_name, page, module_path in entries:
            link = (
                f"/references/api/nemo-evaluator/adapters/{page}#{_slugify(class_name)}"
            )
            mod = _load_module(module_path)
            cls = mod.classes.get(class_name)
            desc = _doc(cls) if cls is not None else ""
            rows.append((f"[{class_name}]({link})", desc))
        parts.extend(
            [f"## {heading}", "", _render_summary_table(rows, link_names=True), ""]
        )
    return "\n".join(parts).strip()


def _draft_pages() -> dict[str, str]:
    return {
        "api": _render_module("nemo_evaluator.api"),
        "api-dataclasses": _render_grouped_module(
            "nemo_evaluator.api.api_dataclasses", API_DATACLASSES_GROUPS
        ),
        "adapters-summary": _render_adapters_summary(),
        "adapter-config": _render_module(
            "nemo_evaluator.adapters.adapter_config",
            extra_class_paths=[
                "nemo_evaluator.adapters.reports.post_eval_report_hook.PostEvalReportHook"
            ],
        ),
        "interceptors": _render_module("nemo_evaluator.adapters.interceptors"),
        "types": _render_module("nemo_evaluator.adapters.types"),
        "sandbox": _render_module("nemo_evaluator.sandbox"),
    }


def _emit_draft_mdx(output_dir: pathlib.Path) -> list[pathlib.Path]:
    _clear_caches()
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[pathlib.Path] = []
    for page_id, body in _draft_pages().items():
        path = output_dir / f"{page_id}.mdx"
        path.write_text(body.strip() + "\n", encoding="utf-8")
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draft Core Python API reference MDX from Griffe (optional local tool)."
    )
    parser.add_argument(
        "--docs-root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent.parent,
        help="Docs content root (default: docs/)",
    )
    parser.add_argument(
        "--draft-dir",
        type=pathlib.Path,
        help="Write draft MDX fragments here for comparison (not wired into docs-autogen)",
    )
    args = parser.parse_args()
    if args.draft_dir is None:
        parser.error(
            "Core API reference pages are hand-maintained under "
            "docs/references/api/nemo-evaluator/. Pass --draft-dir to emit Griffe drafts."
        )
    draft_dir = args.draft_dir.resolve()
    paths = _emit_draft_mdx(draft_dir)
    _remove_legacy_rst(args.docs_root.resolve())
    print(f"Wrote {len(paths)} draft MDX files under {draft_dir}")


def _remove_legacy_rst(output_root: pathlib.Path) -> None:
    ref_root = output_root / "references" / "api" / "nemo-evaluator"
    for pattern in (
        "api/*.rst",
        "adapters/*.rst",
        "sandbox/*.rst",
    ):
        for path in ref_root.glob(pattern):
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
