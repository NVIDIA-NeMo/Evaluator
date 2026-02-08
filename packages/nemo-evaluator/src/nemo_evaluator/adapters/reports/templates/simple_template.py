# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains the HTML template for generating reports.

Note: The template is stored as a string in a Python file as a workaround
because many of our pip packages have issues with including .html files
in their distributions. This allows the template to be properly packaged
and distributed with the rest of the code.
"""

SIMPLE_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Evaluation Report</title>
    <style>
        :root {
            --bg: #0f1116;
            --bg-accent: #1a1f2b;
            --panel: #141824;
            --panel-2: #10131b;
            --text: #e8ecf1;
            --muted: #9aa6b2;
            --accent: #60d394;
            --accent-2: #6f9bff;
            --warn: #f5c451;
            --border: rgba(255, 255, 255, 0.08);
            --shadow: 0 12px 28px rgba(0, 0, 0, 0.45);
        }

        body {
            font-family: "Space Grotesk", "IBM Plex Sans", "Source Sans 3", sans-serif;
            line-height: 1.5;
            margin: 0;
            color: var(--text);
            background: radial-gradient(1200px 700px at 15% -20%, #1e2a3b 0%, rgba(15, 17, 22, 0) 65%),
                        radial-gradient(900px 900px at 110% 10%, #1f2440 0%, rgba(15, 17, 22, 0) 60%),
                        var(--bg);
            padding: 32px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            font-size: 1.9rem;
            margin: 0 0 8px;
        }
        .subtitle {
            color: var(--muted);
            margin: 0 0 24px;
            font-size: 1.02rem;
        }
        .hero {
            background: linear-gradient(135deg, rgba(96, 211, 148, 0.08), rgba(111, 155, 255, 0.08));
            border: 1px solid var(--border);
            padding: 22px 24px;
            border-radius: 14px;
            box-shadow: var(--shadow);
            margin-bottom: 18px;
        }
        .brand-bar {
            display: flex;
            flex-wrap: wrap;
            gap: 16px;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 16px;
        }
        .brand-left {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        .brand-header {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 10px;
        }
        .brand-logo {
            display: inline-flex;
            align-items: center;
            gap: 10px;
        }
        .nvidia-logo {
            display: inline-flex;
            align-items: center;
            gap: 10px;
        }
        .nvidia-eye {
            width: 38px;
            height: 26px;
            flex: 0 0 auto;
        }
        .nvidia-word {
            font-weight: 800;
            letter-spacing: 0.16em;
            font-size: 0.9rem;
            color: #76b900;
            text-transform: uppercase;
        }
        .brand-title {
            font-size: 1.1rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }
        .brand-sub {
            color: var(--muted);
            font-size: 0.85rem;
            margin-top: 2px;
        }
        .hero-pills {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin: 10px 0 6px;
        }
        .pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 12px;
            border-radius: 999px;
            border: 1px solid var(--border);
            background: rgba(118, 185, 0, 0.12);
            box-shadow: inset 0 0 0 1px rgba(118, 185, 0, 0.12);
            font-size: 0.82rem;
        }
        .pill-label {
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.62rem;
            color: var(--muted);
        }
        .pill-value {
            color: var(--text);
            font-weight: 600;
        }
        .pill-link {
            color: var(--text);
            font-weight: 600;
            text-decoration: none;
        }
        .pill-link:hover {
            text-decoration: underline;
        }
        .pill-model {
            background: rgba(111, 155, 255, 0.14);
            border-color: rgba(111, 155, 255, 0.35);
            box-shadow: inset 0 0 0 1px rgba(111, 155, 255, 0.16);
        }
        .pill-model .pill-label {
            color: rgba(208, 223, 255, 0.9);
        }
        .pill-benchmark {
            background: rgba(118, 185, 0, 0.14);
            border-color: rgba(118, 185, 0, 0.38);
            box-shadow: inset 0 0 0 1px rgba(118, 185, 0, 0.14);
        }
        .pill-benchmark .pill-label {
            color: rgba(204, 232, 190, 0.92);
        }
        .pill-container {
            background: rgba(255, 255, 255, 0.06);
            border-color: rgba(255, 255, 255, 0.15);
            box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.08);
        }
        .brand-link {
            color: var(--accent-2);
            text-decoration: none;
            font-size: 0.85rem;
            border: 1px solid rgba(111, 155, 255, 0.35);
            padding: 6px 12px;
            border-radius: 999px;
            background: rgba(111, 155, 255, 0.08);
        }
        .brand-link:hover {
            filter: brightness(1.1);
        }
        .brand-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }
        .meta-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 12px;
            border-radius: 999px;
            border: 1px solid var(--border);
            background: rgba(255, 255, 255, 0.04);
            font-size: 0.85rem;
            color: var(--muted);
        }
        .meta-pill strong {
            color: var(--text);
        }
        .hero strong {
            color: var(--accent);
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 12px;
            margin: 20px 0 26px;
        }
        .summary-card {
            background: var(--panel);
            border: 1px solid var(--border);
            padding: 14px 16px;
            border-radius: 12px;
        }
        .summary-card .label {
            color: var(--muted);
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .summary-card .value {
            font-size: 1.15rem;
            margin-top: 6px;
            word-break: break-word;
        }
        .toolbar {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            align-items: center;
            margin-bottom: 16px;
        }
        .tabs {
            display: inline-flex;
            gap: 8px;
            margin: 10px 0 16px;
            flex-wrap: wrap;
        }
        .tab {
            background: var(--panel-2);
            border: 1px solid var(--border);
            color: var(--text);
            padding: 8px 14px;
            border-radius: 999px;
            cursor: pointer;
            font-size: 0.85rem;
        }
        .tab.active {
            background: var(--accent-2);
            color: #0b0f14;
            border-color: rgba(111, 155, 255, 0.5);
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .table-wrap {
            overflow-x: auto;
        }
        .sample-table {
            width: 100%;
            min-width: 760px;
            table-layout: fixed;
        }
        .sample-table th {
            white-space: nowrap;
            word-break: normal;
            text-align: left;
        }
        .sample-table .idx-col { width: 56px; }
        .sample-table .target-col { width: 180px; min-width: 180px; }
        .sample-table .score-col { width: 120px; min-width: 120px; text-align: center; }
        .sample-table .details-col { width: 90px; min-width: 90px; text-align: center; }
        .sample-table .answer-col { width: 35%; }
        .sample-table .input-col { width: auto; }
        .score-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.75rem;
            border: 1px solid var(--border);
            white-space: nowrap;
        }
        .score-badge.correct {
            background: rgba(126, 224, 129, 0.16);
            color: #7ee081;
            border-color: rgba(126, 224, 129, 0.35);
        }
        .score-badge.incorrect {
            background: rgba(255, 154, 154, 0.16);
            color: #ff9a9a;
            border-color: rgba(255, 154, 154, 0.35);
        }
        .score-badge.unknown {
            background: rgba(246, 211, 101, 0.18);
            color: #f6d365;
            border-color: rgba(246, 211, 101, 0.45);
        }
        .sample-row:hover {
            background: rgba(111, 155, 255, 0.08);
        }
        .sample-row.grade-correct {
            background: rgba(126, 224, 129, 0.06);
        }
        .sample-row.grade-incorrect {
            background: rgba(255, 154, 154, 0.06);
        }
        .sample-row.grade-unknown {
            background: rgba(246, 211, 101, 0.08);
        }
        .sample-table td {
            border-right: 1px solid rgba(255, 255, 255, 0.04);
        }
        .sample-table td:last-child,
        .sample-table th:last-child {
            border-right: none;
        }
        .details-link {
            color: var(--accent-2);
            text-decoration: none;
            font-size: 0.78rem;
        }
        .details-link:hover {
            text-decoration: underline;
        }
        .metric-path {
            max-width: 420px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .metric-value {
            white-space: nowrap;
        }
        .sample-table td, .sample-table th {
            vertical-align: top;
            line-height: 1.35;
        }
        .sample-table td.score-col,
        .sample-table th.score-col,
        .sample-table td.details-col,
        .sample-table th.details-col {
            white-space: nowrap;
            word-break: normal;
        }
        .sample-table td.target-col,
        .sample-table th.target-col {
            white-space: normal;
            word-break: break-word;
            overflow-wrap: anywhere;
        }
        .sample-row.grade-correct td:first-child {
            border-left: 3px solid rgba(126, 224, 129, 0.7);
        }
        .sample-row.grade-incorrect td:first-child {
            border-left: 3px solid rgba(255, 154, 154, 0.7);
        }
        .sample-row.grade-unknown td:first-child {
            border-left: 3px solid rgba(199, 199, 199, 0.4);
        }
        .sample-table .muted {
            color: var(--muted);
            font-size: 0.78rem;
        }
        .search {
            flex: 1;
            min-width: 220px;
            background: var(--panel-2);
            border: 1px solid var(--border);
            padding: 10px 14px;
            border-radius: 999px;
            color: var(--text);
            font-size: 0.95rem;
        }
        .toggle-group {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }
        .toggle {
            background: var(--panel);
            border: 1px solid var(--border);
            color: var(--text);
            padding: 8px 12px;
            border-radius: 999px;
            cursor: pointer;
            font-size: 0.85rem;
        }
        .toggle.active {
            background: var(--accent);
            color: #0b0f14;
            border-color: rgba(96, 211, 148, 0.6);
        }
        .select {
            background: var(--panel-2);
            border: 1px solid var(--border);
            color: var(--text);
            padding: 8px 12px;
            border-radius: 10px;
            font-size: 0.85rem;
        }
        .check {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            color: var(--muted);
            font-size: 0.85rem;
        }
        .visible-count {
            color: var(--muted);
            font-size: 0.85rem;
        }
        .note {
            color: var(--muted);
            font-size: 0.9rem;
            margin: 8px 0 18px;
        }
        .note strong {
            color: var(--warn);
        }
        .chips {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }
        .chip {
            background: var(--panel-2);
            border: 1px solid var(--border);
            color: var(--muted);
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.78rem;
        }
        .section {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 0;
            margin-bottom: 18px;
            box-shadow: var(--shadow);
            overflow: hidden;
        }
        .section h2 {
            margin: 0;
            font-size: 1.15rem;
        }
        .section-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            padding: 16px 20px;
            cursor: pointer;
        }
        .section-toggle {
            border: 1px solid var(--border);
            background: rgba(255, 255, 255, 0.04);
            color: var(--muted);
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.72rem;
            cursor: pointer;
        }
        .section-toggle:hover {
            color: var(--text);
        }
        .section-body {
            padding: 0 20px 18px;
        }
        .section.collapsed .section-body {
            display: none;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }
        th, td {
            text-align: left;
            padding: 10px 8px;
            border-bottom: 1px solid var(--border);
            vertical-align: top;
            word-break: break-word;
        }
        th {
            color: var(--muted);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        details {
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 10px 12px;
            margin-top: 12px;
            background: var(--panel-2);
        }
        details summary {
            cursor: pointer;
            color: var(--accent-2);
            font-weight: 600;
        }

        .entry {
            margin: 16px 0;
            padding: 18px;
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 14px;
            box-shadow: var(--shadow);
            scroll-margin-top: 24px;
            position: relative;
            overflow: hidden;
        }
        .entry::before {
            content: "";
            position: absolute;
            left: 0;
            top: 0;
            height: 100%;
            width: 4px;
            background: rgba(199, 199, 199, 0.4);
        }
        .entry.grade-correct {
            border-color: rgba(126, 224, 129, 0.4);
            box-shadow: 0 0 0 1px rgba(126, 224, 129, 0.08), var(--shadow);
            background: linear-gradient(90deg, rgba(126, 224, 129, 0.06), rgba(20, 24, 36, 0) 45%), var(--panel);
        }
        .entry.grade-incorrect {
            border-color: rgba(255, 154, 154, 0.4);
            box-shadow: 0 0 0 1px rgba(255, 154, 154, 0.08), var(--shadow);
            background: linear-gradient(90deg, rgba(255, 154, 154, 0.06), rgba(20, 24, 36, 0) 45%), var(--panel);
        }
        .entry.grade-unknown {
            border-color: rgba(246, 211, 101, 0.35);
            background: linear-gradient(90deg, rgba(246, 211, 101, 0.07), rgba(20, 24, 36, 0) 45%), var(--panel);
        }
        .entry.grade-correct::before {
            background: rgba(126, 224, 129, 0.9);
        }
        .entry.grade-incorrect::before {
            background: rgba(255, 154, 154, 0.9);
        }
        .entry.grade-unknown::before {
            background: rgba(246, 211, 101, 0.9);
        }
        .entry[data-graded="correct"] {
            border-color: rgba(126, 224, 129, 0.4);
            box-shadow: 0 0 0 1px rgba(126, 224, 129, 0.08), var(--shadow);
            background: linear-gradient(90deg, rgba(126, 224, 129, 0.06), rgba(20, 24, 36, 0) 45%), var(--panel);
        }
        .entry[data-graded="incorrect"] {
            border-color: rgba(255, 154, 154, 0.4);
            box-shadow: 0 0 0 1px rgba(255, 154, 154, 0.08), var(--shadow);
            background: linear-gradient(90deg, rgba(255, 154, 154, 0.06), rgba(20, 24, 36, 0) 45%), var(--panel);
        }
        .entry[data-graded="unknown"] {
            border-color: rgba(246, 211, 101, 0.35);
            background: linear-gradient(90deg, rgba(246, 211, 101, 0.07), rgba(20, 24, 36, 0) 45%), var(--panel);
        }
        .entry[data-graded="correct"]::before {
            background: rgba(126, 224, 129, 0.9);
        }
        .entry[data-graded="incorrect"]::before {
            background: rgba(255, 154, 154, 0.9);
        }
        .entry[data-graded="unknown"]::before {
            background: rgba(246, 211, 101, 0.9);
        }
        .entry.highlight {
            outline: 2px solid rgba(111, 155, 255, 0.5);
            outline-offset: 2px;
        }
        .entry-header {
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin-bottom: 12px;
        }
        .entry-top {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            align-items: center;
            justify-content: space-between;
        }
        .meta-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            align-items: center;
        }
        .meta-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 10px 12px;
        }
        .meta-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 8px 12px;
            min-width: 0;
        }
        .meta-card.span-2 {
            grid-column: 1 / -1;
        }
        .meta-label {
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.68rem;
            color: var(--muted);
        }
        .meta-value {
            margin-top: 4px;
            color: var(--text);
            word-break: break-word;
            overflow-wrap: anywhere;
        }
        .badge {
            padding: 2px 8px;
            border-radius: 999px;
            font-size: 0.72rem;
            border: 1px solid var(--border);
        }
        .badge.error {
            color: #ffb4a2;
            border-color: rgba(255, 180, 162, 0.4);
            background: rgba(255, 180, 162, 0.1);
        }
        .badge.info {
            color: #9bd0ff;
            border-color: rgba(155, 208, 255, 0.35);
            background: rgba(155, 208, 255, 0.12);
        }
        .badge.correct {
            color: #7ee081;
            border-color: rgba(126, 224, 129, 0.4);
            background: rgba(126, 224, 129, 0.12);
        }
        .badge.incorrect {
            color: #ff9a9a;
            border-color: rgba(255, 154, 154, 0.4);
            background: rgba(255, 154, 154, 0.12);
        }
        .badge.unknown {
            color: #f6d365;
            border-color: rgba(246, 211, 101, 0.45);
            background: rgba(246, 211, 101, 0.16);
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.78rem;
            border: 1px solid var(--border);
            background: var(--panel-2);
        }
        .status-pill.correct {
            color: #7ee081;
            border-color: rgba(126, 224, 129, 0.4);
            background: rgba(126, 224, 129, 0.12);
        }
        .status-pill.incorrect {
            color: #ff9a9a;
            border-color: rgba(255, 154, 154, 0.4);
            background: rgba(255, 154, 154, 0.12);
        }
        .status-pill.unknown {
            color: #f6d365;
            border-color: rgba(246, 211, 101, 0.45);
            background: rgba(246, 211, 101, 0.16);
        }
        .prompt-preview {
            margin-top: 10px;
            padding: 10px 12px;
            background: rgba(255, 255, 255, 0.03);
            border: 1px dashed var(--border);
            border-radius: 10px;
            color: var(--text);
            font-size: 0.88rem;
        }
        .sample-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(111, 155, 255, 0.14);
            color: #bcd2ff;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.78rem;
            border: 1px solid rgba(111, 155, 255, 0.35);
        }
        .block {
            background: #0c0f16;
            border: 1px solid var(--border);
            padding: 12px 14px;
            border-radius: 10px;
            margin-bottom: 10px;
        }
        .block-title {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--muted);
            margin-bottom: 8px;
        }
        pre {
            white-space: pre-wrap;
            word-break: break-word;
            margin: 0;
            font-family: "IBM Plex Mono", "SFMono-Regular", ui-monospace, monospace;
            font-size: 0.85rem;
            color: #d9e2ef;
        }
        .buttons {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 10px;
        }
        .button {
            padding: 6px 12px;
            border: 1px solid var(--border);
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.8rem;
            background-color: var(--panel-2);
            color: var(--text);
        }
        .button.primary {
            background-color: var(--accent-2);
            color: #0b0f14;
            border-color: rgba(111, 155, 255, 0.4);
        }
        .button:hover {
            filter: brightness(1.1);
        }
        .hidden {
            display: none !important;
        }
        .empty {
            padding: 40px 20px;
            text-align: center;
            color: var(--muted);
            border: 1px dashed var(--border);
            border-radius: 12px;
            margin-top: 20px;
        }
        @media (max-width: 720px) {
            body {
                padding: 20px;
            }
            h1 {
                font-size: 1.6rem;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        {% if entries %}
            <div class="brand-bar">
                <div class="brand-left">
                    <div class="brand-logo">
                        <span class="nvidia-logo">
                            <svg class="nvidia-eye" viewBox="0 0 64 40" role="img" aria-label="NVIDIA logo">
                                <defs>
                                    <linearGradient id="nvidiaGreen" x1="0" x2="1" y1="0" y2="1">
                                        <stop offset="0%" stop-color="#76b900"/>
                                        <stop offset="100%" stop-color="#8cd63f"/>
                                    </linearGradient>
                                </defs>
                                <rect x="1" y="1" width="62" height="38" rx="9" fill="url(#nvidiaGreen)" />
                                <path d="M10 20c8-8 16-12 22-12 6 0 14 4 22 12-8 8-16 12-22 12-6 0-14-4-22-12z" fill="rgba(8, 18, 8, 0.2)" />
                                <circle cx="32" cy="20" r="7" fill="#0b1a0d" />
                                <circle cx="32" cy="20" r="3.2" fill="#76b900" />
                            </svg>
                            <span class="nvidia-word">NVIDIA</span>
                        </span>
                    </div>
                    <div class="brand-header">
                        <div class="brand-title">NeMo Evaluator</div>
                        <a class="brand-link" href="https://github.com/NVIDIA-NeMo/Evaluator" target="_blank" rel="noopener">GitHub Repo</a>
                    </div>
                    <div class="brand-sub">Evaluation Report</div>
                </div>
            </div>
            <div class="hero">
                <h1>Evaluation Report</h1>
                {% if meta.model_id or meta.task_name or meta.benchmark_container %}
                    <div class="hero-pills">
                        {% if meta.task_name %}
                            <span class="pill pill-benchmark"><span class="pill-label">Benchmark</span><span class="pill-value">{{ meta.task_name }}</span></span>
                        {% endif %}
                        {% if meta.model_id %}
                            <span class="pill pill-model"><span class="pill-label">Model</span><span class="pill-value">{{ meta.model_id }}</span></span>
                        {% endif %}
                        {% if meta.benchmark_container %}
                            <span class="pill pill-container">
                                <span class="pill-label">Container</span>
                                {% if meta.benchmark_container_url %}
                                    <a class="pill-link" href="{{ meta.benchmark_container_url }}" target="_blank" rel="noopener">{{ meta.benchmark_container }}</a>
                                {% else %}
                                    <span class="pill-value">{{ meta.benchmark_container }}</span>
                                {% endif %}
                            </span>
                        {% endif %}
                    </div>
                {% endif %}
                <p class="subtitle">Curated request and response samples captured during evaluation. Built for quick review, sharing, and debugging.</p>
            </div>

            {% if meta.limited %}
                <div class="note"><strong>Limited:</strong> Showing the first {{ meta.shown_entries }} entries of {{ meta.total_entries }} due to HTML report size settings.</div>
            {% endif %}

            <div class="toolbar">
                
                <select id="modelFilter" class="select">
                    <option value="">All models</option>
                    {% for model in meta.filters.models %}
                        <option value="{{ model }}">{{ model }}</option>
                    {% endfor %}
                </select>
                <select id="typeFilter" class="select">
                    <option value="">All request types</option>
                    {% for req_type in meta.filters.request_types %}
                        <option value="{{ req_type }}">{{ req_type }}</option>
                    {% endfor %}
                </select>
                <select id="finishFilter" class="select">
                    <option value="">All finish reasons</option>
                    {% for reason in meta.filters.finish_reasons %}
                        <option value="{{ reason }}">{{ reason }}</option>
                    {% endfor %}
                </select>
                <select id="gradingFilter" class="select">
                    <option value="">All grading</option>
                    {% for grade in meta.filters.grading %}
                        <option value="{{ grade }}">{{ grade }}</option>
                    {% endfor %}
                </select>
                <label class="check"><input type="checkbox" id="errorOnly" /> Errors only</label>
                <label class="check"><input type="checkbox" id="toolOnly" /> Tool calls</label>
                <select id="sortSelect" class="select">
                    <option value="cache">Sort: cache key</option>
                    <option value="request">Sort: request length</option>
                    <option value="response">Sort: response length</option>
                </select>
                <span class="visible-count" id="visibleCount"></span>
                <div class="toggle-group">
                    <button class="toggle active" data-target="request">Requests</button>
                    <button class="toggle active" data-target="response">Responses</button>
                    <button class="toggle" data-target="graded">Graded</button>
                    <button class="toggle" data-target="curl">Curl</button>
                </div>
            </div>

            {% if meta.summary_items %}
                <div class="section">
                    <h2>Run Summary</h2>
                    <div class="summary-grid">
                        {% for item in meta.summary_items %}
                            <div class="summary-card">
                                <div class="label">{{ item.label }}</div>
                                <div class="value">{{ item.value }}</div>
                            </div>
                        {% endfor %}
                    </div>
                </div>
            {% endif %}

            {% if meta.stats %}
                <div class="section">
                    <h2>Sample Stats</h2>
                    <div class="summary-grid">
                        <div class="summary-card">
                            <div class="label">Report Accuracy (Graded)</div>
                            <div class="value">
                                {% if meta.stats.graded_count %}
                                    {{ (meta.stats.accuracy * 100) | round(2) }}% ({{ meta.stats.correct_count }}/{{ meta.stats.graded_count }})
                                {% else %}
                                    —
                                {% endif %}
                            </div>
                        </div>
                        <div class="summary-card">
                            <div class="label">Correct / Incorrect / Unknown</div>
                            <div class="value">{{ meta.stats.correct_count }} / {{ meta.stats.incorrect_count }} / {{ meta.stats.unknown_count }}</div>
                        </div>
                        <div class="summary-card">
                            <div class="label">Report Samples</div>
                            <div class="value">{{ meta.stats.report_count }} shown / {{ meta.total_entries }} cached</div>
                        </div>
                        <div class="summary-card">
                            <div class="label">Error Rate</div>
                            <div class="value">{{ (meta.stats.error_rate * 100) | round(2) }}% ({{ meta.stats.error_count }})</div>
                        </div>
                        <div class="summary-card">
                            <div class="label">Tool Calls</div>
                            <div class="value">{{ meta.stats.tool_count }}</div>
                        </div>
                        <div class="summary-card">
                            <div class="label">Avg Request Chars</div>
                            <div class="value">{{ meta.stats.avg_request_chars | round(0) }}</div>
                        </div>
                        <div class="summary-card">
                            <div class="label">Avg Response Chars</div>
                            <div class="value">{{ meta.stats.avg_response_chars | round(0) }}</div>
                        </div>
                        {% if meta.stats.usage_totals.prompt_tokens or meta.stats.usage_totals.completion_tokens or meta.stats.usage_totals.total_tokens %}
                            <div class="summary-card">
                                <div class="label">Tokens (Prompt / Completion / Total)</div>
                                <div class="value">{{ meta.stats.usage_totals.prompt_tokens }} / {{ meta.stats.usage_totals.completion_tokens }} / {{ meta.stats.usage_totals.total_tokens }}</div>
                            </div>
                        {% endif %}
                    </div>

                    {% if meta.stats.request_type_counts %}
                        <div class="chips">
                            {% for key, value in meta.stats.request_type_counts.items() %}
                                {% if key != 'unknown' %}
                                    <div class="chip">{{ key }}: {{ value }}</div>
                                {% endif %}
                            {% endfor %}
                        </div>
                    {% endif %}
                    {% if meta.stats.finish_reason_counts %}
                        <div class="chips">
                            {% for key, value in meta.stats.finish_reason_counts.items() %}
                                {% if key != 'unknown' %}
                                    <div class="chip">{{ key }}: {{ value }}</div>
                                {% endif %}
                            {% endfor %}
                        </div>
                    {% endif %}
                    <div class="note">
                        <strong>Note:</strong> Report accuracy uses graded log samples shown above (correct / incorrect).
                        Results rollups below come from results.yml and may use task-specific definitions (inst-level vs prompt-level)
                        and the full eval set.
                    </div>
                </div>
            {% endif %}

            <div class="section">
                <h2>Samples</h2>
                <div class="tabs">
                    <button class="tab active" data-tab="tableView">Table</button>
                    <button class="tab" data-tab="detailView">Details</button>
                </div>
                <div id="tableView" class="tab-content active">
                    <div class="table-wrap">
                        <table class="sample-table">
                            <thead>
                                <tr>
                                    <th class="idx-col">#</th>
                                    <th class="input-col">Input</th>
                                    <th class="target-col">Target</th>
                                    <th class="answer-col">Answer</th>
                                    <th class="score-col">Score</th>
                                    <th class="details-col">Details</th>
                                </tr>
                            </thead>
                            <tbody>
                                {% for entry in entries %}
                                    <tr class="sample-row grade-{{ entry.graded_label or 'unknown' }}"
                                        data-search="{{ entry.search_blob|e }}"
                                        data-cache="{{ entry.cache_key|e }}"
                                        data-model="{{ entry.model|e }}"
                                        data-request-type="{{ entry.request_type|e }}"
                                        data-has-error="{{ 'true' if entry.has_error else 'false' }}"
                                        data-has-tool="{{ 'true' if entry.has_tool_calls else 'false' }}"
                                        data-finish-reasons="{{ entry.finish_reasons | join(',') | e }}"
                                        data-graded="{{ entry.graded_label | e }}"
                                        data-request-chars="{{ entry.request_chars }}"
                                        data-response-chars="{{ entry.response_chars }}">
                                        <td class="idx-col">
                                            <a class="details-link" data-detail-link="true" href="#sample-{{ entry.cache_key }}">{{ loop.index }}</a>
                                        </td>
                                        <td class="input-col">{{ (entry.graded_input or entry.prompt_preview or '—') | truncate(160, True, '...') }}</td>
                                        <td class="target-col">{{ (entry.target_display or entry.graded_target or entry.graded_expected or '—') | truncate(80, True, '...') }}</td>
                                        <td class="answer-col">{{ (entry.graded_response or entry.graded_predicted or '—') | truncate(120, True, '...') }}</td>
                                        <td class="score-col">
                                            {% if entry.graded_label == 'correct' %}
                                                <span class="score-badge correct">Correct</span>
                                            {% elif entry.graded_label == 'incorrect' %}
                                                <span class="score-badge incorrect">Incorrect</span>
                                            {% else %}
                                                <span class="score-badge unknown">Ungraded</span>
                                            {% endif %}
                                        </td>
                                        <td class="details-col">
                                            <a class="details-link" data-detail-link="true" href="#sample-{{ entry.cache_key }}">View</a>
                                        </td>
                                    </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    </div>
                </div>
                <div id="detailView" class="tab-content">
                    <div id="entriesContainer">
                        {% for entry in entries %}
                                    <div class="entry grade-{{ entry.graded_label or 'unknown' }}"
                                         id="sample-{{ entry.cache_key }}"
                                         data-search="{{ entry.search_blob|e }}"
                                         data-cache="{{ entry.cache_key|e }}"
                                         data-model="{{ entry.model|e }}"
                                 data-request-type="{{ entry.request_type|e }}"
                                 data-has-error="{{ 'true' if entry.has_error else 'false' }}"
                                 data-has-tool="{{ 'true' if entry.has_tool_calls else 'false' }}"
                                 data-finish-reasons="{{ entry.finish_reasons | join(',') | e }}"
                                 data-graded="{{ entry.graded_label | e }}"
                                 data-request-chars="{{ entry.request_chars }}"
                                 data-response-chars="{{ entry.response_chars }}">
                                <div class="entry-header">
                                    <div class="entry-top">
                                        <div class="meta-row">
                                            <span class="sample-pill">Sample {{ loop.index }}</span>
                                            {% if entry.graded_label == 'correct' %}
                                                <span class="status-pill correct">Correct</span>
                                            {% elif entry.graded_label == 'incorrect' %}
                                                <span class="status-pill incorrect">Incorrect</span>
                                            {% else %}
                                                <span class="status-pill unknown">Ungraded</span>
                                            {% endif %}
                                            {% if entry.has_error %}
                                                <span class="badge error">Error</span>
                                            {% endif %}
                                            {% if entry.has_tool_calls %}
                                                <span class="badge info">Tool calls</span>
                                            {% endif %}
                                        </div>
                                        <div class="buttons">
                                            <button class="button" onclick="copyBlock('req-{{ entry.cache_key }}')">Copy Request</button>
                                            <button class="button" onclick="copyBlock('res-{{ entry.cache_key }}')">Copy Response</button>
                                            <button class="button primary" onclick="toggleRaw('curl-{{ entry.cache_key }}')">Toggle Curl</button>
                                        </div>
                                    </div>
                                    <div class="meta-grid">
                                        <div class="meta-card span-2">
                                            <div class="meta-label">Cache Key</div>
                                            <div class="meta-value">{{ entry.cache_key }}</div>
                                        </div>
                                        <div class="meta-card">
                                            <div class="meta-label">Model</div>
                                            <div class="meta-value">{{ entry.model }}</div>
                                        </div>
                                        <div class="meta-card">
                                            <div class="meta-label">Type</div>
                                            <div class="meta-value">{{ entry.request_type }}</div>
                                        </div>
                                        {% if entry.finish_reasons %}
                                            <div class="meta-card">
                                                <div class="meta-label">Finish</div>
                                                <div class="meta-value">{{ entry.finish_reasons | join(', ') }}</div>
                                            </div>
                                        {% endif %}
                                        <div class="meta-card">
                                            <div class="meta-label">Req chars</div>
                                            <div class="meta-value">{{ entry.request_chars }}</div>
                                        </div>
                                        <div class="meta-card">
                                            <div class="meta-label">Resp chars</div>
                                            <div class="meta-value">{{ entry.response_chars }}</div>
                                        </div>
                                        {% if entry.usage %}
                                            <div class="meta-card">
                                                <div class="meta-label">Tokens</div>
                                                <div class="meta-value">{{ entry.usage.prompt_tokens }} / {{ entry.usage.completion_tokens }} / {{ entry.usage.total_tokens }}</div>
                                            </div>
                                        {% endif %}
                                        {% if entry.target_display %}
                                            <div class="meta-card span-2">
                                                <div class="meta-label">Target</div>
                                                <div class="meta-value">{{ entry.target_display }}</div>
                                            </div>
                                        {% elif entry.graded_expected %}
                                            <div class="meta-card span-2">
                                                <div class="meta-label">Expected</div>
                                                <div class="meta-value">{{ entry.graded_expected }}</div>
                                            </div>
                                        {% endif %}
                                        {% if entry.graded_predicted %}
                                            <div class="meta-card span-2">
                                                <div class="meta-label">Predicted</div>
                                                <div class="meta-value">{{ entry.graded_predicted }}</div>
                                            </div>
                                        {% endif %}
                                    </div>
                                </div>
                                {% if entry.prompt_sections %}
                                    {% for section in entry.prompt_sections %}
                                        <div class="prompt-preview">
                                            <div class="block-title">{{ section.label }}</div>
                                            <div>{{ section.text }}</div>
                                        </div>
                                    {% endfor %}
                                {% endif %}
                                <div class="block request-block" data-section="request">
                                    <div class="block-title">Request</div>
                                    <pre id="req-{{ entry.cache_key }}">{{ entry.display_request|tojson_utf8|safe }}</pre>
                                </div>
                                <div class="block response-block" data-section="response">
                                    <div class="block-title">Response</div>
                                    <pre id="res-{{ entry.cache_key }}">{{ entry.response|tojson_utf8|safe }}</pre>
                                </div>
                                {% if entry.graded_response %}
                                    <div class="block graded-block" data-section="graded">
                                        <div class="block-title">Graded Response</div>
                                        <pre id="graded-{{ entry.cache_key }}">{{ entry.graded_response | e }}</pre>
                                    </div>
                                {% endif %}
                                {% if entry.target_display %}
                                    <div class="block graded-block" data-section="graded">
                                        <div class="block-title">Target (Ground Truth)</div>
                                        <pre>{{ entry.target_display | e }}</pre>
                                    </div>
                                {% endif %}
                                <div id="curl-{{ entry.cache_key }}" class="block hidden" data-section="curl">
                                    <div class="block-title">Repro Curl</div>
                                    <pre># Save payload to file first:
echo '{{ entry.request_data|tojson }}' > request.json

curl "{{ entry.endpoint }}" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Accept: application/json" \
  -d @request.json</pre>
                                </div>
                            </div>
                        {% endfor %}
                    </div>
                </div>
            </div>

            <div class="section">
                <h2>Results Rollup (Groups)</h2>
                {% if meta.rollup_rows %}
                    <table>
                        <thead>
                            <tr>
                                <th>Group</th>
                                <th>Metric</th>
                                <th>Value</th>
                                <th>Stats</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for row in meta.rollup_rows %}
                                <tr>
                                    <td>{{ row.scope }}</td>
                                    <td class="metric-path" title="{{ row.path }}">{{ row.path_display }}</td>
                                    <td class="metric-value">{{ row.value_display or '—' }}</td>
                                    <td>{{ row.stats_display or '—' }}</td>
                                </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                {% else %}
                    <div class="note">No rollup metrics were found in results.yml.</div>
                {% endif %}
            </div>

            <div class="section">
                <h2>Results (Tasks)</h2>
                {% if meta.task_rows %}
                    <table>
                        <thead>
                            <tr>
                                <th>Task</th>
                                <th>Metric</th>
                                <th>Value</th>
                                <th>Stats</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for row in meta.task_rows %}
                                <tr>
                                    <td>{{ row.scope }}</td>
                                    <td class="metric-path" title="{{ row.path }}">{{ row.path_display }}</td>
                                    <td class="metric-value">{{ row.value_display or '—' }}</td>
                                    <td>{{ row.stats_display or '—' }}</td>
                                </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                {% else %}
                    <div class="note">No task metrics were found in results.yml.</div>
                {% endif %}
            </div>

            <div class="section">
                <h2>All Metrics (Flattened)</h2>
                {% if meta.metrics_rows %}
                    <table>
                        <thead>
                            <tr>
                                <th>Metric Path</th>
                                <th>Value</th>
                                <th>Stats</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for row in meta.metrics_rows %}
                                <tr>
                                    <td class="metric-path" title="{{ row.path }}">{{ row.path_display }}</td>
                                    <td class="metric-value">{{ row.value_display or '—' }}</td>
                                    <td>{{ row.stats_display or '—' }}</td>
                                </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                {% else %}
                    <div class="note">No metrics were found in results.yml.</div>
                {% endif %}
            </div>

            <div class="section">
                <h2>Performance Metrics</h2>
                {% if meta.perf_rows %}
                    <table>
                        <thead>
                            <tr>
                                <th>Metric Path</th>
                                <th>Value</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for row in meta.perf_rows %}
                                <tr>
                                    <td class="metric-path" title="{{ row.path }}">{{ row.path_display }}</td>
                                    <td class="metric-value">{{ row.value_display or '—' }}</td>
                                </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                {% else %}
                    <div class="note">No performance metrics were found.</div>
                {% endif %}
            </div>

            <div class="section">
                <h2>Artifacts</h2>
                {% if meta.artifacts_rows %}
                    <table>
                        <thead>
                            <tr>
                                <th>Artifact</th>
                                <th>Details</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for row in meta.artifacts_rows %}
                                <tr>
                                    <td>{{ row.name }}</td>
                                    <td>{{ row.detail }}</td>
                                </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                {% else %}
                    <div class="note">No artifact metadata found.</div>
                {% endif %}

                {% if meta.raw_results %}
                    <details>
                        <summary>Raw results.yml</summary>
                        <pre>{{ meta.raw_results|e }}</pre>
                    </details>
                {% endif %}
                {% if meta.raw_run_config %}
                    <details>
                        <summary>Raw run_config.yml</summary>
                        <pre>{{ meta.raw_run_config|e }}</pre>
                    </details>
                {% endif %}
                {% if meta.raw_eval_metrics %}
                    <details>
                        <summary>Raw eval_factory_metrics.json</summary>
                        <pre>{{ meta.raw_eval_metrics|e }}</pre>
                    </details>
                {% endif %}
            </div>


        {% else %}
            <div class="empty">No cached entries were found to render a report.</div>
        {% endif %}
    </div>

    <script>
        function toggleRaw(elementId) {
            const element = document.getElementById(elementId);
            if (!element) return;
            element.classList.toggle('hidden');
        }

        function copyBlock(elementId) {
            const element = document.getElementById(elementId);
            if (!element) return;
            const text = element.innerText || element.textContent || "";
            navigator.clipboard.writeText(text);
        }

        const toggles = document.querySelectorAll('.toggle');
        toggles.forEach((toggle) => {
            toggle.addEventListener('click', () => {
                toggle.classList.toggle('active');
                const target = toggle.getAttribute('data-target');
                const sections = document.querySelectorAll(`[data-section="${target}"]`);
                sections.forEach((section) => {
                    if (toggle.classList.contains('active')) {
                        section.classList.remove('hidden');
                    } else {
                        section.classList.add('hidden');
                    }
                });
            });
        });

        const searchBox = null;
        const modelFilter = document.getElementById('modelFilter');
        const typeFilter = document.getElementById('typeFilter');
        const finishFilter = document.getElementById('finishFilter');
        const gradingFilter = document.getElementById('gradingFilter');
        const errorOnly = document.getElementById('errorOnly');
        const toolOnly = document.getElementById('toolOnly');
        const sortSelect = document.getElementById('sortSelect');
        const visibleCount = document.getElementById('visibleCount');
        const entriesContainer = document.getElementById('entriesContainer');
        const tableBody = document.querySelector('.sample-table tbody');
        const tabs = document.querySelectorAll('.tab');
        const tabContents = document.querySelectorAll('.tab-content');

        function matchesFilters(element) {
            const query = "";
            const model = modelFilter?.value || "";
            const type = typeFilter?.value || "";
            const finish = finishFilter?.value || "";
            const errorsOnly = errorOnly?.checked || false;
            const toolsOnly = toolOnly?.checked || false;
            const grading = gradingFilter?.value || "";

            const haystack = element.getAttribute('data-search') || "";
            const entryModel = element.getAttribute('data-model') || "";
            const entryType = element.getAttribute('data-request-type') || "";
            const entryFinish = (element.getAttribute('data-finish-reasons') || "").split(',').filter(Boolean);
            const hasError = element.getAttribute('data-has-error') === 'true';
            const hasTool = element.getAttribute('data-has-tool') === 'true';
            const grade = element.getAttribute('data-graded') || "";

            const matchesSearch = !query || haystack.indexOf(query) !== -1;
            const matchesModel = !model || entryModel === model;
            const matchesType = !type || entryType === type;
            const matchesFinish = !finish || entryFinish.includes(finish);
            const matchesError = !errorsOnly || hasError;
            const matchesTool = !toolsOnly || hasTool;
            const matchesGrading = !grading || grade === grading;

            return matchesSearch && matchesModel && matchesType && matchesFinish && matchesError && matchesTool && matchesGrading;
        }

        function applyFilters() {
            const entries = entriesContainer ? entriesContainer.querySelectorAll('.entry') : document.querySelectorAll('.entry');
            const rows = document.querySelectorAll('.sample-row');
            let visibleEntries = 0;
            let visibleRows = 0;
            const total = rows.length || entries.length;

            entries.forEach((entry) => {
                const visible = matchesFilters(entry);
                entry.style.display = visible ? 'block' : 'none';
                if (visible) visibleEntries += 1;
            });

            rows.forEach((row) => {
                const visible = matchesFilters(row);
                row.style.display = visible ? '' : 'none';
                if (visible) visibleRows += 1;
            });

            if (visibleCount) {
                const count = rows.length ? visibleRows : visibleEntries;
                visibleCount.textContent = `Visible: ${count} / ${total}`;
            }
        }

        function normalizeGradeClasses() {
            document.querySelectorAll('.entry').forEach((entry) => {
                const grade = entry.getAttribute('data-graded') || 'unknown';
                entry.classList.remove('grade-correct', 'grade-incorrect', 'grade-unknown');
                entry.classList.add(`grade-${grade || 'unknown'}`);
            });
            document.querySelectorAll('.sample-row').forEach((row) => {
                const grade = row.getAttribute('data-graded') || 'unknown';
                row.classList.remove('grade-correct', 'grade-incorrect', 'grade-unknown');
                row.classList.add(`grade-${grade || 'unknown'}`);
            });
        }

        function syncEntryCounts() {
            if (!entriesContainer || !tableBody) return;
            const rows = Array.from(tableBody.querySelectorAll('.sample-row'));
            const entries = Array.from(entriesContainer.querySelectorAll('.entry'));
            if (!rows.length || !entries.length) return;
            if (entries.length <= rows.length) return;
            entries.forEach((entry, idx) => {
                if (idx >= rows.length) {
                    entry.style.display = 'none';
                }
            });
        }

        function sortEntries() {
            if (!entriesContainer || !sortSelect) return;
            const entries = Array.from(entriesContainer.querySelectorAll('.entry'));
            const sortBy = sortSelect.value;
            entries.sort((a, b) => {
                if (sortBy === 'cache') {
                    return (a.getAttribute('data-cache') || '').localeCompare(
                        b.getAttribute('data-cache') || ''
                    );
                }
                if (sortBy === 'request') {
                    return (parseInt(a.getAttribute('data-request-chars') || '0', 10) -
                        parseInt(b.getAttribute('data-request-chars') || '0', 10));
                }
                if (sortBy === 'response') {
                    return (parseInt(a.getAttribute('data-response-chars') || '0', 10) -
                        parseInt(b.getAttribute('data-response-chars') || '0', 10));
                }
                return 0;
            });
            entries.forEach((entry) => entriesContainer.appendChild(entry));

            if (tableBody) {
                const rows = Array.from(tableBody.querySelectorAll('.sample-row'));
                rows.sort((a, b) => {
                    if (sortBy === 'cache') {
                        return (a.getAttribute('data-cache') || '').localeCompare(
                            b.getAttribute('data-cache') || ''
                        );
                    }
                    if (sortBy === 'request') {
                        return (parseInt(a.getAttribute('data-request-chars') || '0', 10) -
                            parseInt(b.getAttribute('data-request-chars') || '0', 10));
                    }
                    if (sortBy === 'response') {
                        return (parseInt(a.getAttribute('data-response-chars') || '0', 10) -
                            parseInt(b.getAttribute('data-response-chars') || '0', 10));
                    }
                    return 0;
                });
                rows.forEach((row) => tableBody.appendChild(row));
            }
        }

        [modelFilter, typeFilter, finishFilter, gradingFilter, errorOnly, toolOnly].forEach((el) => {
            if (!el) return;
            el.addEventListener('input', applyFilters);
            el.addEventListener('change', applyFilters);
        });

        if (sortSelect) {
            sortSelect.addEventListener('change', () => {
                sortEntries();
                applyFilters();
            });
        }

        tabs.forEach((tab) => {
            tab.addEventListener('click', () => {
                tabs.forEach((btn) => btn.classList.remove('active'));
                tab.classList.add('active');
                const target = tab.getAttribute('data-tab');
                tabContents.forEach((content) => {
                    if (content.id === target) {
                        content.classList.add('active');
                    } else {
                        content.classList.remove('active');
                    }
                });
            });
        });

        function openDetailById(targetId) {
            if (!targetId) return;
            const targetTab = document.querySelector('.tab[data-tab="detailView"]');
            if (targetTab) targetTab.click();
            const el = document.getElementById(targetId);
            if (!el) return;
            el.classList.add('highlight');
            el.scrollIntoView({ behavior: 'smooth', block: 'start' });
            setTimeout(() => el.classList.remove('highlight'), 1600);
        }

        document.querySelectorAll('a[data-detail-link="true"]').forEach((link) => {
            link.addEventListener('click', (event) => {
                event.preventDefault();
                const href = link.getAttribute('href') || '';
                const targetId = href.startsWith('#') ? href.slice(1) : href;
                openDetailById(targetId);
                history.replaceState(null, '', `#${targetId}`);
            });
        });

        if (window.location.hash) {
            const targetId = window.location.hash.slice(1);
            if (targetId.startsWith('sample-')) {
                openDetailById(targetId);
            }
        }

        document.querySelectorAll('.entry').forEach((entry) => {
            const previews = Array.from(entry.querySelectorAll('.prompt-preview'));
            const texts = previews.map((preview) =>
                (preview.textContent || '').replace(/\s+/g, ' ').trim()
            );

            previews.forEach((preview, idx) => {
                const text = texts[idx];
                if (!text) {
                    preview.remove();
                }
            });

            for (let i = 0; i < previews.length; i += 1) {
                if (!previews[i] || !texts[i]) continue;
                for (let j = 0; j < previews.length; j += 1) {
                    if (i === j || !previews[j] || !texts[j]) continue;
                    if (texts[j].includes(texts[i])) {
                        previews[i].remove();
                        texts[i] = '';
                        previews[i] = null;
                        break;
                    }
                }
            }

            const seen = new Set();
            previews.forEach((preview, idx) => {
                if (!preview) return;
                const text = texts[idx];
                if (!text) return;
                if (seen.has(text)) {
                    preview.remove();
                } else {
                    seen.add(text);
                }
            });
        });

        function setupCollapsibles() {
            document.querySelectorAll('.section').forEach((section) => {
                const header = section.querySelector('h2');
                if (!header) return;
                if (!header.classList.contains('section-header')) {
                    header.classList.add('section-header');
                }

                let body = section.querySelector('.section-body');
                if (!body) {
                    body = document.createElement('div');
                    body.className = 'section-body';
                    const siblings = [];
                    let node = header.nextSibling;
                    while (node) {
                        const next = node.nextSibling;
                        siblings.push(node);
                        node = next;
                    }
                    siblings.forEach((node) => body.appendChild(node));
                    section.appendChild(body);
                }

                if (!header.querySelector('.section-toggle')) {
                    const toggle = document.createElement('button');
                    toggle.type = 'button';
                    toggle.className = 'section-toggle';
                    toggle.textContent = 'Collapse';
                    toggle.addEventListener('click', (event) => {
                        event.stopPropagation();
                        section.classList.toggle('collapsed');
                        toggle.textContent = section.classList.contains('collapsed')
                            ? 'Expand'
                            : 'Collapse';
                    });
                    header.appendChild(toggle);
                }

                header.addEventListener('click', (event) => {
                    if (event.target && event.target.closest('.section-toggle')) return;
                    const toggle = header.querySelector('.section-toggle');
                    section.classList.toggle('collapsed');
                    if (toggle) {
                        toggle.textContent = section.classList.contains('collapsed')
                            ? 'Expand'
                            : 'Collapse';
                    }
                });
            });
        }

        setupCollapsibles();
        normalizeGradeClasses();
        syncEntryCounts();
        sortEntries();
        applyFilters();
    </script>
</body>
</html>"""
