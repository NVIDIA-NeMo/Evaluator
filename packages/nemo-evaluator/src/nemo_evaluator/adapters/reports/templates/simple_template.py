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
    <title>{% if entries %}Example Request-Response Pairs{% else %}Example Request-Response Pair{% endif %}</title>
    <style>
        /* Container and general styles */
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
            margin-bottom: 10px;
        }
        .subtitle {
            color: #666;
            font-style: italic;
            margin-bottom: 20px;
            font-size: 1.1em;
        }
        .note {
            color: #546E7A;
            font-weight: bold;
            font-style: normal;
        }
        .cache-info-header {
            background-color: #E8F5E9;
            color: #2E7D32;
            padding: 12px 16px;
            border-radius: 4px;
            margin-bottom: 20px;
            font-size: 0.95em;
            border: 1px solid #C8E6C9;
        }
        .cache-info-header .note {
            color: #546E7A;
            font-weight: bold;
            font-style: normal;
            display: block;
            margin-top: 8px;
        }

        /* Entry styles */
        .entry {
            margin: 20px 0;
            padding: 15px;
            background: #f9f9f9;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .cache-info {
            padding: 8px 16px;
            margin-bottom: 8px;
            border-radius: 4px;
            font-family: monospace;
            background-color: #E3F2FD;
            color: #0D47A1;
            font-size: 0.9em;
        }
        .cache-explanation {
            font-size: 0.8em;
            color: #666;
            margin-top: 4px;
            font-style: italic;
        }
        .request, .response, .curl-command {
            padding: 8px 16px;
            margin-bottom: 8px;
            border-radius: 4px;
            font-family: monospace;
        }
        .request {
            background-color: #B2DFDB;
            color: #00695C;
        }
        .response {
            background-color: #B39DDB;
            color: #4527A0;
        }
        .curl-command {
            background-color: #EEEEEE;
            color: #212121;
            white-space: pre-wrap;
            display: none;
        }
        pre {
            white-space: pre-wrap;
            margin: 0;
            background: #f5f5f5;
            padding: 10px;
            border-radius: 3px;
            overflow-x: auto;
        }
        .buttons {
            display: flex;
            gap: 10px;
            margin-bottom: 10px;
        }
        .button {
            padding: 5px 10px;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 12px;
            background-color: #4CAF50;
            color: white;
        }
        .button:hover {
            background-color: #45a049;
        }
    </style>
</head>
<body>
    <div class="container">
        {% if entries %}
            <h1>Example Request-Response Pairs</h1>
            <p class="subtitle">Final request (sent directly to the endpoint) and response pairs with curl commands for reproducibility. <span class="note">*Note: Responses may vary due to generation parameters, randomness, and other factors.</span></p>
            <div class="cache-info-header">
                <strong>About Cache Keys:</strong> Each request-response pair is identified by a unique cache key generated from the request payload hash. <span class="note">Note: Cache keys may differ across different models because the model field in the request payload will be different.</span>
            </div>
            {% for entry in entries %}
                <div class="entry">
                    <div class="cache-info">
                        <strong>Cache Key:</strong> {{ entry.cache_key }}
                    </div>
                    <div class="request">
                        <strong>Request:</strong>
                        <pre>{{ entry.display_request|tojson_utf8|safe }}</pre>
                    </div>

                    <div class="response">
                        <strong>Response:</strong>
                        <pre>{{ entry.response|tojson_utf8|safe }}</pre>
                    </div>

                    <div class="buttons">
                        <button class="button" onclick="toggleRaw('curl-{{ entry.cache_key }}')">Show Curl Command</button>
                    </div>

                    <div id="curl-{{ entry.cache_key }}" class="curl-command">
                        # Save payload to file first:
                        echo '{{ entry.request_data|tojson }}' > request.json

                        curl "{{ entry.endpoint }}" \
                        -H "Content-Type: application/json" \
                        -H "Authorization: Bearer $API_KEY" \
                        -H "Accept: application/json" \
                        -d @request.json
                    </div>
                </div>
            {% endfor %}
        {% else %}
            <h1>Example Request-Response Pair</h1>
            <p class="subtitle">Final request (sent directly to the endpoint) and response with curl command for reproducibility. <span class="note">*Note: Responses may vary due to generation parameters, randomness, and other factors.</span></p>
            <div class="cache-info-header">
                <strong>About Cache Keys:</strong> Each request-response pair is identified by a unique cache key generated from the request payload hash. <span class="note">Note: Cache keys may differ across different models because the model field in the request payload will be different.</span>
            </div>
            <div class="entry">
                <div class="cache-info">
                    <strong>Cache Key:</strong> {{ cache_key }}
                </div>
                <div class="request">
                    <strong>Request:</strong>
                    <pre>{{ display_request|tojson_utf8|safe }}</pre>
                </div>

                <div class="response">
                    <strong>Response:</strong>
                    <pre>{{ response|tojson_utf8|safe }}</pre>
                </div>

                <div class="buttons">
                    <button class="button" onclick="toggleRaw('curl-{{ cache_key }}')">Show Curl Command</button>
                </div>

                <div id="curl-{{ cache_key }}" class="curl-command">
                        # Save payload to file first:
                        echo '{{ request_data|tojson }}' > request.json

                        curl "{{ endpoint }}" \
                        -H "Content-Type: application/json" \
                        -H "Authorization: Bearer $API_KEY" \
                        -H "Accept: application/json" \
                        -d @request.json
                </div>
            </div>
        {% endif %}
    </div>

    <script>
        function toggleRaw(elementId) {
            const element = document.getElementById(elementId);
            if (element.style.display === 'none' || element.style.display === '') {
                element.style.display = 'block';
            } else {
                element.style.display = 'none';
            }
        }
    </script>
</body>
</html>"""
