(deployment-testing-compatibility)=
# Testing Endpoint Compatibility

This guide helps you test your hosted endpoint to verify OpenAI-compatible API compatibility using `curl` requests for different task types. Models deployed using `nemo-evaluator-launcher` should be compatible with these tests.

To test your endpoint run the provided command and check the model's response. Make sure to populate
`FULL_ENDPOINT_URL` and  `API_KEY` and replace `<YOUR_MODEL_NAME>` with your own values.

# Chat endpoint testing


:::{tip}
If you model is not gated, skip the line with authorization header:

```bash
-H "Authorization: Bearer ${API_KEY}"
```
from the commands below.
:::

## General Requirements

Your endpoint should support the following parameters:
- `top_p`
- `temperature`
- `max_tokens`


## Chat endpoint testing

```bash
export FULL_ENDPOINT_URL="https://your-server.com/v1/chat/completions"
export API_KEY="your-api-key-here"

curl -X POST ${FULL_ENDPOINT_URL} \
-H "Content-Type: application/json" \
-H "Authorization: Bearer ${API_KEY}" \
-d '{
  "messages": [
    {
      "role": "user",
      "content": "Write Python code that can add a list of numbers together."
    }
  ],
  "model": "<YOUR_MODEL_NAME>",
  "temperature": 0.6,
  "top_p": 0.95,
  "max_tokens": 256,
  "stream": false
}'
```

## Completions endpoint testing

```bash
export FULL_ENDPOINT_URL="https://your-server.com/v1/completions"
export API_KEY="your-api-key-here"

curl -X POST ${FULL_ENDPOINT_URL} \
-H "Content-Type: application/json" \
-H "Authorization: Bearer ${API_KEY}" \
-d '{
  "prompt": "Write Python code that can add a list of numbers together.",
  "model": "<YOUR_MODEL_NAME>",
  "temperature": 0.6,
  "top_p": 0.95,
  "max_tokens": 256,
  "stream": false
}'
```

## VLM chat endpoint testing

NeMo Evaluator supports the **OpenAI Images API** ([docs](https://platform.openai.com/docs/guides/images-vision#giving-a-model-images-as-input)) and **vLLM** ([docs](https://docs.vllm.ai/en/stable/features/multimodal_inputs.html)) with the image provided as **base64-encoded image**, and the following content types:

- `image_url`
- `text`

```bash
export FULL_ENDPOINT_URL="https://your-server.com/v1/chat/completions"
export API_KEY="your-api-key-here"

curl -X POST ${FULL_ENDPOINT_URL} \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Accept: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "image_url",
            "image_url": {
              "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAAQABADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD3+iiigAooooA//9k="
            }
          },
          {
            "type": "text",
            "text": "Describe the image:"
          }
        ]
      }
    ],
    "model": "<YOUR_MODEL_NAME>",
    "stream": false,
    "max_tokens": 16,
        "temperature": 0.0,
    "top_p": 1.0
}'
```

## Function calling testing

We support OpenAI-compatible function calling ([docs](https://platform.openai.com/docs/guides/function-calling?api-mode=responses)):

Function calling request:

```bash
export FULL_ENDPOINT_URL="https://your-server.com/v1/chat/completions"
export API_KEY="your-api-key-here"

curl -X POST ${FULL_ENDPOINT_URL} \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Accept: application/json" \
  -d '{
    "model": "<YOUR_MODEL_NAME>",
    "stream": false,
    "max_tokens": 16,
    "temperature": 0.0,
    "top_p": 1.0,
    "messages": [
      {
        "role": "user",
        "content": "What is the slope of the line which is perpendicular to the line with the equation y = 3x + 2?"
      }
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "find_critical_points",
          "description": "Finds the critical points of the function. Note that the provided function is in Python 3 syntax.",
          "parameters": {
            "type": "object",
            "properties": {
              "function": {
                "type": "string",
                "description": "The function to find the critical points for."
              },
              "variable": {
                "type": "string",
                "description": "The variable in the function."
              },
              "range": {
                "type": "array",
                "items": {
                  "type": "number"
                },
                "description": "The range to consider for finding critical points. Optional. Default is [0.0, 3.4]."
              }
            },
            "required": ["function", "variable"]
          }
        }
      }
    ]
  }'

```

# Audio endpoint testing

We support audio input with the following content types:

- `audio_url`

Example:

``` bash
export FULL_ENDPOINT_URL="https://your-server.com/v1/chat/completions"
export API_KEY="your-api-key-here"

curl -X POST ${FULL_ENDPOINT_URL} \
-H "Content-Type: application/json" \
-H "Authorization: Bearer ${API_KEY}" \
  -H "Accept: application/json" \
  -d '{
    "max_tokens": 256,
    "model": "<YOUR_MODEL_NAME>",
    "messages": [
        {
            "content": [
                {
                    "audio_url": {
                        "url": "data:audio/wav;base64,UklGRqQlAABXQVZFZm10IBAAAAABAAEAgLsAAAB3AQACABAAZGF0YYAlAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD/////AAAAAAAAAAAAAAAAAAAAAAAA/////wAAAAD/////AAAAAAAAAAAAAAAAAAAAAP//////////AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA////////////////////////////////////////////////////////AAD/////////////////////AAD//wAAAAAAAAAA//////////////////8AAAAAAAAAAAAA/////wAAAAD/////AAAAAAAAAAAAAAAAAAAAAAAA////////AAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//////////AAAAAAAAAAAAAAAAAAD/////////////AAAAAAAA////////////////////////////////////////////////////////AAAAAAAAAAD/////////////AAD//wAAAAAAAAAA//////////////////8AAAAAAAAAAAAA/////wAAAAD/////AAAAAAAAAAAAAAAAAAAAAAAA/////wAA////////AAAAAAAAAAAAAP//////////////////AAAAAAAAAAD///////////////////////8AAAAAAAD/////////////////////AAAAAP//////////////////////////AAAAAAAAAAAAAAAA/////wAAAAAAAAAAAAAAAAAA//////////8AAAAAAAAAAAAAAAAAAAAAAAD//wAA////////AAAAAAAAAAAAAAAAAAAAAP////8AAAAA////////AAAAAAAAAAAAAP//////////////////AAAAAAAAAAD//wAAAAD///////8AAAAAAAAAAAAA//////////////////////////8AAAAA/////////////////////wAAAAAAAP///////////////////////wAAAAAAAAAA////////////////AAAAAAAAAAAAAP//////////AAAAAP////8AAAAA////////AAAAAAAA/////wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/////wAAAAAAAAAAAAAAAAAAAAAAAAAAAAD///////8AAAAAAAAAAAAAAAD///////8AAP////8AAAAAAAAAAAAAAAAAAAAA/////wAAAAD/////AAAAAAAAAAAAAAAAAAAAAAAAAAAAAP////8AAAAA////////AAAAAAAAAAAAAAAAAAAAAAAAAAD/////////////////////AAAAAP//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA//8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD///////8AAAAA/////wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD/////////////////////AAAAAAAAAAAAAAAA//////////////////////////////////8AAAAAAAAAAAAAAAAAAP//AAAAAAAAAAAAAAAAAAAAAP///////wAAAAAAAAAAAAAAAAAAAAD///////8AAAAAAAD//////////////////////////////////////////////////////////wAA//8AAAAAAAAAAP///////wAAAAAAAAAAAAAAAAAAAAD///////////////////////////////////////8AAAAAAAAAAAAAAAAAAP///////wAAAAAAAAAA//8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//AAAAAAAAAAAAAAAA//////////8AAAAAAAAAAP//////////////////////////AAAAAAAAAAAAAAAAAAAAAP////////////////////8AAAAA//////////////////////////////////////////8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP///////////////////////////////wAAAAAAAAAAAAAAAP//////////AAAAAAAAAAAAAAAAAAD///////8AAAAAAAD/////////////////////////////////////////////////////////////////////AAAAAP////////////////////8AAAAAAAD/////AAAAAAAAAAAAAAAAAAD/////AAAAAP///////////////////////////////wAAAAD///////////////////////////////////////8AAP///////wAAAAD/////////////////////////////////////////////////////AAAAAP//////////////////////////AAAAAAAAAAAAAAAAAAAAAP//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAP/////+////////////AAAAAAAA//8AAAAAAAAAAP////8AAP//////////////////AAAAAAAAAAAAAAAAAAAAAP//AAAAAP///////wAAAAD/////AAAAAAAA/////wAA//////////8AAAAA//8AAAAA/////////////////////wAAAAAAAAAAAAAAAP//////////AAAAAAAAAAAAAAAAAAD//////////wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA//8AAAAAAAAAAP///////wAAAAAAAAAAAAAAAAAAAAD//////////wAAAAAAAP//AAAAAP/////////////////////////////////////////////////////////////////////+//7//v//////////////////////////////////////AAD/////////////////////////////////////////////////////AAAAAAAA//8AAAAA/////////v/+//7//v//////AAAAAAAAAAAAAAAAAAD//////v/+//7//v/+//3//f/+///////+//7//f/+//7//f/8//3//f/+//7///8AAAAAAAAAAAAAAQAAAP7//f/+//7//f/8//z//f/9//z/+//8//z//P/7//v/+//7//v/+v/6//r/+v/7//z//f/9//z//P/8//z/+//5//n/+P/4//j/+P/5//r/+v/6//r/+//8//z/+v/5//n/+f/4//f/9v/1//X/9v/4//n/+f/4//n/+v/7//v/+f/4//f/9v/2//f/+f/6//n/+P/4//n/+f/5//j/+P/5//n/+v/6//n/+P/4//n/+//7//r/+P/4//n/+f/4//b/9f/2//j/+P/4//f/9//3//b/9P/z//P/9P/0//T/9f/1//b/9v/2//f/+P/3//b/9v/2//b/9v/1//X/9v/4//n/+f/5//n/+P/3//f/9//2//X/9f/2//f/9//5//z///8AAAAAAAD///7//P/8//7///8AAAEAAgAFAAUAAwABAAIAAwACAP///f/9////AQADAAQABgAHAAcABgAFAAQAAgACAAMABAAEAAQABQAJAAsADAALAAsACgAJAAYABAADAAIAAQAAAP///v/9//7/AQAEAAUAAgD/////AAABAAAAAAAAAAEAAQAAAAAAAgAEAAUAAwABAAAAAQABAAEAAAD//wAAAQABAP7//P/6//r/+v/6//n/9//2//f/+v/9//3/+//6//7/AgACAP///v8CAAYABgACAAAAAwAFAAEA/P/6//7/AAD+//z//f8BAAMAAgAAAAIABAADAAAA/v8AAAMAAwABAP///v8AAAIAAQD+//3//f/9//3/+//7//z//f/6//j/+P/6//z/+f/1//b/+f/7//f/9P/0//j/+//6//j/+P/5//z//P/8//v/+f/4//j/+v/8//z/+//5//j/+f/8//3//P/6//f/9v/1//P/8f/v/+//8f/0//X/9v/2//f//P8BAAMAAAD8//z/AQAEAAIA+//4//v///////3//f8AAAMAAQD9//v//v8DAAUABQAFAAcACQAMAAwACwAJAAgABwAIAAgABwAFAAUABgAHAAcABwAHAAgACwANAA8AEAAPAA8AEAASABIADQAIAAcADAAPAA4ACgAJAA0ADgAJAAQABAALABAADgAJAAkADwAVABYAEwAPAA4ADwAOAAwACAAEAAEA//////7//v/+/wEABAAHAAcABgAEAAQABQAIAAkABwAFAAMAAwAEAAUABwAIAAYAAwADAAYACQAHAAEA/f8BAAUABQD///z//v8BAAAA+//5//r//P/7//n/+v/7//v/+f/3//f/+v/9//z/+f/4//v/AAACAP//+//8/wAAAwABAAAAAgAHAAYAAQD9/wAAAwD///j/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"
                    },
                    "type": "audio_url"
                },
                {
                    "text": "Please recognize the speech and only output the recognized content:",
                    "type": "text"
                }
            ],
            "role": "user"
        }
    ],
    "temperature": 0.0,
    "top_p": 1.0
}'
```

## Next Steps
- **Run your first evaluation**: Choose your path with {ref}`gs-quickstart`
- **Select benchmarks**: Explore available evaluation tasks
