package renderers

import (
	"strings"

	"github.com/ollama/ollama/api"
)

// GLM46VRenderer handles GLM-4.6V and GLM-4.6V-Flash models
// These models support vision with <|begin_of_image|><|image|><|end_of_image|> tokens
// Tool calling uses JSON format similar to Qwen3VL
type GLM46VRenderer struct {
	enableThinking bool
	useImgTags     bool
}

func (r *GLM46VRenderer) renderContent(content api.Message) string {
	var subSb strings.Builder
	for range content.Images {
		if r.useImgTags {
			subSb.WriteString("[img]")
		} else {
			subSb.WriteString("<|begin_of_image|><|image|><|end_of_image|>")
		}
	}
	// TODO: support videos with <|begin_of_video|><|video|><|end_of_video|>

	subSb.WriteString(content.Content)
	return subSb.String()
}

func (r *GLM46VRenderer) Render(messages []api.Message, tools []api.Tool, _ *api.ThinkValue) (string, error) {
	var sb strings.Builder

	// GLM-4.6V uses [gMASK]<sop> as prefix
	sb.WriteString("[gMASK]<sop>")

	// Handle system message with tools - use GLM-4 native format
	if len(tools) > 0 {
		sb.WriteString("<|system|>\n")
		sb.WriteString("# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n")
		for _, tool := range tools {
			if b, err := marshalWithSpaces(tool.Function); err == nil {
				sb.Write(b)
				sb.WriteString("\n")
			}
		}
		sb.WriteString("</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n<tool_call>{function-name}\n<arg_key>{arg-key-1}</arg_key>\n<arg_value>{arg-value-1}</arg_value>\n<arg_key>{arg-key-2}</arg_key>\n<arg_value>{arg-value-2}</arg_value>\n...\n</tool_call>")

		// User's system prompt comes AFTER tool instructions
		if len(messages) > 0 && messages[0].Role == "system" {
			sb.WriteString("\n\n" + messages[0].Content)
		}
	} else if len(messages) > 0 && messages[0].Role == "system" {
		sb.WriteString("<|system|>\n" + messages[0].Content)
	}

	// Find last user message index for thinking logic
	lastUserIndex := -1
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" {
			lastUserIndex = i
			break
		}
	}

	for i, message := range messages {
		content := r.renderContent(message)
		lastMessage := i == len(messages)-1
		prefill := lastMessage && message.Role == "assistant"

		if message.Role == "system" && i == 0 {
			// Already handled above
			continue
		}

		if message.Role == "user" {
			sb.WriteString("<|user|>\n" + content)
			// Add /nothink if thinking is disabled
			if !r.enableThinking && !strings.HasSuffix(content, "/nothink") {
				sb.WriteString("/nothink")
			}
		} else if message.Role == "system" {
			sb.WriteString("<|system|>\n" + content)
		} else if message.Role == "assistant" {
			sb.WriteString("<|assistant|>")

			// Reasoning logic: if reasoning_content is present, or if it's after last user message
			reasoningContent := strings.TrimSpace(message.Thinking)
			if i > lastUserIndex && reasoningContent != "" {
				sb.WriteString("\n<think>" + reasoningContent + "</think>")
			} else {
				sb.WriteString("\n<think></think>")
			}

			if trimmed := strings.TrimSpace(content); trimmed != "" {
				sb.WriteString("\n" + trimmed)
			}

			// Handle tool calls - Use XML format from zitemplate.md
			if len(message.ToolCalls) > 0 {
				for _, toolCall := range message.ToolCalls {
					sb.WriteString("\n<tool_call>" + toolCall.Function.Name)
					for k, v := range toolCall.Function.Arguments {
						sb.WriteString("\n<arg_key>" + k + "</arg_key>")
						sb.WriteString("\n<arg_value>")
						if s, ok := v.(string); ok {
							sb.WriteString(s)
						} else if b, err := marshalWithSpaces(v); err == nil {
							sb.Write(b)
						}
						sb.WriteString("</arg_value>")
					}
					sb.WriteString("\n</tool_call>")
				}
			}
		} else if message.Role == "tool" {
			// Tool responses use <|observation|> tag once for a block of tool responses
			if i == 0 || messages[i-1].Role != "tool" {
				sb.WriteString("<|observation|>")
			}
			sb.WriteString("\n<tool_response>\n" + message.Content + "\n</tool_response>")
		}

		// Add generation prompt for last message
		if lastMessage && !prefill {
			sb.WriteString("<|assistant|>")
			if !r.enableThinking {
				sb.WriteString("\n<think></think>\n")
			}
		}
	}

	// print(sb.String()) // For debugging
	return sb.String(), nil
}
