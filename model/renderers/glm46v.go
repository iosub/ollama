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
		// CRITICAL: Tool instructions MUST come FIRST with highest priority
		sb.WriteString("<<MANDATORY TOOL USAGE PROTOCOL>>\n\n")
		sb.WriteString("You are operating in TOOL-ASSISTED MODE. This means:\n")
		sb.WriteString("1. You MUST call the available tools to validate data BEFORE generating any final output\n")
		sb.WriteString("2. You CANNOT skip the tool calls - they are REQUIRED\n")
		sb.WriteString("3. Only after receiving all tool responses can you produce the final result\n\n")

		sb.WriteString("# Available Tools\n\n")
		for _, tool := range tools {
			sb.WriteString("## " + tool.Function.Name + "\n")
			if b, err := marshalWithSpaces(tool.Function); err == nil {
				sb.Write(b)
			}
			sb.WriteString("\n")
		}
		sb.WriteString("\n# REQUIRED Tool Call Format\n\n")
		sb.WriteString("To call a tool, output EXACTLY this format in a code block:\n\n")
		sb.WriteString("```\nAction: <tool_name>\nAction Input: {\"param\": \"value\"}\n```\n\n")
		sb.WriteString("Example:\n```\nAction: dbCheckData\nAction Input: {\"fieldName\": \"Items[0].albaran\", \"value\": \"C25-22564\"}\n```\n\n")
		sb.WriteString("<<END MANDATORY PROTOCOL>>\n\n")

		// User's system prompt comes AFTER tool instructions
		if len(messages) > 0 && messages[0].Role == "system" {
			sb.WriteString("# Additional Context\n" + messages[0].Content)
		}
	} else if len(messages) > 0 && messages[0].Role == "system" {
		sb.WriteString("<|system|>\n" + messages[0].Content)
	}

	// Find last user message index for thinking logic
	lastQueryIndex := len(messages) - 1
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" {
			content := r.renderContent(messages[i])
			if !(strings.HasPrefix(content, "<tool_response>") && strings.HasSuffix(content, "</tool_response>")) {
				lastQueryIndex = i
				break
			}
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

			// Handle thinking content - only if thinking is enabled
			if r.enableThinking {
				if i > lastQueryIndex {
					thinkContent := ""
					if message.Thinking != "" {
						thinkContent = message.Thinking
					}
					if thinkContent != "" {
						sb.WriteString("\n<think>" + strings.TrimSpace(thinkContent) + "</think>")
					} else if !lastMessage {
						// Empty think tags for non-last messages in conversation history
						sb.WriteString("\n<think></think>")
					}
					// For lastMessage, model will generate thinking
				} else {
					sb.WriteString("\n<think></think>")
				}
			}
			// When thinking is disabled, no <think> tags are added

			if content != "" {
				sb.WriteString("\n" + strings.TrimSpace(content))
			}

			// Handle tool calls - Use ReAct format inside markdown block to match instructions
			if len(message.ToolCalls) > 0 {
				sb.WriteString("\n```")
				for _, toolCall := range message.ToolCalls {
					sb.WriteString("\nAction: " + toolCall.Function.Name + "\nAction Input: ")
					if b, err := marshalWithSpaces(toolCall.Function.Arguments); err == nil {
						sb.Write(b)
					}
				}
				sb.WriteString("\n```")
			}
		} else if message.Role == "tool" {
			// Tool responses use <|observation|> tag - GLM-4 native format
			sb.WriteString("<|observation|>\n" + message.Content)
		}

		// Add generation prompt for last message
		if lastMessage && !prefill {
			sb.WriteString("<|assistant|>")
			// No <think> tags when thinking is disabled - model responds directly
		}
	}

	return sb.String(), nil
}
