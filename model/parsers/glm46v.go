package parsers

import (
	"context"
	"encoding/json"
	"log/slog"
	"strings"
	"unicode"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/logutil"
)

const (
	glm46vCollectingContent glm46vParserState = iota
	glm46vCollectingThinkingContent
	glm46vCollectingToolArgs
	glm46vThinkingDoneEatingWhitespace
	glm46vToolCallDoneEatingWhitespace
)

type glm46vParserState int

const (
	glm46vThinkOpenTag  = "<think>"
	glm46vThinkCloseTag = "</think>"
)

type GLM46VParser struct {
	state              glm46vParserState
	buffer             strings.Builder
	tools              []api.Tool
	toolNames          []string
	hasThinkingSupport bool
	currentToolName    string
	jsonBraceCount     int
	// For accumulating content when tools are available
	pendingContent strings.Builder
	foundToolCall  bool
}

func (p *GLM46VParser) HasToolSupport() bool {
	return true
}

func (p *GLM46VParser) HasThinkingSupport() bool {
	return p.hasThinkingSupport
}

func (p *GLM46VParser) setInitialState(lastMessage *api.Message) {
	prefill := lastMessage != nil && lastMessage.Role == "assistant"
	if !p.HasThinkingSupport() {
		p.state = glm46vCollectingContent
		return
	}

	if prefill && lastMessage.Content != "" {
		p.state = glm46vCollectingContent
		return
	}

	p.state = glm46vCollectingThinkingContent
}

func (p *GLM46VParser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	p.tools = tools
	// Extract tool names for detection
	p.toolNames = make([]string, len(tools))
	for i, tool := range tools {
		p.toolNames[i] = tool.Function.Name
	}
	if thinkValue != nil {
		if v, ok := thinkValue.Value.(bool); ok && v {
			p.hasThinkingSupport = true
		} else if s, ok := thinkValue.Value.(string); ok && s != "" {
			p.hasThinkingSupport = true
		}
	}
	p.setInitialState(lastMessage)
	return tools
}

type glm46vEvent interface {
	isGLM46VEvent()
}

type glm46vEventContent struct {
	content string
}

func (glm46vEventContent) isGLM46VEvent() {}

type glm46vEventThinkingContent struct {
	content string
}

func (glm46vEventThinkingContent) isGLM46VEvent() {}

type glm46vEventToolCall struct {
	name string
	args string
}

func (glm46vEventToolCall) isGLM46VEvent() {}

func (p *GLM46VParser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	p.buffer.WriteString(s)
	events := p.parseEvents()

	var contentSb strings.Builder
	var thinkingSb strings.Builder
	for _, event := range events {
		switch event := event.(type) {
		case glm46vEventToolCall:
			toolCall := api.ToolCall{
				Function: api.ToolCallFunction{
					Name: event.name,
				},
			}
			if event.args != "" {
				// Try to parse as JSON, handling Python-style single quotes
				argsStr := event.args
				// Convert Python-style single quotes to double quotes for JSON
				argsStr = convertPythonToJSON(argsStr)
				if err := json.Unmarshal([]byte(argsStr), &toolCall.Function.Arguments); err != nil {
					slog.Warn("glm46v tool call args parsing failed", "error", err, "args", event.args)
					// Still include the tool call with empty args
				}
			}
			calls = append(calls, toolCall)
			// Mark that we found a tool call - discard any pending content
			p.foundToolCall = true
			p.pendingContent.Reset()
		case glm46vEventThinkingContent:
			thinkingSb.WriteString(event.content)
		case glm46vEventContent:
			// When tools are available, accumulate content instead of emitting immediately
			if p.HasToolSupport() && len(p.toolNames) > 0 && !p.foundToolCall {
				p.pendingContent.WriteString(event.content)
			} else if !p.foundToolCall {
				contentSb.WriteString(event.content)
			}
			// If foundToolCall is true, we discard content
		}
	}

	// If done and no tool call was found yet, do a final check on pending content
	if done && !p.foundToolCall && p.pendingContent.Len() > 0 {
		pendingStr := p.pendingContent.String()
		// Final attempt to extract ReAct tool call from accumulated content
		if toolName, args, _, found := p.extractReActFromPlainText(pendingStr); found {
			slog.Log(context.TODO(), logutil.LevelTrace, "glm46v final ReAct extraction from pending", "toolName", toolName, "args", args)
			toolCall := api.ToolCall{
				Function: api.ToolCallFunction{
					Name: toolName,
				},
			}
			if args != "" {
				argsStr := convertPythonToJSON(args)
				if err := json.Unmarshal([]byte(argsStr), &toolCall.Function.Arguments); err != nil {
					slog.Warn("glm46v tool call args parsing failed", "error", err, "args", args)
				}
			}
			calls = append(calls, toolCall)
			p.foundToolCall = true
			p.pendingContent.Reset()
		} else {
			// No tool call found, emit pending content
			contentSb.WriteString(pendingStr)
			p.pendingContent.Reset()
		}
	}

	return contentSb.String(), thinkingSb.String(), calls, nil
}

func (p *GLM46VParser) parseEvents() []glm46vEvent {
	var all []glm46vEvent

	keepLooping := true
	for keepLooping {
		var events []glm46vEvent
		events, keepLooping = p.eat()
		if len(events) > 0 {
			all = append(all, events...)
		}
	}

	if len(all) > 0 {
		slog.Log(context.TODO(), logutil.LevelTrace, "glm46v events parsed", "events", all, "state", p.state, "buffer", p.buffer.String())
	}

	return all
}

func (p *GLM46VParser) eatLeadingWhitespaceAndTransitionTo(nextState glm46vParserState) ([]glm46vEvent, bool) {
	trimmed := strings.TrimLeftFunc(p.buffer.String(), unicode.IsSpace)
	p.buffer.Reset()
	if trimmed == "" {
		return nil, false
	}
	p.state = nextState
	p.buffer.WriteString(trimmed)
	return nil, true
}

// containsToolName checks if the buffer contains a tool name in various formats:
// - GLM-4 native: function_name\n{json}
// - ReAct format: Action: function_name\nAction Input: {json}
// - ReAct variant: function_name\nAction Input: {json}
// - Markdown code block format: ```\nAction: function_name\nAction Input: {json}\n```
func (p *GLM46VParser) containsToolName(buf string) (string, int, bool) {
	for _, toolName := range p.toolNames {
		// Check ReAct format: "Action: function_name" followed by "Action Input:"
		reactPatterns := []string{
			"\nAction: " + toolName + "\n",
			"\nAction: " + toolName + "\r\n",
			"Action: " + toolName + "\n",
			"Action: " + toolName + "\r\n",
		}
		for _, pattern := range reactPatterns {
			if idx := strings.Index(buf, pattern); idx >= 0 {
				afterAction := buf[idx+len(pattern):]
				if strings.HasPrefix(afterAction, "Action Input:") || strings.HasPrefix(strings.TrimLeft(afterAction, " \t"), "Action Input:") {
					return toolName, idx, true
				}
			}
		}

		// Check ReAct variant: "function_name\nAction Input:"
		variantPatterns := []string{
			"\n" + toolName + "\nAction Input:",
			"\n" + toolName + "\r\nAction Input:",
			toolName + "\nAction Input:",
		}
		for _, pattern := range variantPatterns {
			if idx := strings.Index(buf, pattern); idx >= 0 {
				actualIdx := idx
				if buf[idx] == '\n' {
					actualIdx = idx + 1
				}
				return toolName, actualIdx, true
			}
		}

		// Look for tool name on its own line followed by JSON (GLM-4 native format)
		patterns := []string{"\n" + toolName + "\n{", "\n" + toolName + "\r\n{"}
		for _, pattern := range patterns {
			if idx := strings.Index(buf, pattern); idx >= 0 {
				return toolName, idx + 1, true // +1 to skip the leading newline
			}
		}
		// Also check if starts with tool name followed by newline and JSON
		if strings.HasPrefix(buf, toolName+"\n{") || strings.HasPrefix(buf, toolName+"\r\n{") {
			return toolName, 0, true
		}
	}
	return "", -1, false
}

// extractReActFromPlainText extracts tool call from ReAct format in plain text (outside code blocks)
// Format: Action: function_name\nAction Input: {json}
func (p *GLM46VParser) extractReActFromPlainText(buf string) (toolName string, args string, contentBefore string, found bool) {
	for _, tn := range p.toolNames {
		// Look for "Action: toolname\nAction Input:" pattern
		patterns := []string{
			"\nAction: " + tn + "\nAction Input:",
			"\nAction: " + tn + "\r\nAction Input:",
			"Action: " + tn + "\nAction Input:",
			"Action: " + tn + "\r\nAction Input:",
		}
		for _, pattern := range patterns {
			if idx := strings.Index(buf, pattern); idx >= 0 {
				// Found the pattern - extract the JSON args after "Action Input:"
				argsStart := idx + len(pattern)
				remaining := buf[argsStart:]
				remaining = strings.TrimLeft(remaining, " \t")

				// Find the JSON object - count braces
				if len(remaining) > 0 && remaining[0] == '{' {
					braceCount := 0
					jsonEnd := -1
					for i, c := range remaining {
						if c == '{' {
							braceCount++
						} else if c == '}' {
							braceCount--
							if braceCount == 0 {
								jsonEnd = i + 1
								break
							}
						}
					}
					if jsonEnd > 0 {
						args = strings.TrimSpace(remaining[:jsonEnd])
						// Content before is everything before "Action:"
						if idx > 0 && buf[idx] == '\n' {
							contentBefore = buf[:idx]
						} else {
							contentBefore = buf[:idx]
						}
						return tn, args, contentBefore, true
					}
				}
			}
		}
	}
	return "", "", "", false
}

// extractToolCallFromMarkdown extracts tool call from markdown code block format
// Format: ```\nAction: function_name\nAction Input: {json}\n``` or ```json\n...\n```
func (p *GLM46VParser) extractToolCallFromMarkdown(buf string) (toolName string, args string, contentBefore string, found bool) {
	// Look for code block start
	codeBlockStarts := []string{"```\n", "```json\n", "```\r\n", "```json\r\n"}

	for _, startMarker := range codeBlockStarts {
		if startIdx := strings.Index(buf, startMarker); startIdx >= 0 {
			// Find the closing ``` (can be preceded by newline or not)
			codeStart := startIdx + len(startMarker)
			remaining := buf[codeStart:]

			// Try different closing patterns
			endIdx := -1
			closingLen := 0
			for _, closer := range []string{"\n```", "\r\n```", "```"} {
				if idx := strings.Index(remaining, closer); idx >= 0 {
					if endIdx < 0 || idx < endIdx {
						endIdx = idx
						closingLen = len(closer)
					}
				}
			}

			if endIdx >= 0 {
				// We have a complete code block
				codeContent := remaining[:endIdx]
				contentBefore = buf[:startIdx]

				slog.Log(context.TODO(), logutil.LevelTrace, "glm46v markdown code block found", "codeContent", codeContent, "closingLen", closingLen)

				// Check if it's ReAct format inside code block
				for _, tn := range p.toolNames {
					// Check "Action: toolname\nAction Input: {json}"
					actionPrefix := "Action: " + tn + "\n"
					if strings.HasPrefix(codeContent, actionPrefix) {
						rest := codeContent[len(actionPrefix):]
						if strings.HasPrefix(rest, "Action Input:") {
							args = strings.TrimPrefix(rest, "Action Input:")
							args = strings.TrimSpace(args)
							return tn, args, contentBefore, true
						}
					}
					// Check "Action: toolname\r\nAction Input: {json}"
					actionPrefixCR := "Action: " + tn + "\r\n"
					if strings.HasPrefix(codeContent, actionPrefixCR) {
						rest := codeContent[len(actionPrefixCR):]
						if strings.HasPrefix(rest, "Action Input:") {
							args = strings.TrimPrefix(rest, "Action Input:")
							args = strings.TrimSpace(args)
							return tn, args, contentBefore, true
						}
					}
					// Check without Action: prefix - just "toolname\nAction Input:"
					if strings.HasPrefix(codeContent, tn+"\n") {
						rest := codeContent[len(tn)+1:]
						if strings.HasPrefix(rest, "Action Input:") {
							args = strings.TrimPrefix(rest, "Action Input:")
							args = strings.TrimSpace(args)
							return tn, args, contentBefore, true
						}
					}
				}
			}
		}
	}
	return "", "", "", false
}

func (p *GLM46VParser) eat() ([]glm46vEvent, bool) {
	var events []glm46vEvent
	buf := p.buffer.String()

	switch p.state {
	case glm46vCollectingContent:
		// When not in thinking mode, strip out any <think>...</think> tags the model might generate
		if !p.hasThinkingSupport {
			// Check for complete <think>...</think> pattern and remove it
			if strings.Contains(buf, glm46vThinkOpenTag) && strings.Contains(buf, glm46vThinkCloseTag) {
				startIdx := strings.Index(buf, glm46vThinkOpenTag)
				endIdx := strings.Index(buf, glm46vThinkCloseTag) + len(glm46vThinkCloseTag)
				if startIdx < endIdx {
					before := buf[:startIdx]
					trimmedBefore := strings.TrimSpace(before)
					if len(trimmedBefore) > 0 {
						events = append(events, glm46vEventContent{content: trimmedBefore})
					}
					after := strings.TrimLeft(buf[endIdx:], " \t\r\n")
					p.buffer.Reset()
					p.buffer.WriteString(after)
					return events, len(after) > 0
				}
			} else if strings.Contains(buf, glm46vThinkOpenTag) {
				// Have <think> but no </think> yet - wait for more
				return events, false
			}
		}

		// Check for tool call in XML format: <tool_call>name\n<arg_key>k</arg_key>\n<arg_value>v</arg_value>\n</tool_call>
		if p.HasToolSupport() {
			if strings.Contains(buf, "<tool_call>") {
				startIdx := strings.Index(buf, "<tool_call>")
				endIdx := strings.Index(buf, "</tool_call>")

				if endIdx > 0 {
					endIdx += len("</tool_call>")
					toolCallContent := buf[startIdx:endIdx]

					// Extract tool name and args from the XML-like block
					name, args := p.extractXMLToolCall(toolCallContent)
					if name != "" {
						events = append(events, glm46vEventToolCall{
							name: name,
							args: args,
						})
					}

					remaining := buf[endIdx:]
					p.buffer.Reset()
					p.buffer.WriteString(remaining)
					p.state = glm46vToolCallDoneEatingWhitespace
					return events, len(strings.TrimSpace(remaining)) > 0
				} else {
					// Incomplete tool call, but emit content before it if any
					if startIdx > 0 {
						unambiguous := buf[:startIdx]
						p.buffer.Reset()
						p.buffer.WriteString(buf[startIdx:])
						events = append(events, glm46vEventContent{content: unambiguous})
						return events, true
					}
					return events, false
				}
			}
		}

		// No tool call found, emit content normally
		whitespaceLen := trailingWhitespaceLen(buf)
		ambiguousStart := len(buf) - whitespaceLen

		// If we see a partial <tool_call tag, wait
		if strings.Contains(buf, "<tool") {
			ambiguousStart = strings.Index(buf, "<tool")
		}

		unambiguous := buf[:ambiguousStart]
		ambiguous := buf[ambiguousStart:]
		p.buffer.Reset()
		p.buffer.WriteString(ambiguous)
		if len(unambiguous) > 0 {
			events = append(events, glm46vEventContent{content: unambiguous})
		}
		return events, false

	case glm46vCollectingToolArgs:
		// Not used anymore as we parse the whole <tool_call> block at once
		p.state = glm46vCollectingContent
		return nil, true

	case glm46vCollectingThinkingContent:
		if strings.Contains(buf, glm46vThinkCloseTag) {
			thinking, remaining := splitAtTag(&p.buffer, glm46vThinkCloseTag, true)
			if len(thinking) > 0 {
				events = append(events, glm46vEventThinkingContent{content: thinking})
			}
			if remaining == "" {
				p.state = glm46vThinkingDoneEatingWhitespace
			} else {
				p.state = glm46vCollectingContent
			}
			return events, true
		} else if overlapLen := overlap(buf, glm46vThinkCloseTag); overlapLen > 0 {
			beforePartialTag := buf[:len(buf)-overlapLen]
			trailingLen := trailingWhitespaceLen(beforePartialTag)
			unambiguous := beforePartialTag[:len(beforePartialTag)-trailingLen]
			ambiguous := buf[len(unambiguous):]
			p.buffer.Reset()
			p.buffer.WriteString(ambiguous)
			if len(unambiguous) > 0 {
				events = append(events, glm46vEventThinkingContent{content: unambiguous})
			}
			return events, false
		} else {
			whitespaceLen := trailingWhitespaceLen(buf)
			unambiguous := buf[:len(buf)-whitespaceLen]
			ambiguous := buf[len(unambiguous):]
			p.buffer.Reset()
			p.buffer.WriteString(ambiguous)
			if len(unambiguous) > 0 {
				events = append(events, glm46vEventThinkingContent{content: unambiguous})
			}
			return events, false
		}

	case glm46vThinkingDoneEatingWhitespace:
		return p.eatLeadingWhitespaceAndTransitionTo(glm46vCollectingContent)

	case glm46vToolCallDoneEatingWhitespace:
		return p.eatLeadingWhitespaceAndTransitionTo(glm46vCollectingContent)

	default:
		panic("unreachable")
	}
}

func (p *GLM46VParser) extractXMLToolCall(xml string) (string, string) {
	// xml looks like: <tool_call>name\n<arg_key>k</arg_key>\n<arg_value>v</arg_value>\n</tool_call>
	name := ""
	argsMap := make(map[string]interface{})

	// Get name
	startName := strings.Index(xml, "<tool_call>") + len("<tool_call>")
	endName := strings.Index(xml, "\n")
	if endName < 0 || endName < startName {
		// Try to find the first <arg_key> if no newline
		endName = strings.Index(xml, "<arg_key>")
	}

	if endName > startName {
		name = strings.TrimSpace(xml[startName:endName])
	}

	// Extract key/value pairs
	tempXML := xml
	for {
		keyIdx := strings.Index(tempXML, "<arg_key>")
		if keyIdx < 0 {
			break
		}
		keyEndIdx := strings.Index(tempXML, "</arg_key>")
		if keyEndIdx < 0 {
			break
		}
		key := strings.TrimSpace(tempXML[keyIdx+len("<arg_key>") : keyEndIdx])

		valIdx := strings.Index(tempXML, "<arg_value>")
		if valIdx < 0 {
			break
		}
		valEndIdx := strings.Index(tempXML, "</arg_value>")
		if valEndIdx < 0 {
			break
		}
		valStr := strings.TrimSpace(tempXML[valIdx+len("<arg_value>") : valEndIdx])

		// Attempt to parse valStr as JSON if it looks like it
		var val interface{}
		if (strings.HasPrefix(valStr, "{") && strings.HasSuffix(valStr, "}")) ||
			(strings.HasPrefix(valStr, "[") && strings.HasSuffix(valStr, "]")) {
			valStr = convertPythonToJSON(valStr)
			if err := json.Unmarshal([]byte(valStr), &val); err != nil {
				val = valStr
			}
		} else {
			val = valStr
		}

		argsMap[key] = val
		tempXML = tempXML[valEndIdx+len("</arg_value>"):]
	}

	argsJSON, _ := json.Marshal(argsMap)
	return name, string(argsJSON)
}

// convertPythonToJSON converts Python-style dict strings to valid JSON
// Handles: {'key': 'value'} -> {"key": "value"}
func convertPythonToJSON(s string) string {
	// Simple conversion: replace single quotes with double quotes
	// This is a basic approach - a more robust solution would use a proper parser
	result := strings.Builder{}
	inString := false
	stringChar := byte(0)

	for i := 0; i < len(s); i++ {
		c := s[i]

		if !inString {
			if c == '\'' {
				result.WriteByte('"')
				inString = true
				stringChar = '\''
			} else if c == '"' {
				result.WriteByte('"')
				inString = true
				stringChar = '"'
			} else {
				result.WriteByte(c)
			}
		} else {
			if c == stringChar && (i == 0 || s[i-1] != '\\') {
				result.WriteByte('"')
				inString = false
				stringChar = 0
			} else if c == '"' && stringChar == '\'' {
				// Escape double quotes inside single-quoted strings
				result.WriteString("\\\"")
			} else {
				result.WriteByte(c)
			}
		}
	}

	return result.String()
}
