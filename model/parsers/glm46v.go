package parsers

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
	"unicode"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/logutil"
)

// GLM46VParser handles GLM-4-6V's tool call formats:
// - ReAct format: Action: function_name\nAction Input: {json}
// - Markdown format: ```\nAction: function_name\nAction Input: {json}\n```
// - Native format: function_name\n{json}
//
// And thinking format:
// <think>thinking content</think>
type GLM46VParser struct {
	state  GLM46VParserState
	buffer strings.Builder
	tools  []api.Tool
	err    error // Store critical errors (like unknown tools)
}

type GLM46VParserState int

const (
	GLM46VCollectingContent GLM46VParserState = iota
	GLM46VCollectingThinking
	GLM46VCollectingToolCall
)

const (
	glm46vThinkingOpenTag   = "<think>"
	glm46vThinkingCloseTag  = "</think>"
	glm46vActionPrefix      = "Action:"
	glm46vActionInputPrefix = "Action Input:"
)

func (p *GLM46VParser) HasToolSupport() bool {
	return true
}

func (p *GLM46VParser) HasThinkingSupport() bool {
	return true
}

func (p *GLM46VParser) setInitialState(lastMessage *api.Message, tools []api.Tool, thinkValue *api.ThinkValue) {
	prefill := lastMessage != nil && lastMessage.Role == "assistant"

	// Check both model capability AND request preference
	thinkingEnabled := thinkValue != nil && thinkValue.Bool()

	// If tools are present, we don't start in thinking mode
	if len(tools) > 0 {
		p.state = GLM46VCollectingContent
		return
	}

	if !thinkingEnabled {
		p.state = GLM46VCollectingContent
		return
	}

	if prefill && lastMessage.Content != "" {
		p.state = GLM46VCollectingContent
		return
	}

	p.state = GLM46VCollectingThinking
}

func (p *GLM46VParser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	p.tools = tools
	p.err = nil
	p.setInitialState(lastMessage, tools, thinkValue)
	return tools
}

// Event types
type glm46vEvent interface {
	isGLM46VEvent()
}

type glm46vEventContent struct {
	content string
}

type glm46vEventThinkingContent struct {
	content string
}

type glm46vEventToolCall struct {
	toolCall api.ToolCall
}

func (glm46vEventContent) isGLM46VEvent()         {}
func (glm46vEventThinkingContent) isGLM46VEvent() {}
func (glm46vEventToolCall) isGLM46VEvent()        {}

func (p *GLM46VParser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	p.buffer.WriteString(s)
	events := p.parseEvents()

	// Check for critical errors
	if p.err != nil {
		return "", "", nil, p.err
	}

	var toolCalls []api.ToolCall
	var contentSb strings.Builder
	var thinkingSb strings.Builder
	for _, event := range events {
		switch event := event.(type) {
		case glm46vEventToolCall:
			toolCalls = append(toolCalls, event.toolCall)
		case glm46vEventThinkingContent:
			thinkingSb.WriteString(event.content)
		case glm46vEventContent:
			contentSb.WriteString(event.content)
		}
	}

	return contentSb.String(), thinkingSb.String(), toolCalls, nil
}

func (p *GLM46VParser) parseEvents() []glm46vEvent {
	var all []glm46vEvent

	keepLooping := true
	for keepLooping && p.err == nil {
		var events []glm46vEvent
		events, keepLooping = p.eat()
		if len(events) > 0 {
			all = append(all, events...)
		}
	}

	return all
}

func (p *GLM46VParser) eat() ([]glm46vEvent, bool) {
	var events []glm46vEvent
	bufStr := p.buffer.String()
	if bufStr == "" {
		return events, false
	}

	switch p.state {
	case GLM46VCollectingThinking:
		if strings.Contains(bufStr, glm46vThinkingCloseTag) {
			// thinking[</think>] -> content
			split := strings.SplitN(bufStr, glm46vThinkingCloseTag, 2)
			thinking := split[0]
			thinking = strings.TrimRightFunc(thinking, unicode.IsSpace)

			remaining := split[1]
			remaining = strings.TrimLeftFunc(remaining, unicode.IsSpace)

			p.buffer.Reset()
			p.buffer.WriteString(remaining)
			p.state = GLM46VCollectingContent

			if len(thinking) > 0 {
				events = append(events, glm46vEventThinkingContent{content: thinking})
			}
			return events, true
		} else if overlapLen := overlap(bufStr, glm46vThinkingCloseTag); overlapLen > 0 {
			// partial </think>
			beforePartialTag := bufStr[:len(bufStr)-overlapLen]
			trailingLen := trailingWhitespaceLen(beforePartialTag)
			ambiguousStart := len(beforePartialTag) - trailingLen

			unambiguous := bufStr[:ambiguousStart]
			ambiguous := bufStr[ambiguousStart:]
			p.buffer.Reset()
			p.buffer.WriteString(ambiguous)
			if len(unambiguous) > 0 {
				events = append(events, glm46vEventThinkingContent{content: unambiguous})
			}
			return events, false
		} else {
			// otherwise it's thinking content
			whitespaceLen := trailingWhitespaceLen(bufStr)
			ambiguousStart := len(bufStr) - whitespaceLen

			unambiguous := bufStr[:ambiguousStart]
			ambiguous := bufStr[ambiguousStart:]
			p.buffer.Reset()
			p.buffer.WriteString(ambiguous)
			if len(unambiguous) > 0 {
				events = append(events, glm46vEventThinkingContent{content: unambiguous})
			}
			return events, false
		}

	case GLM46VCollectingContent:
		// Check which pattern appears first: thinking tag or tool call
		thinkIdx := strings.Index(bufStr, glm46vThinkingOpenTag)
		toolCallIdx := p.findToolCallStart(bufStr)

		// Determine which comes first
		var tagIdx int
		var nextState GLM46VParserState
		var skipLen int

		if thinkIdx >= 0 && (toolCallIdx < 0 || thinkIdx < toolCallIdx) {
			tagIdx = thinkIdx
			skipLen = len(glm46vThinkingOpenTag)
			nextState = GLM46VCollectingThinking
		} else if toolCallIdx >= 0 {
			tagIdx = toolCallIdx
			skipLen = 0 // Tool call parsing handles its own prefix
			nextState = GLM46VCollectingToolCall
		} else {
			tagIdx = -1
		}

		if tagIdx >= 0 {
			// Found a tag - emit content before it
			before := bufStr[:tagIdx]
			if before != "" {
				events = append(events, glm46vEventContent{content: before})
			}

			// Move past the tag
			remaining := bufStr[tagIdx+skipLen:]
			p.buffer.Reset()
			p.buffer.WriteString(remaining)
			p.state = nextState

			logutil.Trace("glm46v: found tag", "before", before, "nextState", nextState)
			return events, true
		}

		// No complete tag found - check for partial tags at end
		thinkingOverlap := overlap(bufStr, glm46vThinkingOpenTag)
		actionOverlap := overlap(bufStr, glm46vActionPrefix)
		codeBlockOverlap := overlap(bufStr, "```")
		maxOverlap := max(thinkingOverlap, max(actionOverlap, codeBlockOverlap))

		if maxOverlap > 0 {
			// Hold back potential partial tag
			content := bufStr[:len(bufStr)-maxOverlap]
			if content != "" {
				events = append(events, glm46vEventContent{content: content})
			}
			// Keep the potential partial tag in buffer
			p.buffer.Reset()
			p.buffer.WriteString(bufStr[len(bufStr)-maxOverlap:])
			logutil.Trace("glm46v: holding potential partial tag", "overlap", maxOverlap)
			return events, false
		}

		// No partial tag - emit everything
		if bufStr != "" {
			events = append(events, glm46vEventContent{content: bufStr})
			p.buffer.Reset()
		}
		return events, false

	case GLM46VCollectingToolCall:
		// Try to parse tool call from buffer
		toolCall, remaining, found := p.parseToolCall(bufStr)
		if found {
			events = append(events, glm46vEventToolCall{toolCall: toolCall})
			remaining = strings.TrimLeftFunc(remaining, unicode.IsSpace)
			p.buffer.Reset()
			p.buffer.WriteString(remaining)
			p.state = GLM46VCollectingContent
			return events, len(remaining) > 0
		}

		// Check if we have enough to determine there's no complete tool call
		// If buffer doesn't seem to be forming a valid tool call, go back to content
		if !p.mightBeToolCall(bufStr) {
			p.state = GLM46VCollectingContent
			return events, true
		}

		// Still collecting tool call content
		return events, false
	}

	return events, false
}

// findToolCallStart finds the index where a tool call pattern starts
// Returns -1 if no tool call pattern found
func (p *GLM46VParser) findToolCallStart(bufStr string) int {
	// Check for markdown code block with ReAct format
	if codeBlockIdx := strings.Index(bufStr, "```"); codeBlockIdx >= 0 {
		afterBlock := bufStr[codeBlockIdx:]
		// Check if it contains Action: pattern
		if strings.Contains(afterBlock, glm46vActionPrefix) {
			return codeBlockIdx
		}
	}

	// Check for ReAct format: "Action: function_name"
	for _, tool := range p.tools {
		patterns := []string{
			glm46vActionPrefix + " " + tool.Function.Name,
			"\n" + glm46vActionPrefix + " " + tool.Function.Name,
		}
		for _, pattern := range patterns {
			if idx := strings.Index(bufStr, pattern); idx >= 0 {
				// Adjust index to start at "Action:"
				if bufStr[idx] == '\n' {
					return idx + 1
				}
				return idx
			}
		}
	}

	// Check for native format: function_name\n{
	for _, tool := range p.tools {
		patterns := []string{
			"\n" + tool.Function.Name + "\n{",
			"\n" + tool.Function.Name + "\r\n{",
		}
		for _, pattern := range patterns {
			if idx := strings.Index(bufStr, pattern); idx >= 0 {
				return idx + 1 // Skip the leading newline
			}
		}
		// Check if starts with tool name
		if strings.HasPrefix(bufStr, tool.Function.Name+"\n{") {
			return 0
		}
	}

	return -1
}

// mightBeToolCall checks if the buffer content could potentially become a tool call
func (p *GLM46VParser) mightBeToolCall(bufStr string) bool {
	// If we see Action: but no complete tool call yet, keep waiting
	if strings.Contains(bufStr, glm46vActionPrefix) {
		return true
	}

	// If we see a tool name but JSON isn't complete
	for _, tool := range p.tools {
		if strings.Contains(bufStr, tool.Function.Name) {
			// Check if we have incomplete JSON (unbalanced braces)
			braceCount := 0
			for _, c := range bufStr {
				if c == '{' {
					braceCount++
				} else if c == '}' {
					braceCount--
				}
			}
			if braceCount > 0 {
				return true // Still waiting for closing braces
			}
		}
	}

	return false
}

// parseToolCall attempts to parse a complete tool call from the buffer
func (p *GLM46VParser) parseToolCall(bufStr string) (api.ToolCall, string, bool) {
	// Try markdown code block format first
	if toolCall, remaining, found := p.parseMarkdownToolCall(bufStr); found {
		return toolCall, remaining, true
	}

	// Try ReAct format: Action: function_name\nAction Input: {json}
	if toolCall, remaining, found := p.parseReActToolCall(bufStr); found {
		return toolCall, remaining, true
	}

	// Try native format: function_name\n{json}
	if toolCall, remaining, found := p.parseNativeToolCall(bufStr); found {
		return toolCall, remaining, true
	}

	return api.ToolCall{}, "", false
}

// parseMarkdownToolCall extracts tool call from markdown code block format
// Format: ```\nAction: function_name\nAction Input: {json}\n```
func (p *GLM46VParser) parseMarkdownToolCall(bufStr string) (api.ToolCall, string, bool) {
	// Look for opening ```
	codeBlockStarts := []string{"```\n", "```json\n", "```\r\n", "```json\r\n"}

	for _, startMarker := range codeBlockStarts {
		startIdx := strings.Index(bufStr, startMarker)
		if startIdx < 0 {
			continue
		}

		codeStart := startIdx + len(startMarker)
		remaining := bufStr[codeStart:]

		// Find closing ```
		endIdx := -1
		for _, closer := range []string{"\n```", "\r\n```"} {
			if idx := strings.Index(remaining, closer); idx >= 0 {
				if endIdx < 0 || idx < endIdx {
					endIdx = idx
				}
			}
		}

		if endIdx < 0 {
			continue // No closing tag yet
		}

		codeContent := remaining[:endIdx]
		afterBlock := remaining[endIdx:]
		// Skip past the closing ```
		for _, closer := range []string{"\n```", "\r\n```"} {
			if strings.HasPrefix(afterBlock, closer) {
				afterBlock = afterBlock[len(closer):]
				break
			}
		}

		logutil.Trace("glm46v: found markdown code block", "content", codeContent)

		// Parse ReAct format inside code block
		for _, tool := range p.tools {
			actionPattern := glm46vActionPrefix + " " + tool.Function.Name
			if idx := strings.Index(codeContent, actionPattern); idx >= 0 {
				afterAction := codeContent[idx+len(actionPattern):]
				afterAction = strings.TrimLeft(afterAction, " \t\r\n")

				if strings.HasPrefix(afterAction, glm46vActionInputPrefix) {
					argsStr := strings.TrimPrefix(afterAction, glm46vActionInputPrefix)
					argsStr = strings.TrimSpace(argsStr)

					toolCall, err := p.createToolCall(tool.Function.Name, argsStr)
					if err != nil {
						slog.Warn("glm46v: failed to parse markdown tool call", "error", err)
						continue
					}
					return toolCall, afterBlock, true
				}
			}
		}
	}

	return api.ToolCall{}, "", false
}

// parseReActToolCall extracts tool call from ReAct format
// Format: Action: function_name\nAction Input: {json}
func (p *GLM46VParser) parseReActToolCall(bufStr string) (api.ToolCall, string, bool) {
	for _, tool := range p.tools {
		patterns := []string{
			glm46vActionPrefix + " " + tool.Function.Name + "\n",
			glm46vActionPrefix + " " + tool.Function.Name + "\r\n",
		}

		for _, pattern := range patterns {
			idx := strings.Index(bufStr, pattern)
			if idx < 0 {
				continue
			}

			afterAction := bufStr[idx+len(pattern):]
			afterAction = strings.TrimLeft(afterAction, " \t")

			if !strings.HasPrefix(afterAction, glm46vActionInputPrefix) {
				continue
			}

			argsStart := strings.TrimPrefix(afterAction, glm46vActionInputPrefix)
			argsStart = strings.TrimLeft(argsStart, " \t")

			// Extract JSON object
			if len(argsStart) == 0 || argsStart[0] != '{' {
				continue
			}

			argsStr, remaining, found := extractJSONObject(argsStart)
			if !found {
				continue
			}

			toolCall, err := p.createToolCall(tool.Function.Name, argsStr)
			if err != nil {
				slog.Warn("glm46v: failed to parse ReAct tool call", "error", err)
				continue
			}

			logutil.Trace("glm46v: parsed ReAct tool call", "name", tool.Function.Name, "args", argsStr)
			return toolCall, remaining, true
		}
	}

	return api.ToolCall{}, "", false
}

// parseNativeToolCall extracts tool call from native format
// Format: function_name\n{json}
func (p *GLM46VParser) parseNativeToolCall(bufStr string) (api.ToolCall, string, bool) {
	for _, tool := range p.tools {
		patterns := []string{
			tool.Function.Name + "\n{",
			tool.Function.Name + "\r\n{",
		}

		for _, pattern := range patterns {
			idx := strings.Index(bufStr, pattern)
			if idx < 0 {
				continue
			}

			// Find where JSON starts
			jsonStart := idx + len(pattern) - 1 // Point to the '{'

			argsStr, remaining, found := extractJSONObject(bufStr[jsonStart:])
			if !found {
				continue
			}

			toolCall, err := p.createToolCall(tool.Function.Name, argsStr)
			if err != nil {
				slog.Warn("glm46v: failed to parse native tool call", "error", err)
				continue
			}

			logutil.Trace("glm46v: parsed native tool call", "name", tool.Function.Name, "args", argsStr)
			return toolCall, remaining, true
		}
	}

	return api.ToolCall{}, "", false
}

// extractJSONObject extracts a complete JSON object from the start of a string
// Returns the JSON string, remaining content, and whether extraction was successful
func extractJSONObject(s string) (string, string, bool) {
	if len(s) == 0 || s[0] != '{' {
		return "", "", false
	}

	braceCount := 0
	inString := false
	escape := false

	for i, c := range s {
		if escape {
			escape = false
			continue
		}

		if c == '\\' && inString {
			escape = true
			continue
		}

		if c == '"' {
			inString = !inString
			continue
		}

		if inString {
			continue
		}

		if c == '{' {
			braceCount++
		} else if c == '}' {
			braceCount--
			if braceCount == 0 {
				return s[:i+1], s[i+1:], true
			}
		}
	}

	return "", "", false // Incomplete JSON
}

// createToolCall creates an api.ToolCall from function name and args string
func (p *GLM46VParser) createToolCall(functionName string, argsStr string) (api.ToolCall, error) {
	// Validate tool exists
	tool := p.findToolByName(functionName)
	if tool == nil {
		availableTools := make([]string, len(p.tools))
		for i, t := range p.tools {
			availableTools[i] = t.Function.Name
		}
		p.err = fmt.Errorf("model called unknown tool %q - available tools: %v (ensure tools are provided in API request)", functionName, availableTools)
		slog.Error("GLM46V model attempted to call unregistered tool",
			"tool", functionName,
			"available_tools", availableTools,
			"recommendation", "ensure tools array includes this tool in API request")
		return api.ToolCall{}, p.err
	}

	toolCall := api.ToolCall{
		Function: api.ToolCallFunction{
			Name: tool.Function.Name,
		},
	}

	if argsStr != "" {
		// Convert Python-style single quotes to JSON double quotes
		argsStr = convertPythonToJSON(argsStr)
		if err := json.Unmarshal([]byte(argsStr), &toolCall.Function.Arguments); err != nil {
			return api.ToolCall{}, fmt.Errorf("failed to parse tool call arguments: %w", err)
		}
	}

	return toolCall, nil
}

func (p *GLM46VParser) findToolByName(name string) *api.Tool {
	name = strings.TrimSpace(name)
	for i := range p.tools {
		if p.tools[i].Function.Name == name {
			return &p.tools[i]
		}
	}
	return nil
}

// convertPythonToJSON converts Python-style dict strings to valid JSON
// Handles: {'key': 'value'} -> {"key": "value"}
func convertPythonToJSON(s string) string {
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
