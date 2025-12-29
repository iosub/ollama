package renderers

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestGLM46VRenderer_Render(t *testing.T) {
	renderer := &GLM46VRenderer{}
	renderer.enableThinking = false
	renderer.useImgTags = false

	tool := api.Tool{}
	tool.Function.Name = "dbCheckData"
	tools := []api.Tool{tool}

	messages := []api.Message{
		{
			Role:    "user",
			Content: "Check the invoice",
		},
		{
			Role: "assistant",
			ToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "dbCheckData",
						Arguments: map[string]any{
							"fieldName": "Items[0].albaran",
							"value":     "C25-22564",
						},
					},
				},
			},
		},
		{
			Role:    "tool",
			Content: "OK",
		},
	}

	rendered, err := renderer.Render(messages, tools, nil)
	if err != nil {
		t.Fatalf("Render failed: %v", err)
	}
	panic(rendered)

	expectedTools := "<tools>\n{\"name\":\"dbCheckData\",\"parameters\":null}\n</tools>"
	if !strings.Contains(rendered, expectedTools) {
		t.Errorf("Expected tool signatures not found.\nExpected: %s\nGot: %s", expectedTools, rendered)
	}

	expectedToolCall := "<tool_call>dbCheckData\n<arg_key>fieldName</arg_key>\n<arg_value>Items[0].albaran</arg_value>\n<arg_key>value</arg_key>\n<arg_value>C25-22564</arg_value>\n</tool_call>"
	if !strings.Contains(rendered, expectedToolCall) {
		t.Errorf("Expected tool call format not found.\nExpected: %s\nGot: %s", expectedToolCall, rendered)
	}

	expectedObservation := "<|observation|>\n<tool_response>\nOK\n</tool_response>"
	if !strings.Contains(rendered, expectedObservation) {
		t.Errorf("Expected observation format not found.\nExpected: %s\nGot: %s", expectedObservation, rendered)
	}

	expectedThink := "<think></think>"
	if !strings.Contains(rendered, expectedThink) {
		t.Errorf("Expected empty think tags not found.\nExpected: %s\nGot: %s", expectedThink, rendered)
	}
}
