#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Interactive Qwen3VL Model Downloader and Setup for Ollama
.DESCRIPTION
    This script downloads Qwen3VL models from HuggingFace, creates appropriate Modelfiles,
    and sets up everything automatically for use with Ollama.
.NOTES
    Author: AI Assistant
    Version: 1.0
    Compatible with: Windows PowerShell 5.1+ and PowerShell Core 7+
#>

# Set error handling
$ErrorActionPreference = "Stop"

# Define available models
$Models = @{
    "1" = @{
        Name = "Qwen3-VL-2B-Instruct"
        Size = "2.6 GB total"
        TextFile = "Qwen3-VL-2B-Instruct-Q8_0.gguf"
        VisionFile = "mmproj-Qwen3-VL-2B-Instruct.gguf"
        Description = "Lightweight model, good for testing and low-resource environments"
    }
    "2" = @{
        Name = "Qwen3-VL-4B-Instruct"
        Size = "5.1 GB total"
        TextFile = "Qwen3-VL-4B-Instruct-Q8_0.gguf"
        VisionFile = "mmproj-Qwen3-VL-4B-Instruct.gguf"
        Description = "Balanced performance and resource usage"
    }
    "3" = @{
        Name = "Qwen3-VL-8B-Instruct"
        Size = "9.9 GB total"
        TextFile = "Qwen3-VL-8B-Instruct-Q8_0.gguf"
        VisionFile = "mmproj-Qwen3-VL-8B-Instruct.gguf"
        Description = "High-quality model with best performance"
    }
    "4" = @{
        Name = "Qwen3-VL-4B-Thinking"
        Size = "5.1 GB total"
        TextFile = "Qwen3-VL-4B-Thinking-Q8_0.gguf"
        VisionFile = "mmproj-Qwen3-VL-4B-Thinking.gguf"
        Description = "Enhanced reasoning capabilities, 4B parameters"
    }
    "5" = @{
        Name = "Qwen3-VL-8B-Thinking"
        Size = "9.9 GB total"
        TextFile = "Qwen3-VL-8B-Thinking-Q8_0.gguf"
        VisionFile = "mmproj-Qwen3-VL-8B-Thinking.gguf"
        Description = "Enhanced reasoning capabilities, 8B parameters"
    }
}

# Base URLs
$BaseURL = "https://huggingface.co/bonswouar/unsloth-Qwen3-VL-GGUF/resolve/main"

function Show-Banner {
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host "           Qwen3VL Model Downloader for Ollama" -ForegroundColor Yellow
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host ""
}

function Show-Models {
    Write-Host "Available Qwen3VL Models:" -ForegroundColor Green
    Write-Host ""
    
    foreach ($key in $Models.Keys | Sort-Object) {
        $model = $Models[$key]
        Write-Host "[$key] " -ForegroundColor Yellow -NoNewline
        Write-Host "$($model.Name)" -ForegroundColor White -NoNewline
        Write-Host " ($($model.Size))" -ForegroundColor Gray
        Write-Host "    $($model.Description)" -ForegroundColor DarkGray
        Write-Host ""
    }
}

function Get-UserChoice {
    while ($true) {
        Write-Host "Which model would you like to download? " -ForegroundColor Cyan -NoNewline
        Write-Host "Options 1-5: " -ForegroundColor Yellow -NoNewline
        $choice = Read-Host
        
        if ($Models.ContainsKey($choice)) {
            return $choice
        } else {
            Write-Host "Invalid choice. Please select a number between 1-5." -ForegroundColor Red
        }
    }
}

function Download-File {
    param(
        [string]$Url,
        [string]$DestinationPath,
        [string]$FileName
    )
    
    Write-Host "Downloading $FileName..." -ForegroundColor Yellow
    
    try {
        # Check if file already exists
        if (Test-Path $DestinationPath) {
            $fileSize = (Get-Item $DestinationPath).Length / 1MB
            Write-Host "File already exists: $FileName ($([math]::Round($fileSize, 2)) MB)" -ForegroundColor Green
            
            Write-Host "Do you want to re-download it? " -ForegroundColor Yellow -NoNewline
            Write-Host "[y/N]: " -ForegroundColor Cyan -NoNewline
            $redownload = Read-Host
            
            if ($redownload -and ($redownload.ToLower() -eq "y" -or $redownload.ToLower() -eq "yes")) {
                Write-Host "Removing existing file..." -ForegroundColor Yellow
                Remove-Item $DestinationPath -Force
            } else {
                Write-Host "Using existing file." -ForegroundColor Green
                return $true
            }
        }
        
        # Use Invoke-WebRequest for download
        Invoke-WebRequest -Uri $Url -OutFile $DestinationPath -UseBasicParsing
        
        if (Test-Path $DestinationPath) {
            $fileSize = (Get-Item $DestinationPath).Length / 1MB
            Write-Host "Downloaded $FileName ($([math]::Round($fileSize, 2)) MB)" -ForegroundColor Green
            return $true
        } else {
            Write-Host "Failed to download $FileName" -ForegroundColor Red
            return $false
        }
    } catch {
        Write-Host "Error downloading $FileName : $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

function Create-Modelfile {
    param(
        [string]$ModelName,
        [string]$TextFile,
        [string]$VisionFile,
        [string]$OutputPath
    )
    
    Write-Host "Creating Modelfile for $ModelName..." -ForegroundColor Yellow
    
    try {
        $templateContent = @'
{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{- range .Messages }}{{- if eq .Role "user" }}<|im_start|>user
{{- if .Images }}
{{- range .Images }}
<|vision_start|><|image_pad|><|vision_end|>
{{- end }}
{{- end }}
{{- if .Tools }}
{{- range .Tools }}
<|tool_call|>{{ .Name }}({{ .Parameters | json }})<|/tool_call|>
{{- end }}
{{- end }}
{{ .Content }}<|im_end|>
{{- else if eq .Role "assistant" }}<|im_start|>assistant
{{- if .ToolCalls }}
{{- range .ToolCalls }}
<|tool_call|>{{ .Function.Name }}({{ .Function.Arguments | json }})<|/tool_call|>
{{- end }}
{{- end }}
{{- if .Thinking }}
<|thinking|>
{{ .Thinking }}
</thinking>
{{- end }}
{{ .Content }}<|im_end|>
{{- else if eq .Role "tool" }}<|im_start|>tool
<|tool_result|>{{ .Content }}<|/tool_result|><|im_end|>
{{- else }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{- end }}
{{- end }}<|im_start|>assistant
'@

        $modelfileContent = @"
FROM ./$TextFile
ADAPTER ./$VisionFile

TEMPLATE """$templateContent"""

PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|thinking|>"
PARAMETER stop "</thinking>"
PARAMETER stop "<|tool_call|>"
PARAMETER stop "<|/tool_call|>"
PARAMETER stop "<|tool_result|>"
PARAMETER stop "<|/tool_result|>"
PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER top_k 50
PARAMETER repeat_penalty 1.1

SYSTEM "You are Qwen3-VL, a vision-language AI assistant that can analyze images and use tools. When given an image, describe what you see. When given tools, use them appropriately to help answer questions."
"@

        $utf8NoBom = New-Object System.Text.UTF8Encoding $false
        [System.IO.File]::WriteAllText($OutputPath, $modelfileContent, $utf8NoBom)
        Write-Host "Modelfile created successfully" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "Error creating Modelfile: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

function Register-Model {
    param(
        [string]$ModelName,
        [string]$ModelfilePath
    )
    
    Write-Host "Registering model with Ollama..." -ForegroundColor Yellow
    
    try {
        $ollamaName = $ModelName.ToLower().Replace("-", "_")
        
        # Check if ollama is available
        $ollamaCheck = Get-Command ollama -ErrorAction SilentlyContinue
        if (-not $ollamaCheck) {
            Write-Host "Ollama not found in PATH. Please ensure Ollama is installed and accessible." -ForegroundColor Yellow
            return $false
        }
        
        # Create the model
        Write-Host "Creating model '$ollamaName'..." -ForegroundColor Cyan
        Write-Host "Command: ollama create $ollamaName -f $ModelfilePath" -ForegroundColor Gray
        $createResult = & ollama create $ollamaName -f $ModelfilePath 2>&1
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Model '$ollamaName' registered successfully!" -ForegroundColor Green
            Write-Host "You can now use: ollama run $ollamaName" -ForegroundColor Cyan
            return $true
        } else {
            Write-Host "Failed to register model. Error: $createResult" -ForegroundColor Red
            return $false
        }
    } catch {
        Write-Host "Error registering model: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

function Confirm-Action {
    param([string]$Message)
    
    while ($true) {
        Write-Host "$Message " -ForegroundColor Yellow -NoNewline
        Write-Host "(Y/n): " -ForegroundColor Cyan -NoNewline
        $response = Read-Host
        
        if ([string]::IsNullOrWhiteSpace($response) -or $response -match "^[Yy]") {
            return $true
        } elseif ($response -match "^[Nn]") {
            return $false
        } else {
            Write-Host "Please answer Y or n" -ForegroundColor Red
        }
    }
}

# ============================================================================
# MAIN SCRIPT EXECUTION
# ============================================================================

# Show banner and model options
Show-Banner
Show-Models

# Get user choice
$selectedModel = Get-UserChoice
$model = $Models[$selectedModel]

Write-Host ""
Write-Host "Selected Model Details:" -ForegroundColor Green
Write-Host "   Name: $($model.Name)" -ForegroundColor White
Write-Host "   Size: $($model.Size)" -ForegroundColor Gray
Write-Host "   Description: $($model.Description)" -ForegroundColor DarkGray
Write-Host ""

# Confirm download
if (-not (Confirm-Action "Do you want to proceed with downloading this model?")) {
    Write-Host "Download cancelled by user." -ForegroundColor Yellow
    exit 0
}

# Setup directories
$currentDir = Get-Location
$modelDir = Join-Path $currentDir $model.Name
$textFilePath = Join-Path $modelDir $model.TextFile
$visionFilePath = Join-Path $modelDir $model.VisionFile
$modelfilePath = Join-Path $modelDir "Modelfile"

# Create model directory
Write-Host "Creating directory: $modelDir" -ForegroundColor Cyan
if (-not (Test-Path $modelDir)) {
    New-Item -ItemType Directory -Path $modelDir -Force | Out-Null
}

# Change to model directory
Set-Location $modelDir

# Download files
Write-Host ""
Write-Host "Starting downloads..." -ForegroundColor Green

$textUrl = "$BaseURL/$($model.TextFile)"
$visionUrl = "$BaseURL/$($model.VisionFile)"

$downloadSuccess = $true

# Download text model
if (-not (Test-Path $model.TextFile)) {
    $downloadSuccess = $downloadSuccess -and (Download-File $textUrl $model.TextFile $model.TextFile)
} else {
    Write-Host "Text model already exists: $($model.TextFile)" -ForegroundColor Green
}

# Download vision model
if (-not (Test-Path $model.VisionFile)) {
    $downloadSuccess = $downloadSuccess -and (Download-File $visionUrl $model.VisionFile $model.VisionFile)
} else {
    Write-Host "Vision model already exists: $($model.VisionFile)" -ForegroundColor Green
}

if (-not $downloadSuccess) {
    Write-Host "Some downloads failed. Please check the errors above." -ForegroundColor Red
    Set-Location $currentDir
    exit 1
}

# Create Modelfile
Write-Host ""
if (Create-Modelfile -ModelName $model.Name -TextFile $model.TextFile -VisionFile $model.VisionFile -OutputPath "Modelfile") {
    Write-Host "Modelfile created: $modelfilePath" -ForegroundColor Green
} else {
    Write-Host "Failed to create Modelfile" -ForegroundColor Red
    Set-Location $currentDir
    exit 1
}

# Register with Ollama
Write-Host ""
if (Confirm-Action "Do you want to register this model with Ollama now?") {
    if (Register-Model -ModelName $model.Name -ModelfilePath "Modelfile") {
        Write-Host ""
        Write-Host "Setup completed successfully!" -ForegroundColor Green
        Write-Host "Model downloaded and registered" -ForegroundColor Green
        Write-Host "Ready to use with Ollama" -ForegroundColor Green
    } else {
        Write-Host "Model downloaded but not registered. You can register it manually later." -ForegroundColor Yellow
    }
} else {
    Write-Host "Model downloaded but not registered with Ollama." -ForegroundColor Yellow
    Write-Host "To register later, run: ollama create $($model.Name.ToLower()) -f Modelfile" -ForegroundColor Cyan
}

# Return to original directory
Set-Location $currentDir

Write-Host ""
Write-Host "Files are located in: $modelDir" -ForegroundColor Cyan
Write-Host "Script completed!" -ForegroundColor Green