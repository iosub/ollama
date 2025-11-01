param(
  [ValidateSet('apply','clear')]
  [string]$Action = 'apply'
)

$ErrorActionPreference = 'Stop'

function Get-RepoRoot {
  $root = (git rev-parse --show-toplevel) 2>$null
  if (-not $root) { throw 'No estás dentro de un repo Git' }
  return $root
}

Push-Location (Get-RepoRoot)
try {
  if (-not (Test-Path 'Z_Iosu')) { Write-Host 'No existe Z_Iosu en el repo'; exit 0 }

  $files = git ls-files Z_Iosu
  if (-not $files) { Write-Host 'No hay ficheros rastreados dentro de Z_Iosu'; exit 0 }

  switch ($Action) {
    'apply' {
      $files | ForEach-Object { git update-index --skip-worktree -- $_ }
      Write-Host 'Marcado skip-worktree aplicado a Z_Iosu'
    }
    'clear' {
      $files | ForEach-Object { git update-index --no-skip-worktree -- $_ }
      Write-Host 'Marcado skip-worktree eliminado de Z_Iosu'
    }
  }

  # Verificación
  $ls = git ls-files -v Z_Iosu
  $countSkip = ($ls | Select-String '^[sS] ').Count
  if ($Action -eq 'apply') {
    Write-Host "Archivos con 'S' (skip-worktree): $countSkip"
  } else {
    Write-Host "Archivos con 'S' tras clear: $countSkip"
  }
} finally {
  Pop-Location
}
