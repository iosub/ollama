param(
    [Parameter(Mandatory=$true)]
    [string]$WorkDir
)

try {
    # Get the list of files with conflicts
    $conflicts = @(git -C $WorkDir diff --name-only --diff-filter=U)
    
    if ($conflicts.Count -eq 0) {
        Write-Host "No conflicts found"
        exit 0
    }
    
    Write-Host "Found $($conflicts.Count) conflicted file(s)"
    
    # For each conflicted file, we'll use a simple strategy:
    # - Take theirs (the patched version from the original branch)
    foreach ($file in $conflicts) {
        Write-Host "Resolving conflict in: $file"
        # Use 'theirs' resolution (the patch's version)
        & git -C $WorkDir checkout --theirs $file
        & git -C $WorkDir add $file
    }
    
    # Continue the am process
    Write-Host "Continuing git am after conflict resolution..."
    & git -C $WorkDir am --continue
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Patches applied successfully after conflict resolution"
        exit 0
    } else {
        Write-Error "Failed to continue patching after conflict resolution"
        exit 1
    }
}
catch {
    Write-Error "Error resolving conflicts: $_"
    exit 1
}
