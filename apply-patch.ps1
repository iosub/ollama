param(
    [Parameter(Mandatory=$true)]
    [string]$PatchFile,
    
    [Parameter(Mandatory=$true)]
    [string]$WorkDir,
    
    [Parameter(Mandatory=$true)]
    [string]$MarkerFile
)

try {
    $patchPath = Resolve-Path $PatchFile
    Write-Host "Applying patch: $($patchPath.Name)"
    
    # Configure git user for am command
    $env:GIT_AUTHOR_NAME = "nobody"
    $env:GIT_AUTHOR_EMAIL = "nobody@example.com"
    $env:GIT_COMMITTER_NAME = "nobody"
    $env:GIT_COMMITTER_EMAIL = "nobody@example.com"
    
    # Apply the patch
    & git -C $WorkDir am -3 $patchPath
    
    if ($LASTEXITCODE -eq 0) {
        # Create marker file
        New-Item -ItemType File -Path $MarkerFile -Force | Out-Null
        Write-Host "Patch applied successfully"
        exit 0
    } else {
        # Check if there are merge conflicts
        $status = & git -C $WorkDir status
        if ($status -like "*both modified*") {
            Write-Host "Merge conflicts detected, attempting auto-resolution..."
            
            # Get conflicted files
            $conflicted = @(git -C $WorkDir diff --name-only --diff-filter=U)
            
            if ($conflicted.Count -gt 0) {
                # Use 'theirs' strategy for conflict resolution
                foreach ($file in $conflicted) {
                    Write-Host "Resolving: $file (using their version)"
                    & git -C $WorkDir checkout --theirs $file
                    & git -C $WorkDir add $file
                }
                
                # Continue am
                & git -C $WorkDir am --continue
                
                if ($LASTEXITCODE -eq 0) {
                    New-Item -ItemType File -Path $MarkerFile -Force | Out-Null
                    Write-Host "Patch applied successfully after conflict resolution"
                    exit 0
                }
            }
        }
        
        Write-Error "Patch failed. Resolve any conflicts then continue."
        Write-Host "1. Run 'git -C $WorkDir am --continue'"
        Write-Host "2. Run 'make -f Makefile.sync format-patches'"
        Write-Host "3. Run 'make -f Makefile.sync clean apply-patches'"
        exit 1
    }
}
catch {
    Write-Error "Error applying patch: $_"
    exit 1
}
