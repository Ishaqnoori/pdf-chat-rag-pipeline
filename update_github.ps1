param (
    [string]$CommitMessage = "Update files"
)

# Refuse to push anything that looks like a secret (item 2)
$patterns = @(
    'AIza[0-9A-Za-z_\-]{35}',          # Google / Firebase keys
    'gsk_[0-9A-Za-z]{20,}',            # Groq keys
    'sk-[0-9A-Za-z_\-]{20,}',          # OpenAI-style keys
    '-----BEGIN [A-Z ]*PRIVATE KEY-----'
)

Write-Host "Staging files..."
git add .

$tracked = git diff --cached --name-only
if ($tracked -match '(^|/)\.env$' -or $tracked -match 'secrets\.toml$' -or $tracked -match 'service-account.*\.json$') {
    Write-Host "STOP: .env, secrets.toml or a service-account key is staged. Check .gitignore." -ForegroundColor Red
    git reset -q
    exit 1
}
foreach ($file in $tracked) {
    if (Test-Path $file) {
        foreach ($p in $patterns) {
            if (Select-String -Path $file -Pattern $p -Quiet) {
                Write-Host "STOP: possible secret in $file. Remove it before pushing." -ForegroundColor Red
                git reset -q
                exit 1
            }
        }
    }
}

Write-Host "Committing changes..."
git commit -m $CommitMessage

Write-Host "Pushing to GitHub..."
git push origin master

Write-Host "Done!"
