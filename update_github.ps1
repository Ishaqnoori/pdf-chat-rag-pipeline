param (
    [string]$CommitMessage = "Update files"
)

Write-Host "Staging files..."
git add .

Write-Host "Committing changes..."
git commit -m $CommitMessage

Write-Host "Pushing to GitHub..."
git push origin master

Write-Host "Done!"
